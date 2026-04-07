"""
MedPhi Core — all logic, no UI dependencies.
Imported by both medphi.py (Streamlit) and medphi_mcp_server.py (MCP).
"""

import json
import re
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import fitz  # pymupdf
from PIL import Image
from sentence_transformers import SentenceTransformer
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

try:
    import pytesseract
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

# ─── Paths ────────────────────────────────────────────────────────────────────

VAULT_DIR = Path.home() / ".medphi"
VAULT_CONFIG = VAULT_DIR / "vault_config.json"
DOCUMENTS_INDEX = VAULT_DIR / "documents.json"
CHROMA_DIR = VAULT_DIR / "chroma_db"

OCR_TEXT_THRESHOLD = 100


# ─── Vault Config ─────────────────────────────────────────────────────────────

def load_vault_config() -> dict:
    VAULT_DIR.mkdir(exist_ok=True)
    if VAULT_CONFIG.exists():
        with open(VAULT_CONFIG) as f:
            return json.load(f)
    config = {"date_offset_days": 730, "created": datetime.now().isoformat()}
    with open(VAULT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)
    return config


def save_vault_config(config: dict):
    VAULT_DIR.mkdir(exist_ok=True)
    with open(VAULT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)


# ─── Document Index ───────────────────────────────────────────────────────────

def load_document_index() -> dict:
    if DOCUMENTS_INDEX.exists():
        with open(DOCUMENTS_INDEX) as f:
            return json.load(f)
    return {}


def register_document(doc_id: str, metadata: dict):
    index = load_document_index()
    index[doc_id] = metadata
    VAULT_DIR.mkdir(exist_ok=True)
    with open(DOCUMENTS_INDEX, "w") as f:
        json.dump(index, f, indent=2)


def unregister_document(doc_id: str):
    index = load_document_index()
    index.pop(doc_id, None)
    with open(DOCUMENTS_INDEX, "w") as f:
        json.dump(index, f, indent=2)


# ─── Custom Medical Recognizers ───────────────────────────────────────────────

mrn_pattern = Pattern(
    name="mrn_pattern",
    regex=r"(?i)(mrn|medical record|patient id|account|visit id)[:\s]*([A-Z0-9-]{4,12})",
    score=0.95,
)
npi_pattern = Pattern(
    name="npi_pattern",
    regex=r"\b[12]\d{9}\b",
    score=0.8,
)
mrn_recognizer = PatternRecognizer(
    supported_entity="MEDICAL_RECORD_NUMBER", patterns=[mrn_pattern]
)
npi_recognizer = PatternRecognizer(
    supported_entity="PROVIDER_ID", patterns=[npi_pattern]
)

# ─── Lazy Singletons (no @st.cache_resource — works in any context) ───────────

_analyzer = None
_anonymizer = None
_collection = None
_embedder = None


def get_analyzer() -> AnalyzerEngine:
    global _analyzer
    if _analyzer is None:
        engine = AnalyzerEngine()
        engine.registry.add_recognizer(mrn_recognizer)
        engine.registry.add_recognizer(npi_recognizer)
        _analyzer = engine
    return _analyzer


def get_anonymizer() -> AnonymizerEngine:
    global _anonymizer
    if _anonymizer is None:
        _anonymizer = AnonymizerEngine()
    return _anonymizer


def get_collection():
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection("medphi_vault")
    return _collection


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ─── Date Shifter ─────────────────────────────────────────────────────────────

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

_NUMERIC_DATE_RE = re.compile(
    r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b"
    r"|\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b"
)
_VERBOSE_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s+(\d{4})\b"
)


def _shift_numeric(m: re.Match, delta: timedelta) -> str:
    try:
        if m.group(1):
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            shifted = d + delta
            sep = "-" if "-" in m.group(0) else "/"
            return shifted.strftime(f"%Y{sep}%m{sep}%d")
        else:
            year = int(m.group(6))
            if year < 100:
                year += 2000
            d = datetime(year, int(m.group(4)), int(m.group(5)))
            shifted = d + delta
            sep = "-" if "-" in m.group(0) else "/"
            return shifted.strftime(f"%m{sep}%d{sep}%Y")
    except (ValueError, IndexError):
        return m.group(0)


def _shift_verbose(m: re.Match, delta: timedelta) -> str:
    try:
        month = MONTH_MAP[m.group(1)]
        d = datetime(int(m.group(3)), month, int(m.group(2)))
        shifted = d + delta
        return shifted.strftime("%B %d, %Y")
    except (ValueError, KeyError):
        return m.group(0)


def shift_dates(text: str, offset_days: int) -> str:
    delta = timedelta(days=offset_days)
    text = _NUMERIC_DATE_RE.sub(lambda m: _shift_numeric(m, delta), text)
    text = _VERBOSE_DATE_RE.sub(lambda m: _shift_verbose(m, delta), text)
    return text


# ─── PDF Extraction ───────────────────────────────────────────────────────────

def _ocr_pdf(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(pytesseract.image_to_string(img))
    return "\n\n".join(pages)


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, bool]:
    """Returns (text, used_ocr). Falls back to OCR for image-based PDFs."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    text = "\n\n".join(pages)

    if len(text.strip()) >= OCR_TEXT_THRESHOLD:
        return text, False

    if not _OCR_AVAILABLE:
        return text, False

    try:
        return _ocr_pdf(pdf_bytes), True
    except Exception:
        return text, False


# ─── De-identification Pipeline ───────────────────────────────────────────────

_OPERATORS = {
    "PERSON": OperatorConfig("replace", {"new_value": "[PATIENT]"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
    "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION]"}),
    # DATE_TIME excluded — dates are shifted, not erased, to preserve timeline
    "US_SSN": OperatorConfig("replace", {"new_value": "[SSN]"}),
    "US_PASSPORT": OperatorConfig("replace", {"new_value": "[PASSPORT]"}),
    "MEDICAL_LICENSE": OperatorConfig("replace", {"new_value": "[LICENSE]"}),
    "MEDICAL_RECORD_NUMBER": OperatorConfig("replace", {"new_value": "[MRN]"}),
    "PROVIDER_ID": OperatorConfig("replace", {"new_value": "[NPI]"}),
}
_ENTITIES = list(_OPERATORS.keys())


def deidentify(text: str, offset_days: int) -> tuple[str, list]:
    """Shift dates then strip PII. Returns (clean_text, analyzer_results)."""
    text = shift_dates(text, offset_days)
    analyzer = get_analyzer()
    anonymizer = get_anonymizer()
    results = analyzer.analyze(text=text, language="en", entities=_ENTITIES)
    anonymized = anonymizer.anonymize(
        text=text, analyzer_results=results, operators=_OPERATORS
    )
    return anonymized.text, results


# ─── Vector Store ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def store_document(doc_id: str, clean_text: str, metadata: dict) -> int:
    collection = get_collection()
    embedder = get_embedder()
    chunks = chunk_text(clean_text)
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {**metadata, "chunk_index": i, "doc_id": doc_id}
        for i in range(len(chunks))
    ]
    collection.upsert(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    register_document(doc_id, {**metadata, "chunk_count": len(chunks)})
    return len(chunks)


def delete_document(doc_id: str):
    collection = get_collection()
    collection.delete(where={"doc_id": doc_id})
    unregister_document(doc_id)


def query_vault(question: str, n_results: int = 5) -> list[dict]:
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []
    n_results = min(n_results, total)
    embedder = get_embedder()
    q_emb = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=n_results)
    return [
        {
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["documents"][0]))
    ]


# ─── Ingest from file path (used by MCP server) ───────────────────────────────

def ingest_pdf(file_path: str, label: str = "") -> dict:
    """
    Full ingest pipeline from a file path.
    Returns a result dict with doc_id, chunk_count, pii_count, used_ocr.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    pdf_bytes = path.read_bytes()
    config = load_vault_config()

    raw_text, used_ocr = extract_text_from_pdf(pdf_bytes)
    clean_text, findings = deidentify(raw_text, config["date_offset_days"])

    doc_id = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    metadata = {
        "filename": path.name,
        "label": label or path.name,
        "uploaded_at": datetime.now().isoformat(),
        "pii_entities_found": len(findings),
        "ocr": used_ocr,
    }

    n_chunks = store_document(doc_id, clean_text, metadata)
    return {
        "doc_id": doc_id,
        "label": metadata["label"],
        "chunk_count": n_chunks,
        "pii_entities_removed": len(findings),
        "used_ocr": used_ocr,
    }
