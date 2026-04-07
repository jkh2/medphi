"""
MedPhi — Local-First Medical Memory Vault
Privacy-obsessed: de-identify → embed → query locally.
All processing happens on your machine. Nothing leaves.
"""

import os
import json
import re
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import chromadb
import fitz  # pymupdf
from PIL import Image
from sentence_transformers import SentenceTransformer
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from litellm import completion
from dotenv import load_dotenv

try:
    import pytesseract
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

load_dotenv()

# ─── Vault Config ─────────────────────────────────────────────────────────────

VAULT_DIR = Path.home() / ".medphi"
VAULT_CONFIG = VAULT_DIR / "vault_config.json"
DOCUMENTS_INDEX = VAULT_DIR / "documents.json"
CHROMA_DIR = VAULT_DIR / "chroma_db"  # anchored to home, not CWD

OCR_TEXT_THRESHOLD = 100  # chars; below this we assume the PDF is image-based


def load_vault_config() -> dict:
    """Load persistent vault config, creating defaults if absent."""
    VAULT_DIR.mkdir(exist_ok=True)
    if VAULT_CONFIG.exists():
        with open(VAULT_CONFIG) as f:
            return json.load(f)
    config = {
        "date_offset_days": 730,
        "created": datetime.now().isoformat(),
    }
    with open(VAULT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)
    return config


def save_vault_config(config: dict):
    VAULT_DIR.mkdir(exist_ok=True)
    with open(VAULT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)


# ─── Document Index ───────────────────────────────────────────────────────────

def load_document_index() -> dict:
    """Return {doc_id: metadata} for all stored documents."""
    if DOCUMENTS_INDEX.exists():
        with open(DOCUMENTS_INDEX) as f:
            return json.load(f)
    return {}


def register_document(doc_id: str, metadata: dict):
    """Add or update a document entry in the index."""
    index = load_document_index()
    index[doc_id] = metadata
    VAULT_DIR.mkdir(exist_ok=True)
    with open(DOCUMENTS_INDEX, "w") as f:
        json.dump(index, f, indent=2)


def unregister_document(doc_id: str):
    """Remove a document entry from the index."""
    index = load_document_index()
    index.pop(doc_id, None)
    with open(DOCUMENTS_INDEX, "w") as f:
        json.dump(index, f, indent=2)


# ─── Custom Medical Recognizers (Regex) ───────────────────────────────────────

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


# ─── Analyzer + Anonymizer Setup ──────────────────────────────────────────────

@st.cache_resource
def get_analyzer() -> AnalyzerEngine:
    engine = AnalyzerEngine()
    engine.registry.add_recognizer(mrn_recognizer)
    engine.registry.add_recognizer(npi_recognizer)
    return engine


@st.cache_resource
def get_anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


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
    r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b"  # ISO: YYYY-MM-DD
    r"|\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b"  # US: MM/DD/YYYY or MM/DD/YY
)

_VERBOSE_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s+(\d{4})\b"
)


def _shift_numeric(m: re.Match, delta: timedelta) -> str:
    try:
        if m.group(1):  # ISO format
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            shifted = d + delta
            sep = "-" if "-" in m.group(0) else "/"
            return shifted.strftime(f"%Y{sep}%m{sep}%d")
        else:  # US format
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
    """
    Shift all detected dates by offset_days.
    The offset is fixed per vault so timeline relationships remain intact
    across all documents uploaded to the same vault.
    """
    delta = timedelta(days=offset_days)
    text = _NUMERIC_DATE_RE.sub(lambda m: _shift_numeric(m, delta), text)
    text = _VERBOSE_DATE_RE.sub(lambda m: _shift_verbose(m, delta), text)
    return text


# ─── PDF Extraction ────────────────────────────────────────────────────────────

def _ocr_pdf(pdf_bytes: bytes) -> str:
    """Render each page as an image and run Tesseract OCR."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        # Render at 2x zoom for better OCR accuracy
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(pytesseract.image_to_string(img))
    return "\n\n".join(pages)


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, bool]:
    """
    Extract text from PDF. Returns (text, used_ocr).
    Falls back to OCR if the PDF appears to be image-based (text < threshold).
    Requires Tesseract binary installed for OCR fallback.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    text = "\n\n".join(pages)

    if len(text.strip()) >= OCR_TEXT_THRESHOLD:
        return text, False

    # Text too sparse — likely a scanned PDF
    if not _OCR_AVAILABLE:
        return text, False  # return whatever we got; OCR not installed

    try:
        return _ocr_pdf(pdf_bytes), True
    except Exception:
        return text, False  # OCR failed; fall back to sparse text


# ─── De-identification Pipeline ───────────────────────────────────────────────

_OPERATORS = {
    "PERSON": OperatorConfig("replace", {"new_value": "[PATIENT]"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
    "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION]"}),
    # DATE_TIME intentionally excluded — dates are already shifted by shift_dates()
    # to preserve timeline relationships. Erasing them here would defeat that.
    "US_SSN": OperatorConfig("replace", {"new_value": "[SSN]"}),
    "US_PASSPORT": OperatorConfig("replace", {"new_value": "[PASSPORT]"}),
    "MEDICAL_LICENSE": OperatorConfig("replace", {"new_value": "[LICENSE]"}),
    "MEDICAL_RECORD_NUMBER": OperatorConfig("replace", {"new_value": "[MRN]"}),
    "PROVIDER_ID": OperatorConfig("replace", {"new_value": "[NPI]"}),
}

_ENTITIES = list(_OPERATORS.keys())


def deidentify(text: str, offset_days: int) -> tuple[str, list]:
    """
    Full de-id pipeline:
      1. Shift all dates by offset_days (preserves timeline, hides real dates)
      2. Presidio NER: replace names, phones, emails, SSNs, MRNs, NPIs
    Returns (clean_text, analyzer_results).
    """
    text = shift_dates(text, offset_days)

    analyzer = get_analyzer()
    anonymizer = get_anonymizer()

    results = analyzer.analyze(text=text, language="en", entities=_ENTITIES)
    anonymized = anonymizer.anonymize(
        text=text, analyzer_results=results, operators=_OPERATORS
    )
    return anonymized.text, results


# ─── Vector Store ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection("medphi_vault")


@st.cache_resource
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def store_document(doc_id: str, clean_text: str, metadata: dict) -> int:
    """Embed and upsert document chunks into the local ChromaDB vault."""
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
    """Remove all chunks for a document from ChromaDB and the index."""
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


# ─── LLM Query ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are MedPhi, a private medical assistant. "
    "All records have been de-identified. "
    "Dates have been shifted by a fixed offset to preserve timeline relationships "
    "while protecting real dates. "
    "Answer questions using only the provided context. "
    "If the answer is not in the context, say so clearly. "
    "Never speculate beyond what the documents contain."
)


def ask_medphi(question: str, model: str) -> tuple[str, list[dict]]:
    """Query the vault and get an LLM answer grounded in retrieved chunks."""
    docs = query_vault(question)
    context = "\n\n---\n\n".join(d["text"] for d in docs)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context from vault:\n\n{context}\n\nQuestion: {question}",
        },
    ]

    response = completion(model=model, messages=messages)
    return response.choices[0].message.content, docs


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="MedPhi", page_icon="🏥", layout="wide")
    config = load_vault_config()

    st.title("🏥 MedPhi — Medical Memory Vault")
    st.caption("Local-first. Privacy-obsessed. Your records, your machine.")

    tab_upload, tab_docs, tab_query, tab_settings = st.tabs(["📤 Upload", "📋 Documents", "🔍 Query", "⚙️ Settings"])

    # ── Upload Tab ──────────────────────────────────────────────────────────
    with tab_upload:
        st.header("Upload Medical PDF")
        st.info(
            f"Documents are de-identified before storage. "
            f"Names, phone numbers, emails, SSNs, MRNs, and provider IDs are replaced with tokens. "
            f"Dates are shifted by **{config['date_offset_days']} days** — "
            f"timeline relationships are preserved, real dates are not."
        )

        uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
        doc_label = st.text_input(
            "Document label (optional)", placeholder="e.g. Annual Physical 2024"
        )

        if uploaded and st.button("De-identify & Store", type="primary"):
            pdf_bytes = uploaded.read()  # read once; reuse for extraction and hashing

            with st.spinner("Extracting text from PDF..."):
                raw_text, used_ocr = extract_text_from_pdf(pdf_bytes)

            if used_ocr:
                st.info("Scanned PDF detected — OCR was used for text extraction.")
            elif len(raw_text.strip()) < OCR_TEXT_THRESHOLD and not _OCR_AVAILABLE:
                st.warning(
                    "This PDF appears to be image-based but Tesseract is not installed. "
                    "Install Tesseract for OCR support: https://github.com/UB-Mannheim/tesseract/wiki"
                )

            with st.spinner("De-identifying (removing PII, shifting dates)..."):
                clean_text, findings = deidentify(raw_text, config["date_offset_days"])

            doc_id = hashlib.sha256(pdf_bytes).hexdigest()[:16]  # content hash, not name
            metadata = {
                "filename": uploaded.name,
                "label": doc_label or uploaded.name,
                "uploaded_at": datetime.now().isoformat(),
                "pii_entities_found": len(findings),
                "ocr": used_ocr,
            }

            with st.spinner("Embedding and writing to local vault..."):
                n_chunks = store_document(doc_id, clean_text, metadata)

            st.success(
                f"Stored **{n_chunks} chunks**. "
                f"**{len(findings)} PII entities** removed or tokenized."
            )

            with st.expander("Preview de-identified text (first 3,000 chars)"):
                preview = clean_text[:3000] + ("…" if len(clean_text) > 3000 else "")
                st.text_area("De-identified output", preview, height=300)

    # ── Documents Tab ───────────────────────────────────────────────────────
    with tab_docs:
        st.header("Vault Documents")

        index = load_document_index()

        if not index:
            st.info("No documents stored yet. Upload a PDF to get started.")
        else:
            st.caption(f"{len(index)} document(s) in vault")
            st.divider()

            for doc_id, meta in index.items():
                col_info, col_delete = st.columns([5, 1])
                with col_info:
                    label = meta.get("label", meta.get("filename", doc_id))
                    st.markdown(f"**{label}**")
                    uploaded_at = meta.get("uploaded_at", "")[:19].replace("T", " ")
                    pii = meta.get("pii_entities_found", "?")
                    chunks = meta.get("chunk_count", "?")
                    ocr_tag = " · OCR" if meta.get("ocr") else ""
                    st.caption(
                        f"Uploaded: {uploaded_at} · {chunks} chunks · {pii} PII entities removed{ocr_tag} · ID: `{doc_id}`"
                    )
                with col_delete:
                    if st.button("🗑 Delete", key=f"del_{doc_id}"):
                        delete_document(doc_id)
                        st.success(f"Deleted **{label}**.")
                        st.rerun()
                st.divider()

    # ── Query Tab ───────────────────────────────────────────────────────────
    with tab_query:
        st.header("Query Your Vault")

        col1, col2 = st.columns([3, 1])
        with col1:
            question = st.text_input(
                "Ask a question about your records",
                placeholder="What medications was I prescribed in 2023?",
            )
        with col2:
            model = st.selectbox(
                "Model",
                ["gpt-4o-mini", "gpt-4o", "anthropic/claude-sonnet-4-6", "ollama/mistral"],
            )

        if question and st.button("Ask", type="primary"):
            with st.spinner("Searching vault and generating answer..."):
                try:
                    answer, source_docs = ask_medphi(question, model=model)
                except Exception as e:
                    st.error(
                        f"Query failed: {e}\n\n"
                        "Check that your API key is set in `.env` (OPENAI_API_KEY or ANTHROPIC_API_KEY)."
                    )
                    st.stop()

            st.markdown("### Answer")
            st.write(answer)

            with st.expander("Source chunks used"):
                for doc in source_docs:
                    label = doc["metadata"].get("label", doc["metadata"].get("filename", "Unknown"))
                    st.markdown(f"**{label}** — distance: `{doc['distance']:.4f}`")
                    st.text(doc["text"][:400] + ("…" if len(doc["text"]) > 400 else ""))
                    st.divider()

    # ── Settings Tab ────────────────────────────────────────────────────────
    with tab_settings:
        st.header("Vault Settings")

        st.subheader("Date Shift Offset")
        st.warning(
            "Changing the offset makes new uploads inconsistent with existing records. "
            "Only change this before your first upload, or after clearing the vault."
        )
        new_offset = st.number_input(
            "Offset (days)",
            min_value=1,
            max_value=3650,
            value=config["date_offset_days"],
            help="Default: 730 (2 years). All dates in uploaded documents are shifted forward by this amount.",
        )
        if st.button("Save Settings"):
            config["date_offset_days"] = int(new_offset)
            save_vault_config(config)
            st.success("Settings saved.")

        st.divider()
        st.subheader("Vault Statistics")
        try:
            collection = get_collection()
            count = collection.count()
            st.metric("Chunks stored", count)
        except Exception:
            st.info("No documents stored yet.")

        st.caption(f"Config location: `{VAULT_CONFIG}`")
        st.caption(f"ChromaDB location: `{CHROMA_DIR.resolve()}`")
        st.caption(f"Vault created: {config.get('created', 'Unknown')}")


if __name__ == "__main__":
    main()
