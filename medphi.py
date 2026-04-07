"""
MedPhi — Streamlit UI
All logic lives in medphi_core.py.
Run: streamlit run medphi.py
"""

import hashlib
from datetime import datetime

import streamlit as st
from litellm import completion
from dotenv import load_dotenv

from medphi_core import (
    OCR_TEXT_THRESHOLD,
    VAULT_CONFIG,
    CHROMA_DIR,
    _OCR_AVAILABLE,
    load_vault_config,
    save_vault_config,
    load_document_index,
    delete_document,
    extract_text_from_pdf,
    deidentify,
    store_document,
    query_vault,
    get_collection,
)

load_dotenv()

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
    docs = query_vault(question)
    context = "\n\n---\n\n".join(d["text"] for d in docs)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Context from vault:\n\n{context}\n\nQuestion: {question}"},
    ]
    response = completion(model=model, messages=messages)
    return response.choices[0].message.content, docs


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="MedPhi", page_icon="🏥", layout="wide")
    config = load_vault_config()

    st.title("🏥 MedPhi — Medical Memory Vault")
    st.caption("Local-first. Privacy-obsessed. Your records, your machine.")

    tab_upload, tab_docs, tab_query, tab_settings = st.tabs(
        ["📤 Upload", "📋 Documents", "🔍 Query", "⚙️ Settings"]
    )

    # ── Upload Tab ──────────────────────────────────────────────────────────
    with tab_upload:
        st.header("Upload Medical PDF")
        st.info(
            f"Documents are de-identified before storage. "
            f"Names, phones, emails, SSNs, MRNs, and provider IDs are replaced with tokens. "
            f"Dates are shifted by **{config['date_offset_days']} days** — "
            f"timeline relationships are preserved, real dates are not."
        )

        uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
        doc_label = st.text_input(
            "Document label (optional)", placeholder="e.g. Annual Physical 2024"
        )

        if uploaded and st.button("De-identify & Store", type="primary"):
            pdf_bytes = uploaded.read()

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

            doc_id = hashlib.sha256(pdf_bytes).hexdigest()[:16]
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
                        f"Uploaded: {uploaded_at} · {chunks} chunks · "
                        f"{pii} PII entities removed{ocr_tag} · ID: `{doc_id}`"
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
                        "Check that your API key is set in `.env`."
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
            help="Default: 730 (2 years). All dates are shifted forward by this amount.",
        )
        if st.button("Save Settings"):
            config["date_offset_days"] = int(new_offset)
            save_vault_config(config)
            st.success("Settings saved.")

        st.divider()
        st.subheader("Vault Statistics")
        try:
            count = get_collection().count()
            st.metric("Chunks stored", count)
        except Exception:
            st.info("No documents stored yet.")

        st.caption(f"Config: `{VAULT_CONFIG}`")
        st.caption(f"ChromaDB: `{CHROMA_DIR.resolve()}`")
        st.caption(f"Vault created: {config.get('created', 'Unknown')}")


if __name__ == "__main__":
    main()
