# 🏥 MedPhi — Private Medical Memory Vault

Local-first. Privacy-obsessed. Your records, your machine.

MedPhi turns your medical PDFs into a queryable memory vault — stripping all personal identifiers before anything is stored, so you can ask intelligent questions about your own health history without your data ever leaving your computer.

---

## What It Does

1. **Upload** — drop any medical PDF (lab results, doctor notes, imaging reports, prescriptions)
2. **De-identify** — names, phone numbers, emails, SSNs, MRNs, and provider IDs are replaced with tokens. Dates are shifted by a fixed offset to preserve timeline relationships while hiding real dates.
3. **Store** — de-identified text is embedded and written to a local ChromaDB vector database
4. **Query** — ask plain-English questions about your records using any LLM you choose

Nothing leaves your machine unless you explicitly choose a cloud LLM.

---

## Quick Start

```bash
git clone https://github.com/jkh2/medphi.git
cd medphi
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env         # add your API key if using a cloud LLM
streamlit run medphi.py
```

Then open `http://localhost:8501` in your browser.

---

## OCR Support (Scanned PDFs)

MedPhi auto-detects image-based PDFs and falls back to OCR. To enable it, install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (Windows) or `brew install tesseract` (Mac). No config needed — MedPhi finds it automatically.

---

## Models

MedPhi uses [LiteLLM](https://github.com/BerriAI/litellm), so any supported provider works. Set your key in `.env`:

| Model | Key needed |
|---|---|
| `gpt-4o-mini` / `gpt-4o` | `OPENAI_API_KEY` |
| `anthropic/claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| `ollama/mistral` (fully local) | none — install [Ollama](https://ollama.com) |

---

## Privacy & Security

- **All vault data lives in `~/.medphi/`** — outside the project directory, never committed
- Medical data is de-identified before it ever reaches the vector database
- Dates are shifted (not erased) so your timeline stays intact
- No accounts, no telemetry, no phoning home
- API keys stay in your local `.env` file

> **Disclaimer:** MedPhi is not medical advice and is not a substitute for professional care. It is a personal memory tool for organizing your own health records.

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| PDF extraction | PyMuPDF |
| OCR fallback | pytesseract + Tesseract |
| PII removal | Microsoft Presidio + custom MRN/NPI recognizers |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (persistent, local) |
| LLM | LiteLLM (any provider) |
| Date shifting | Custom regex pipeline |

---

## MCP Server (Use MedPhi from Claude Code or Cursor)

MedPhi ships with an MCP server so you can query your vault directly from any MCP-compatible AI client — no browser, no separate app.

**Add to your MCP client config:**

| Client | Config file location |
|---|---|
| Claude Code (Mac/Linux) | `~/.claude/settings.json` |
| Claude Code (Windows) | `%USERPROFILE%\.claude\settings.json` |
| Cursor | `~/.cursor/mcp.json` |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` |

```json
{
  "mcpServers": {
    "medphi": {
      "command": "python",
      "args": ["/absolute/path/to/medphi/medphi_mcp_server.py"]
    }
  }
}
```

**Available tools:**

| Tool | What it does |
|---|---|
| `medphi_ingest` | Ingest a PDF by file path — de-identifies and stores it |
| `medphi_query` | Semantic search — returns relevant de-identified chunks |
| `medphi_list_documents` | List all stored documents with metadata |
| `medphi_delete_document` | Remove a document and all its chunks |
| `medphi_vault_stats` | Chunk count, doc count, vault config |

Once configured, you can ask your AI assistant things like:
> *"Search my medical vault for anything about blood pressure medication"*
> *"What does my vault say about my 2023 cardiology visit?"*

The Streamlit UI (`streamlit run medphi.py`) is still available for a visual interface.

---

## Roadmap

Built and shipped:
- [x] PDF text extraction + OCR fallback for scanned documents
- [x] PII stripping — names, phones, emails, SSNs, MRNs, provider IDs
- [x] Date shifting with persistent per-vault offset (timeline preserved)
- [x] Local ChromaDB vector storage
- [x] LiteLLM query layer (Ollama, OpenAI, Anthropic, and more)
- [x] Streamlit UI — Upload, Documents, Query, Settings tabs
- [x] Document list view with metadata
- [x] Per-document delete from vault

Coming next:
- [ ] Consistent fake name surrogates (same placeholder every time a patient appears across documents)
- [ ] Multi-vault support (separate vaults per family member or condition)
- [ ] Encrypted vault export/import
- [ ] Date-shift customization per document
- [ ] Better medical-specific chunking (section-aware for clinical notes)

---

## License

**CC BY-NC 4.0** — free for personal and educational use. Not for commercial use or resale.

Copyright © 2026 James Harwood (github.com/jkh2) & the MedPhi team

Full license: https://creativecommons.org/licenses/by-nc/4.0/
