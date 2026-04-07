"""
MedPhi MCP Server
Exposes the MedPhi vault as MCP tools for any compatible client
(Claude Code, Cursor, etc.).

Configure in your MCP client's settings:
  {
    "medphi": {
      "command": "python",
      "args": ["/path/to/medphi_mcp_server.py"]
    }
  }
"""

import asyncio
import json
import traceback

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from medphi_core import (
    ingest_pdf,
    query_vault,
    load_document_index,
    delete_document,
    get_collection,
    load_vault_config,
    VAULT_CONFIG,
    CHROMA_DIR,
)

server = Server("medphi")


# ─── Tool Definitions ─────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="medphi_ingest",
            description=(
                "Ingest a medical PDF into the local MedPhi vault. "
                "The document is de-identified (names, phones, SSNs, MRNs, NPIs removed; "
                "dates shifted by a fixed offset) before being embedded and stored. "
                "Nothing sensitive is retained."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the PDF file to ingest.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional human-readable label (e.g. 'Annual Physical 2024').",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="medphi_query",
            description=(
                "Search the MedPhi vault for chunks relevant to a question. "
                "Returns de-identified text chunks from stored medical records. "
                "Use the returned context to answer the user's question directly — "
                "no second LLM call is needed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Plain-English question about the medical records.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (default: 5).",
                        "default": 5,
                    },
                },
                "required": ["question"],
            },
        ),
        Tool(
            name="medphi_list_documents",
            description="List all documents currently stored in the MedPhi vault.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="medphi_delete_document",
            description=(
                "Delete a document and all its chunks from the MedPhi vault. "
                "Use medphi_list_documents to find the doc_id first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "The doc_id of the document to delete.",
                    },
                },
                "required": ["doc_id"],
            },
        ),
        Tool(
            name="medphi_vault_stats",
            description="Return statistics and configuration for the MedPhi vault.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


# ─── Tool Handlers ────────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "medphi_ingest":
            result = ingest_pdf(
                file_path=arguments["file_path"],
                label=arguments.get("label", ""),
            )
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2),
            )]

        elif name == "medphi_query":
            question = arguments["question"]
            n_results = int(arguments.get("n_results", 5))
            docs = query_vault(question, n_results=n_results)

            if not docs:
                return [TextContent(
                    type="text",
                    text="No documents found in vault. Ingest some PDFs first using medphi_ingest.",
                )]

            output = f"Found {len(docs)} relevant chunk(s) for: \"{question}\"\n\n"
            for i, doc in enumerate(docs, 1):
                label = doc["metadata"].get("label", doc["metadata"].get("filename", "Unknown"))
                output += f"--- Chunk {i} | Source: {label} | Distance: {doc['distance']:.4f} ---\n"
                output += doc["text"] + "\n\n"

            return [TextContent(type="text", text=output.strip())]

        elif name == "medphi_list_documents":
            index = load_document_index()
            if not index:
                return [TextContent(type="text", text="Vault is empty. No documents stored.")]

            lines = [f"{len(index)} document(s) in vault:\n"]
            for doc_id, meta in index.items():
                label = meta.get("label", meta.get("filename", doc_id))
                uploaded_at = meta.get("uploaded_at", "")[:19].replace("T", " ")
                chunks = meta.get("chunk_count", "?")
                pii = meta.get("pii_entities_found", "?")
                ocr = " [OCR]" if meta.get("ocr") else ""
                lines.append(
                    f"  • {label}{ocr}\n"
                    f"    doc_id: {doc_id} | uploaded: {uploaded_at} | "
                    f"chunks: {chunks} | PII removed: {pii}"
                )
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "medphi_delete_document":
            doc_id = arguments["doc_id"]
            index = load_document_index()
            if doc_id not in index:
                return [TextContent(
                    type="text",
                    text=f"No document with doc_id '{doc_id}' found in vault.",
                )]
            label = index[doc_id].get("label", doc_id)
            delete_document(doc_id)
            return [TextContent(
                type="text",
                text=f"Deleted '{label}' (doc_id: {doc_id}) and all its chunks.",
            )]

        elif name == "medphi_vault_stats":
            config = load_vault_config()
            index = load_document_index()
            try:
                chunk_count = get_collection().count()
            except Exception:
                chunk_count = 0

            stats = {
                "documents": len(index),
                "chunks": chunk_count,
                "date_offset_days": config.get("date_offset_days", 730),
                "vault_created": config.get("created", "unknown"),
                "vault_config_path": str(VAULT_CONFIG),
                "chroma_db_path": str(CHROMA_DIR),
            }
            return [TextContent(type="text", text=json.dumps(stats, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except FileNotFoundError as e:
        return [TextContent(type="text", text=f"File not found: {e}")]
    except ValueError as e:
        return [TextContent(type="text", text=f"Invalid input: {e}")]
    except Exception:
        return [TextContent(type="text", text=f"Error:\n{traceback.format_exc()}")]


# ─── Entry Point ──────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
