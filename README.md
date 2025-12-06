# Smart Paper Citation (prototype)

A tiny Flask app that lets you paste manuscript text, detects `{REF}` placeholders, and prepares the surrounding sentence for downstream AI-powered citation lookups. Front-end is intentionally simple and runs locally with no external services.

## Quickstart

1. In the application directory, create and activate a virtual environment (recommended):
   - Windows (PowerShell): `python -m venv .venv; .\.venv\Scripts\activate`
   - macOS/Linux (bash/zsh): `python3 -m venv .venv && source .venv/bin/activate`
2. Install/start everything with one command:
   - `python start.py` or `python3 start.py`
3. Upload reference sources (optional): use the "Reference sources" section to attach PDFs or paste links (one per line). Files are stored under `uploads/` and **new uploads are appended, not replaced**. The UI always re-reads the `uploads/` directory so externally added files show up too.
5. Vector store (FAISS): auto-initialized on first run under `data/faiss.index` with metadata in `data/faiss.json`; subsequent runs reuse the same index. Check status at `GET /vector/status`.
6. Ingest uploads into the vector index: click "Ingest uploads into index" in the UI or call `POST /vector/ingest`. Then run queries from the UI or via `POST /vector/query` with JSON `{ "text": "...", "k": 5 }`.

## How it works

- Backend: Flask routes
  - `GET /` serves a minimal page with a textarea and client-side fetch logic.
  - `POST /analyze` accepts JSON `{ "text": "..." }`, finds every `{REF}` marker, extracts the immediately preceding sentence fragment, and returns JSON for display.
  - `GET /sources` returns all stored sources (files + links) without modifying them.
  - `POST /sources` accepts multipart form data: `files` (one or more uploads) and `links` (newline-delimited). Saves files to `uploads/`, appends links, and returns all stored sources.
  - `GET /vector/status` reports FAISS index metadata (dim, vector count, paths) to confirm persistence.
  - `POST /vector/ingest` ingests unindexed files from `uploads/` (PDF/text) into FAISS with feature-hash embeddings and chunk metadata.
  - `POST /vector/query` queries FAISS with a text prompt and returns top hits with chunk previews.
  - `POST /resolve` resolves `{REF}` markers by querying the vector index with the surrounding sentence, replacing each marker with `[n]` for the top hit, and returning a bibliography (one entry per document).
- Frontend: a single HTML template (`templates/index.html`) with inline CSS/JS.

## Notes on vector store (FAISS)

- The app initializes a FAISS CPU index on startup using a default dimension of 384 (adjust `VECTOR_DIM` in `app.py` to match your embedding model).
- Index path: `data/faiss.index`; metadata: `data/faiss.json`. These are created if missing and reused otherwise.
- A lightweight, deterministic feature-hash embedding is used as a placeholder (no external model downloads) with `VECTOR_DIM=768`. Swap in your real embedding model and keep dimensions in sync.
- Index path: `data/faiss.index`; metadata: `data/faiss.json`; chunk/doc metadata: `data/chunks.json`. These are created if missing and reused otherwise.
- Ingestion currently chunks by words (200 max, 50 overlap) for PDFs and plain text files in `uploads/`.
- PDF parsing now uses PyMuPDF (fitz). You can pass `{"columns": {"yourfile.pdf": 2}}` in the JSON body to `POST /vector/ingest` to force 2-column extraction; otherwise a heuristic compares single vs 2-column extraction per page and chooses the better layout.
- Grobid integration (preferred for messy scientific PDFs): extraction attempts Grobid first (assumes the service is running), then falls back to fitz if Grobid fails or returns empty text. Configure `GROBID_URL` if not on `http://localhost:8070`. Bundled setup: `python scripts/grobid_bootstrap.py setup` then `python scripts/grobid_bootstrap.py start`.
