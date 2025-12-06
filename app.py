import json
import re
import os
import hashlib
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from xml.etree import ElementTree as ET

import faiss
import numpy as np
import fitz
import requests
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename


app = Flask(__name__)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SOURCES_MANIFEST = UPLOAD_DIR / "sources.json"
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
FAISS_META_PATH = DATA_DIR / "faiss.json"
VECTOR_DIM = 768  # Adjust to match your embedding model output.
CHUNKS_META_PATH = DATA_DIR / "chunks.json"
DEFAULT_MAX_CHUNK_WORDS = 200
DEFAULT_CHUNK_OVERLAP = 50
GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")
GROBID_ENABLED = os.getenv("GROBID_DISABLED", "false").lower() not in {"1", "true", "yes"}

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smart-paper")

# Matches the end of a sentence so we can find the sentence immediately before a {REF} marker.
SENTENCE_END = re.compile(r"[.!?]")
REF_MARKER = re.compile(r"\{REF\}")


def _sentence_before(text: str, position: int) -> str:
    """Return the sentence fragment immediately preceding the given index."""
    last_end = 0
    for match in SENTENCE_END.finditer(text):
        if match.end() > position:
            break
        last_end = match.end()
    return text[last_end:position].strip()


def extract_ref_context(text: str) -> List[Dict[str, Any]]:
    """Find all {REF} markers and return surrounding context for later AI search."""
    contexts: List[Dict[str, Any]] = []
    for index, match in enumerate(REF_MARKER.finditer(text), start=1):
        sentence = _sentence_before(text, match.start())
        contexts.append(
            {
                "placeholder_number": index,
                "start_index": match.start(),
                "context_sentence": sentence,
                "status": "pending",
                "note": "Hook up your AI search here to resolve this reference.",
            }
        )
    return contexts


def _dedupe_path(path: Path) -> Path:
    """Return a path that does not yet exist by appending a counter if needed."""
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}-{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _list_uploaded_files() -> List[Dict[str, str]]:
    """Read the uploads directory fresh each time to reflect any externally added files."""
    files: List[Dict[str, str]] = []
    if not UPLOAD_DIR.exists():
        return files
    for path in sorted(UPLOAD_DIR.iterdir()):
        if path == SOURCES_MANIFEST or path.name.startswith("."):
            continue
        if path.is_file():
            files.append({"filename": path.name, "path": str(path)})
    logger.debug("Discovered %d uploaded files in %s", len(files), UPLOAD_DIR)
    return files


def _load_sources() -> Dict[str, Any]:
    """Load all stored sources (files from disk + links from manifest)."""
    links: List[str] = []
    if SOURCES_MANIFEST.exists():
        try:
            data = json.loads(SOURCES_MANIFEST.read_text())
            links = data.get("links", [])
        except json.JSONDecodeError:
            links = []
    logger.debug("Loaded sources manifest; links=%d", len(links))
    return {"files": _list_uploaded_files(), "links": links}


def _save_sources(data: Dict[str, Any]) -> None:
    """Persist only link data; file list comes from the uploads directory."""
    SOURCES_MANIFEST.write_text(json.dumps({"links": data.get("links", [])}, indent=2))


def _init_vector_store(dim: int = VECTOR_DIM) -> Tuple[faiss.Index, Dict[str, Any]]:
    """Load FAISS index if it exists; otherwise create a fresh one and persist it."""
    if FAISS_INDEX_PATH.exists():
        logger.debug("Loading existing FAISS index from %s", FAISS_INDEX_PATH)
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        meta: Dict[str, Any] = {}
        if FAISS_META_PATH.exists():
            try:
                meta = json.loads(FAISS_META_PATH.read_text())
            except json.JSONDecodeError:
                meta = {}
        meta.setdefault("dim", index.d)
        if meta["dim"] != index.d:
            logger.warning("Metadata dim %s differs from index dim %s; using index dim", meta["dim"], index.d)
            meta["dim"] = index.d
        meta.setdefault("created", datetime.utcnow().isoformat())
        return index, meta

    logger.info("No FAISS index found; creating new index at %s", FAISS_INDEX_PATH)
    index = faiss.IndexFlatL2(dim)
    meta = {"dim": dim, "created": datetime.utcnow().isoformat(), "vectors": 0}
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    FAISS_META_PATH.write_text(json.dumps(meta, indent=2))
    return index, meta


def _persist_vector_state(index: faiss.Index, meta: Dict[str, Any]) -> None:
    """Persist FAISS index and accompanying metadata."""
    meta = {**meta, "vectors": int(index.ntotal), "updated": datetime.utcnow().isoformat()}
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    FAISS_META_PATH.write_text(json.dumps(meta, indent=2))
    VECTOR_META.update(meta)
    logger.debug("Persisted FAISS index; vectors=%s", index.ntotal)


def _index_dim() -> int:
    """Return the active FAISS index dimension."""
    return FAISS_INDEX.d if FAISS_INDEX else VECTOR_META.get("dim", VECTOR_DIM)


def _feature_hash(text: str, dim: int) -> np.ndarray:
    """A lightweight, deterministic embedding using feature hashing (placeholder for a real model)."""
    vec = np.zeros(dim, dtype="float32")
    tokens = re.findall(r"\b\w+\b", text.lower())
    for token in tokens:
        h = int.from_bytes(hashlib.sha1(token.encode("utf-8")).digest()[:8], "little")
        idx = h % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _chunk_text(text: str, max_words: int = DEFAULT_MAX_CHUNK_WORDS, overlap: int = DEFAULT_CHUNK_OVERLAP):
    """Split text into word chunks with overlap."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start = max(0, end - overlap)
    logger.debug("Chunked text into %d chunks (max_words=%d, overlap=%d)", len(chunks), max_words, overlap)
    return chunks


def _extract_page_with_columns(page: fitz.Page, columns: int) -> str:
    """Extract text from a single page by splitting into columns."""
    if columns <= 1:
        return page.get_text("text") or ""
    rect = page.rect
    col_width = rect.width / columns
    parts = []
    for i in range(columns):
        x0 = rect.x0 + i * col_width
        x1 = rect.x0 + (i + 1) * col_width
        clip = fitz.Rect(x0, rect.y0, x1, rect.y1)
        col_text = page.get_text("text", clip=clip) or ""
        if col_text.strip():
            parts.append(col_text.strip())
    return "\n\n".join(parts)


def _choose_column_layout(default_text: str, column_text: str) -> str:
    """Heuristic to pick between single-column and multi-column extraction."""
    def score(text: str) -> float:
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return 0.0
        avg_len = sum(len(line) for line in lines) / len(lines)
        return len(lines) - 0.01 * avg_len

    return column_text if score(column_text) > score(default_text) else default_text


def _extract_text_from_tei(tei_xml: str) -> str:
    """Extract body text from a TEI XML string returned by Grobid."""
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError:
        return ""
    ns = {"tei": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

    def gather(xpath: str) -> list[str]:
        return [" ".join(elem.itertext()).strip() for elem in root.findall(xpath, ns) if " ".join(elem.itertext()).strip()]

    paragraphs = gather(".//tei:body//tei:p") if ns else gather(".//body//p")
    if paragraphs:
        return "\n\n".join(paragraphs)
    body_text = " ".join(root.itertext()).strip()
    return body_text


def _extract_with_grobid(path: Path) -> str:
    """Send a PDF to Grobid and return extracted body text."""
    logger.debug("Attempting Grobid extraction for %s via %s", path, GROBID_URL)
    with path.open("rb") as f:
        files = {"input": f}
        params = {"consolidateHeader": "1", "includeRawCitations": "0"}
        resp = requests.post(f"{GROBID_URL}/api/processFulltextDocument", files=files, data=params, timeout=60)
    resp.raise_for_status()
    text = _extract_text_from_tei(resp.text)
    if not text.strip():
        logger.warning("Grobid returned empty text for %s; falling back", path)
    return text


def _extract_pdf_text(path: Path, columns: int | None = None) -> str:
    """Extract text from a PDF using Grobid if available, otherwise PyMuPDF with optional column handling."""
    logger.debug("Extracting PDF text from %s (columns=%s)", path, columns)

    if GROBID_ENABLED:
        try:
            grobid_text = _extract_with_grobid(path)
            if grobid_text.strip():
                logger.debug("Grobid extraction succeeded for %s", path)
                return grobid_text
            logger.warning("Grobid returned empty text for %s; falling back to PyMuPDF", path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Grobid extraction failed for %s: %s; falling back to PyMuPDF", path, exc)

    text_parts = []
    with fitz.open(path) as doc:
        for page in doc:
            single = page.get_text("text") or ""
            if columns and columns > 1:
                multi = _extract_page_with_columns(page, columns)
                text = _choose_column_layout(single, multi)
            else:
                multi = _extract_page_with_columns(page, 2)
                text = _choose_column_layout(single, multi)
            if text.strip():
                text_parts.append(text.strip())
    logger.debug("Extracted %d text segments from %s via PyMuPDF", len(text_parts), path)
    return "\n\n".join(text_parts)


def _persist_chunks():
    CHUNKS_META_PATH.write_text(json.dumps(CHUNKS_DB, indent=2))


def _ingest_file(path: Path, doc_id: str, columns: int | None = None) -> Dict[str, Any]:
    """Ingest a single file into the vector store."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        text = _extract_pdf_text(path, columns=columns)
    else:
        text = path.read_text(errors="ignore")

    logger.info("Ingesting %s (doc_id=%s, columns=%s)", path, doc_id, columns)
    chunks = _chunk_text(text)
    logger.debug("Doc %s produced %d chunks", doc_id, len(chunks))
    dim = _index_dim()
    vectors = np.stack([_feature_hash(chunk, dim=dim) for chunk in chunks]) if chunks else np.zeros((0, dim), dtype="float32")

    start_id = int(FAISS_INDEX.ntotal)
    if len(vectors):
        FAISS_INDEX.add(vectors)
        _persist_vector_state(FAISS_INDEX, VECTOR_META)
        logger.info("Added %d vectors to FAISS (ntotal=%d)", len(vectors), FAISS_INDEX.ntotal)

    ingested_chunks = []
    for i, chunk in enumerate(chunks):
        vector_id = start_id + i
        record = {
            "vector_id": vector_id,
            "doc_id": doc_id,
            "path": str(path),
            "chunk_index": i,
            "text_preview": chunk[:200],
            "word_count": len(chunk.split()),
        }
        CHUNKS_DB["chunks"].append(record)
        CHUNK_LOOKUP[vector_id] = record
        ingested_chunks.append(record)

    CHUNKS_DB["docs"][doc_id] = {
        "path": str(path),
        "filename": path.name,
        "chunks": len(chunks),
        "ingested_at": datetime.utcnow().isoformat(),
    }
    _persist_chunks()
    logger.info("Finished ingesting doc_id=%s; chunks=%d", doc_id, len(chunks))

    return {"doc_id": doc_id, "chunks_added": len(chunks), "vectors_added": len(vectors)}


# Initialize vector store on startup (creates files on first run, reuses otherwise).
FAISS_INDEX, VECTOR_META = _init_vector_store()
if VECTOR_META.get("dim") != VECTOR_DIM:
    logger.warning(
        "Configured VECTOR_DIM=%s but index dimension is %s. Using index dim to avoid add/search errors.",
        VECTOR_DIM,
        VECTOR_META.get("dim"),
    )
CHUNKS_DB: Dict[str, Any] = {"chunks": [], "docs": {}}
if CHUNKS_META_PATH.exists():
    try:
        CHUNKS_DB = json.loads(CHUNKS_META_PATH.read_text())
    except json.JSONDecodeError:
        CHUNKS_DB = {"chunks": [], "docs": {}}
CHUNK_LOOKUP = {chunk["vector_id"]: chunk for chunk in CHUNKS_DB.get("chunks", [])}


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(force=True, silent=True) or {}
    text = payload.get("text", "")

    references = extract_ref_context(text) if text else []

    return jsonify(
        {
            "input_length": len(text),
            "reference_count": len(references),
            "references": references,
        }
    )


@app.route("/sources", methods=["POST"])
def upload_sources():
    """Accept uploaded files and/or newline-delimited links as reference sources."""
    files = request.files.getlist("files")
    links_raw = request.form.get("links", "") or ""
    links = [link.strip() for link in links_raw.splitlines() if link.strip()]

    sources = _load_sources()
    added_links: List[str] = []
    saved_files = []
    for upload in files:
        if not upload or not upload.filename:
            continue
        safe_name = secure_filename(upload.filename) or "upload"
        destination = _dedupe_path(UPLOAD_DIR / safe_name)
        upload.save(destination)
        record = {
            "filename": destination.name,
            "original_name": upload.filename,
            "path": str(destination),
        }
        saved_files.append(record)

    for link in links:
        if link not in sources["links"]:
            sources["links"].append(link)
            added_links.append(link)

    _save_sources(sources)
    all_files = _list_uploaded_files()

    return jsonify(
        {
            "saved_files": saved_files,
            "file_count": len(saved_files),
            "links": added_links,
            "link_count": len(added_links),
            "all_files": all_files,
            "all_links": sources["links"],
            "upload_dir": str(UPLOAD_DIR),
            "note": "Files stored locally and appended; wire this into your embedding/search pipeline.",
        }
    )


@app.route("/sources", methods=["GET"])
def list_sources():
    """Return currently stored sources without modifying them."""
    sources = _load_sources()
    return jsonify(
        {
            "all_files": sources["files"],
            "all_links": sources["links"],
            "file_count": len(sources["files"]),
            "link_count": len(sources["links"]),
            "upload_dir": str(UPLOAD_DIR),
        }
    )


@app.route("/vector/status", methods=["GET"])
def vector_status():
    """Return FAISS index metadata so callers know whether persistence worked."""
    meta = {
        **VECTOR_META,
        "vectors": int(FAISS_INDEX.ntotal),
        "index_path": str(FAISS_INDEX_PATH),
        "metadata_path": str(FAISS_META_PATH),
        "chunk_metadata_path": str(CHUNKS_META_PATH),
        "documents_indexed": len(CHUNKS_DB.get("docs", {})),
        "chunks_indexed": len(CHUNKS_DB.get("chunks", [])),
        "dim": _index_dim(),
    }
    return jsonify(meta)


def _search_hits(query_text: str, k: int = 5) -> List[Dict[str, Any]]:
    """Return top-k hits from FAISS for the given query text."""
    if FAISS_INDEX.ntotal == 0:
        raise ValueError("index is empty")
    dim = _index_dim()
    query_vec = _feature_hash(query_text, dim=dim).reshape(1, -1)
    distances, indices = FAISS_INDEX.search(query_vec, k)
    hits = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = CHUNK_LOOKUP.get(int(idx))
        if not chunk:
            continue
        hits.append(
            {
                "vector_id": int(idx),
                "score": float(dist),
                "doc_id": chunk["doc_id"],
                "path": chunk["path"],
                "chunk_index": chunk["chunk_index"],
                "text_preview": chunk.get("text_preview", ""),
            }
        )
    return hits


def _resolve_manuscript(text: str) -> Dict[str, Any]:
    """Resolve {REF} markers by retrieving the top vector hit for each context."""
    matches = list(REF_MARKER.finditer(text))
    if not matches:
        return {
            "original_text": text,
            "resolved_text": text,
            "references": [],
            "resolved": [],
            "unresolved": [],
        }

    resolved_refs = []
    unresolved_refs = []
    doc_to_number: Dict[str, int] = {}
    bibliography: List[Dict[str, Any]] = []
    output_parts = []
    last_idx = 0

    for i, match in enumerate(matches, start=1):
        context_sentence = _sentence_before(text, match.start())
        ref_label = "{REF}"
        assigned_doc = None
        hit = None

        if context_sentence.strip():
            try:
                hits = _search_hits(context_sentence, k=3)
                if hits:
                    hit = hits[0]
                    assigned_doc = hit["doc_id"]
                    if assigned_doc not in doc_to_number:
                        doc_to_number[assigned_doc] = len(doc_to_number) + 1
                    ref_label = f"[{doc_to_number[assigned_doc]}]"
                else:
                    logger.debug("No hits for ref %s", i)
            except Exception as exc:  # noqa: BLE001
                logger.error("Search failed for ref %s: %s", i, exc)
                unresolved_refs.append({"ref_number": i, "context": context_sentence, "error": str(exc)})
        else:
            logger.debug("Empty context for ref %s", i)

        output_parts.append(text[last_idx:match.start()])
        output_parts.append(ref_label)
        last_idx = match.end()

        ref_entry = {
            "ref_number": i,
            "label": ref_label,
            "context": context_sentence,
            "hit": hit,
        }
        resolved_refs.append(ref_entry)
        if not hit:
            unresolved_refs.append(ref_entry)

    output_parts.append(text[last_idx:])
    resolved_text = "".join(output_parts)

    for doc_id, number in sorted(doc_to_number.items(), key=lambda kv: kv[1]):
        doc_meta = CHUNKS_DB.get("docs", {}).get(doc_id, {})
        bibliography.append(
            {
                "number": number,
                "doc_id": doc_id,
                "filename": doc_meta.get("filename", doc_id),
                "path": doc_meta.get("path", ""),
                "note": "Placeholder citation; swap in real metadata when available.",
            }
        )

    return {
        "original_text": text,
        "resolved_text": resolved_text,
        "references": resolved_refs,
        "unresolved": unresolved_refs,
        "bibliography": bibliography,
        "doc_to_number": doc_to_number,
    }


@app.route("/vector/ingest", methods=["POST"])
def vector_ingest():
    """Ingest all unindexed files from uploads/ into FAISS using feature hashing embeddings."""
    payload = request.get_json(silent=True) or {}
    force = bool(payload.get("force"))
    column_hints = payload.get("columns", {}) or {}

    uploads = _list_uploaded_files()
    logger.info("Starting ingestion of %d upload(s); force=%s", len(uploads), force)
    ingested = []
    skipped = []

    for file_entry in uploads:
        doc_id = file_entry["filename"]
        path = Path(file_entry["path"])
        col_hint = column_hints.get(doc_id)
        if isinstance(col_hint, str) and col_hint.isdigit():
            col_hint = int(col_hint)
        already_indexed = doc_id in CHUNKS_DB.get("docs", {})
        if already_indexed and not force:
            skipped.append({"doc_id": doc_id, "reason": "already indexed"})
            logger.debug("Skipping %s (already indexed)", doc_id)
            continue
        if already_indexed and force:
            skipped.append({"doc_id": doc_id, "reason": "force reindex not supported yet"})
            logger.debug("Skipping %s (force reindex not supported)", doc_id)
            continue

        try:
            result = _ingest_file(path, doc_id, columns=col_hint)
            ingested.append(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Ingestion failed for %s: %s", doc_id, exc)
            logger.debug("Traceback: %s", traceback.format_exc())
            skipped.append({"doc_id": doc_id, "reason": f"ingest failed: {exc}", "trace": traceback.format_exc()})

    status = {
        "ingested": ingested,
        "skipped": skipped,
        "indexed_documents": len(CHUNKS_DB.get("docs", {})),
        "indexed_chunks": len(CHUNKS_DB.get("chunks", [])),
        "vectors_total": int(FAISS_INDEX.ntotal),
        "vector_meta": VECTOR_META,
    }
    return jsonify(status)


@app.route("/vector/query", methods=["POST"])
def vector_query():
    """Query the FAISS index using feature-hashed embeddings."""
    payload = request.get_json(force=True)
    query_text = payload.get("text", "")
    top_k = int(payload.get("k", 5))
    if not query_text.strip():
        return jsonify({"error": "text is required"}), 400
    if FAISS_INDEX.ntotal == 0:
        return jsonify({"error": "index is empty. Ingest sources first."}), 400

    logger.debug("Running vector query (k=%d)", top_k)
    query_vec = _feature_hash(query_text, dim=_index_dim()).reshape(1, -1)
    distances, indices = FAISS_INDEX.search(query_vec, top_k)

    # Build lookup map for chunks by vector_id.
    chunk_by_vector = {chunk["vector_id"]: chunk for chunk in CHUNKS_DB.get("chunks", [])}
    hits = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunk_by_vector.get(int(idx))
        if not chunk:
            continue
        hits.append(
            {
                "vector_id": int(idx),
                "score": float(dist),
                "doc_id": chunk["doc_id"],
                "path": chunk["path"],
                "chunk_index": chunk["chunk_index"],
                "text_preview": chunk.get("text_preview", ""),
            }
        )

    return jsonify({"count": len(hits), "results": hits})


@app.route("/resolve", methods=["POST"])
def resolve_manuscript():
    """Resolve {REF} markers by searching the vector index and replacing with citations."""
    payload = request.get_json(force=True, silent=True) or {}
    text = payload.get("text", "")
    if not text.strip():
        return jsonify({"error": "text is required"}), 400
    if FAISS_INDEX.ntotal == 0:
        return jsonify({"error": "index is empty. Ingest sources first."}), 400

    result = _resolve_manuscript(text)
    return jsonify(result)


if __name__ == "__main__":
    # Use debug mode for local development; remove or change in production.
    app.run(debug=True, port=8888)
