import hashlib
import os
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import tiktoken

from .embeddings import embed_many
from .elastic import index_docs, ensure_index


def _read_files(path: str) -> List[Tuple[str, str]]:
    paths: List[Tuple[str, str]] = []
    for root, _, files in os.walk(path):
        for name in files:
            if name.lower().endswith((".md", ".txt")):
                full = os.path.join(root, name)
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    paths.append((full, content))
                except Exception:
                    continue
    return paths


def _chunk_text(text: str, max_tokens: int = 750, overlap: int = 100) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks


def _doc_id(source: str, title: str, content: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(title.encode("utf-8"))
    h.update(content.encode("utf-8"))
    return h.hexdigest()


def ingest_directory(path: str) -> Dict[str, int]:
    ensure_index()
    files = _read_files(path)
    chunks: List[Dict[str, str]] = []

    for full, content in files:
        title = os.path.basename(full)
        for chunk in _chunk_text(content):
            chunks.append({"title": title, "content": chunk, "source": "fs", "url": full})

    if not chunks:
        return {"files": len(files), "chunks": 0, "indexed": 0}

    embeddings = embed_many([c["content"] for c in chunks])
    now = datetime.utcnow().isoformat() + "Z"
    to_index: List[Dict[str, object]] = []
    for c, emb in zip(chunks, embeddings):
        _id = _doc_id(c["url"], c["title"], c["content"])
        to_index.append(
            {
                "id": _id,
                "title": c["title"],
                "url": c["url"],
                "source": c["source"],
                "content": c["content"],
                "embedding": emb,
                "created_at": now,
            }
        )

    success, _ = index_docs(to_index)
    return {"files": len(files), "chunks": len(chunks), "indexed": success}


