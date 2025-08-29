from typing import Any, Dict, List

import time
import uuid
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .elastic import ensure_index
from .ingest import ingest_directory
from .retrieve import retrieve
from .rerank import rerank
from .prompts import build_answer_prompt
from .embeddings import embed_text
from .answer import answer_question


logger = logging.getLogger("ragbot.chat")


def create_app() -> FastAPI:
    app = FastAPI(title="RAGBot API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> Dict[str, str]:
        ensure_index()
        logger.info("[health] ok")
        return {"status": "ok"}

    @app.post("/api/ingest")
    def ingest(payload: Dict[str, Any]) -> Dict[str, Any]:
        path = payload.get("path")
        if not path:
            raise HTTPException(status_code=400, detail="Missing 'path'")
        t0 = time.time()
        stats = ingest_directory(path)
        dt_ms = int((time.time() - t0) * 1000)
        logger.info(f"[ingest] path='{path}' files={stats.get('files',0)} chunks={stats.get('chunks',0)} indexed={stats.get('indexed',0)} dt_ms={dt_ms}")
        return stats

    @app.post("/api/chat")
    def chat(payload: Dict[str, str] | None = None):
        if not payload or not payload.get("query"):
            raise HTTPException(status_code=400, detail="Missing 'query'")
        query = payload["query"]
        rid = str(uuid.uuid4())[:8]
        t_start = time.time()
        preview = query.replace("\n", " ")[:120]
        logger.info(f"[{rid}] chat start q_len={len(query)} q_preview='{preview}'")

        t0 = time.time()
        initial_docs = retrieve(query, top_k=12)
        dt_retrieve = int((time.time() - t0) * 1000)
        logger.info(f"[{rid}] retrieve docs={len(initial_docs)} dt_ms={dt_retrieve}")

        t1 = time.time()
        qvec = embed_text(query)
        dt_embed = int((time.time() - t1) * 1000)
        logger.info(f"[{rid}] embed dt_ms={dt_embed}")

        t2 = time.time()
        reranked = rerank(qvec, initial_docs, top_k=8)
        dt_rerank = int((time.time() - t2) * 1000)
        logger.info(f"[{rid}] rerank in={len(initial_docs)} out={len(reranked)} dt_ms={dt_rerank}")

        messages = build_answer_prompt(query, reranked)
        logger.info(f"[{rid}] prompt messages={len(messages)} passages={len(reranked)}")

        def stream() -> Any:
            logger.info(f"[{rid}] stream start")
            chars = 0
            for token in answer_question(messages):
                chars += len(token)
                yield token
            total_ms = int((time.time() - t_start) * 1000)
            logger.info(f"[{rid}] stream end chars={chars} total_ms={total_ms}")

        return StreamingResponse(stream(), media_type="text/plain")

    return app


app = create_app()


