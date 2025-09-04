from typing import Any, Dict, List
import json

import time
import uuid
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .elastic import ensure_index
from .ingest import ingest_directory
from .prompts import build_answer_prompt
from .agg_planner import plan_and_run
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
        logger.info(f"[ingest] path='{path}' files={stats.get('files',0)} chunks={stats.get('chunks',0)} indoexed={stats.get('indexed',0)} dt_ms={dt_ms}")
        return stats

    @app.post("/api/chat")
    def chat(payload: Dict[str, Any] | None = None):
        if not payload or not payload.get("query"):
            raise HTTPException(status_code=400, detail="Missing 'query'")
        query = payload["query"]
        history = payload.get("messages") or []

        rid = str(uuid.uuid4())[:8]
        t_start = time.time()
        preview = query.replace("\n", " ")[:120]
        logger.info(f"[{rid}] chat start q_len={len(query)} q_preview='{preview}'")

        # Step A: Try LLM-driven aggregation planning+execution
        agg_result = plan_and_run(query, rid=rid)

        # Build context only from aggregation results (no retrieval)
        packed: List[Dict[str, Any]] = []
        if agg_result:
            packed = [
                {
                    "title": "Aggregations",
                    "source": "elasticsearch",
                    "url": "",
                    "content": json.dumps(agg_result, separators=(",", ":")),
                }
            ]
        messages = build_answer_prompt(query, packed, history=history)
        logger.info(f"[{rid}] prompt messages={len(messages)} passages={len(packed)} history_turns={len(history)}")

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


