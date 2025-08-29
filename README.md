# RAGBot: Functional RAG Chatbot (FastAPI + Elasticsearch + OpenAI)

RAGBot is a production-ready Retrieval-Augmented Generation system built in functional Python (no classes). It uses Elasticsearch for hybrid retrieval (BM25 + vector kNN) and OpenAI for embeddings and answer generation. The frontend is a React + Vite + Tailwind app with a terminal/hacker theme.

## Stack
- Backend: FastAPI (functional), Python 3.11
- Search: Elasticsearch 8 with `dense_vector` kNN + BM25 (hybrid)
- LLM + Embeddings: OpenAI (`text-embedding-3-large`, `gpt-4o`)
- Frontend: React + Vite + Tailwind

## Repository Layout
```
/ragbot
  /api
    env.py            # env loading (full and ES-only), embedding_dimensions
    openai_client.py  # OpenAI HTTP calls: embeddings, chat (streaming)
    embeddings.py     # thin wrappers for embedding calls
    elastic.py        # ES client, ensure_index, index_docs, hybrid search
    ingest.py         # read .md/.txt, chunk, embed, index
    retrieve.py       # embed query, hybrid ES search
    rerank.py         # MMR re-ranking using cosine
    prompts.py        # build system+user messages with citations
    answer.py         # stream tokens from OpenAI
    server.py         # FastAPI app and endpoints
  /web
    index.html, package.json, vite.config.ts, tailwind.css, tailwind.config.js, postcss.config.js
    src/main.tsx, src/App.tsx, src/components/{Chat.tsx,Message.tsx,TypingDots.tsx}, src/lib/api.ts, tsconfig.json
  /scripts
    load_sample.py    # ingest a directory of .md/.txt
    ingest_animals.py # generate + ingest N synthetic animal records
  requirements.txt
  docker-compose.yml  # optional: local Elasticsearch (or use docker run)
```

## Environment
Create `.env` at repo root (dotenv is loaded by the backend):
```env
OPENAI_API_KEY=sk-...
ES_HOST=http://localhost:9200
ES_USERNAME=elastic
ES_PASSWORD=changeme
ES_INDEX=kb_docs
EMBED_MODEL=text-embedding-3-large
GEN_MODEL=gpt-4o
```
Notes:
- ES-only operations (health check, index creation) do not require `OPENAI_API_KEY`.
- Changing `EMBED_MODEL` affects vector `dims`. Reindex if you change it.

## Run Elasticsearch
Option A: docker-compose (from `ragbot/`)
```bash
cd /home/manav/Documents/RAG-based-LLM/ragbot
docker compose up -d
```
Option B: docker run
```bash
docker run -d --name es-ragbot -e discovery.type=single-node \
  -e xpack.security.enabled=true -e ELASTIC_PASSWORD=changeme \
  -p 9200:9200 -p 9300:9300 docker.elastic.co/elasticsearch/elasticsearch:8.13.4
```

## Backend Setup
```bash
python3.11 -m venv /home/manav/Documents/RAG-based-LLM/.venv
source /home/manav/Documents/RAG-based-LLM/.venv/bin/activate
pip install -r /home/manav/Documents/RAG-based-LLM/ragbot/requirements.txt

# environment
cp /home/manav/Documents/RAG-based-LLM/.env.example /home/manav/Documents/RAG-based-LLM/.env
# edit .env, set OPENAI_API_KEY

export PYTHONPATH=/home/manav/Documents/RAG-based-LLM
uvicorn ragbot.api.server:app --host 0.0.0.0 --port 8000 --reload
```
Health check (creates index if missing):
```bash
curl http://localhost:8000/api/health
```

## Frontend Setup
```bash
cd /home/manav/Documents/RAG-based-LLM/ragbot/web
npm install
npm run dev
```
Open http://localhost:5173

The frontend streams from `http://localhost:8000`. If the API runs elsewhere, edit `ragbot/web/src/lib/api.ts`.

## Ingest Content
Directory of `.md` / `.txt` files:
```bash
# run from repo root, module mode ensures PYTHONPATH and dotenv
source /home/manav/Documents/RAG-based-LLM/.venv/bin/activate
python -m ragbot.scripts.load_sample --path /path/to/docs
```
Synthetic animals (default 100):
```bash
python -m ragbot.scripts.ingest_animals --count 100
```

## Chat
```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"Summarize what was ingested"}'
```
Or open the frontend and ask via the UI.

## Architecture & Flow
- Ingest pipeline (`ingest.py`):
  1) Read `.md/.txt` files
  2) Tokenize with `tiktoken` and chunk (~750 tokens, 100 overlap)
  3) Batch-embed chunks with OpenAI embeddings (`text-embedding-3-large`)
  4) Index docs into ES with fields: `id, title, url, source, content, embedding, created_at`
- Chat pipeline (`server.py` → `retrieve.py` → `rerank.py` → `prompts.py` → `answer.py`):
  1) Embed query, perform hybrid ES search (BM25 + kNN)
  2) Re-rank with MMR for diversity
  3) Build messages with the top passages (include source/citation tags)
  4) Stream generation from OpenAI Chat Completions to client

## API Endpoints
- `GET /api/health` → `{ "status": "ok" }` (ensures index)
- `POST /api/ingest` → `{ path: string }` → `{ files, chunks, indexed }`
- `POST /api/chat` → `{ query: string }` → streams `text/plain`

## Inspect Indexed Documents
```bash
# total
curl -u elastic:changeme http://localhost:9200/kb_docs/_count

# a few animals
curl -s -u elastic:changeme http://localhost:9200/kb_docs/_search \
  -H 'Content-Type: application/json' \
  -d '{"query":{"term":{"source":"animals"}}, "size":5, "_source":["id","title","source","created_at"]}'
```

## Logging & Observability
- The backend logs structured messages with request ids and timings:
  - chat start, retrieve, embed, rerank, prompt, stream start/end
- Example startup:
  ```bash
  uvicorn ragbot.api.server:app --host 0.0.0.0 --port 8000 --reload
  ```

## Troubleshooting
- Health 500 with index setting errors:
  - We create the index using `dense_vector` mappings only (no `index.knn` setting).
- "No module named ragbot":
  - Run scripts as modules from repo root:
    ```bash
    cd /home/manav/Documents/RAG-based-LLM
    source .venv/bin/activate
    python -m ragbot.scripts.ingest_animals --count 100
    ```
- Two assistant bubbles in UI:
  - Fixed by updating the existing assistant placeholder rather than appending a second.
- UI scroll:
  - Internal scroll removed; page scrolls naturally as content grows.

## Production Notes
- Security:
  - Enable TLS for Elasticsearch and `verify_certs=True` in production.
  - Restrict CORS to the frontend origin.
  - Store secrets in environment or a secret manager.
- Performance:
  - Batch embeddings (already implemented), tune ES `k` and `num_candidates`.
  - Consider caching frequent query embeddings.
- Index evolution:
  - Changing `EMBED_MODEL` changes vector dims; reindex to a new index.

## Extensibility
- Swap `rerank.py` for alternate strategies (e.g., cross-encoders).
- Customize `prompts.py` for tone, citations, guardrails.
- Add new ingesters (PDFs, web crawler) as long as they output the same doc shape.
