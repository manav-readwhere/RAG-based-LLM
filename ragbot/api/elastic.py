from typing import Any, Dict, Iterable, List, Optional, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from .env import load_env_es, embedding_dimensions


def get_client() -> Elasticsearch:
    env = load_env_es()
    es = Elasticsearch(
        hosts=[env["ES_HOST"]],
        basic_auth=(env["ES_USERNAME"], env["ES_PASSWORD"]),
        verify_certs=False,
        request_timeout=60,
        retry_on_timeout=True,
        max_retries=3,
    )
    return es


def ensure_index() -> Dict[str, Any]:
    env = load_env_es()
    index_name = env["ES_INDEX"]
    es = get_client()
    if es.indices.exists(index=index_name):
        return {"index": index_name, "created": False}

    dims = embedding_dimensions(env.get("EMBED_MODEL", "text-embedding-3-large"))
    mappings = {
        "properties": {
            "id": {"type": "keyword"},
            "title": {"type": "text"},
            "url": {"type": "keyword", "index": False},
            "source": {"type": "keyword"},
            "content": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine",
            },
            "created_at": {"type": "date"},
        }
    }
    es.indices.create(index=index_name, mappings=mappings)
    return {"index": index_name, "created": True}


def _bulk_actions(index: str, docs: Iterable[Dict[str, Any]]):
    for doc in docs:
        _id = doc.get("id")
        yield {"_op_type": "index", "_index": index, "_id": _id, "_source": doc}


def index_docs(docs: List[Dict[str, Any]]) -> Tuple[int, List[Any]]:
    env = load_env_es()
    es = get_client()
    ensure_index()
    success, results = bulk(es, _bulk_actions(env["ES_INDEX"], docs))
    return success, results


def search(query: str, query_vector: Optional[List[float]], top_k: int = 8) -> Dict[str, Any]:
    env = load_env_es()
    es = get_client()
    index = env["ES_INDEX"]

    knn_clause: Optional[Dict[str, Any]] = None
    if query_vector is not None:
        knn_clause = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": max(top_k * 4, 10),
            "num_candidates": max(top_k * 25, 100),
        }

    text_query = {
        "bool": {
            "should": [
                {"match": {"content": {"query": query, "boost": 1.0}}},
                {"match": {"title": {"query": query, "boost": 1.5}}},
            ],
            "minimum_should_match": 1,
        }
    }

    body: Dict[str, Any] = {
        "size": top_k,
        "query": text_query,
        "_source": True,
        "track_total_hits": False,
    }
    if knn_clause:
        body["knn"] = knn_clause

    resp = es.search(index=index, body=body)
    return resp


