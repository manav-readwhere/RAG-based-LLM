from typing import Dict, List

from .embeddings import embed_text
from .elastic import search


def retrieve(query: str, top_k: int = 8) -> List[Dict]:
    qvec = embed_text(query)
    resp = search(query=query, query_vector=qvec, top_k=top_k)
    hits = resp.get("hits", {}).get("hits", [])
    results: List[Dict] = []
    for h in hits:
        src = h.get("_source", {})
        score = h.get("_score", 0.0)
        results.append({"score": score, **src})
    return results


