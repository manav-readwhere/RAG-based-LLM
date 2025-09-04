from typing import Dict, List

from .elastic import search


def retrieve(query: str, top_k: int = 10000) -> List[Dict]:
    # Pure keyword retrieval across the alias; no kNN constraint
    resp = search(query=query, query_vector=None, top_k=top_k)

    hits = resp.get("hits", {}).get("hits", [])
    results: List[Dict] = []
    for h in hits:
        src = h.get("_source", {})
        score = h.get("_score", 0.0)
        results.append({"score": score, **src})
    return results


