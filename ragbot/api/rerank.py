from typing import Dict, List, Tuple

import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def rerank(query_vector: List[float], docs: List[Dict], lambda_mult: float = 0.7, top_k: int = 8) -> List[Dict]:
    # Maximal Marginal Relevance
    if not docs:
        return []
    q = np.array(query_vector, dtype=np.float32)
    embeddings = [np.array(d.get("embedding", []), dtype=np.float32) for d in docs]
    selected: List[int] = []
    remaining = set(range(len(docs)))

    sim_to_query = [
        _cosine_similarity(q, emb) if emb.size > 0 else 0.0 for emb in embeddings
    ]

    while remaining and len(selected) < top_k:
        if not selected:
            idx = int(np.argmax(sim_to_query))
            selected.append(idx)
            remaining.remove(idx)
            continue
        # compute marginal score
        best_idx = None
        best_score = -1e9
        for i in list(remaining):
            max_sim_selected = max(
                (
                    _cosine_similarity(embeddings[i], embeddings[j])
                    if embeddings[i].size > 0 and embeddings[j].size > 0
                    else 0.0
                )
                for j in selected
            ) if selected else 0.0
            score = lambda_mult * sim_to_query[i] - (1 - lambda_mult) * max_sim_selected
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [docs[i] for i in selected]


