from typing import Dict, List, Optional


def build_answer_prompt(query: str, passages: List[Dict], history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    context_blocks = []
    for i, p in enumerate(passages, start=1):
        title = p.get("title", "")
        src = p.get("source", "")
        url = p.get("url", "")
        content = p.get("content", "")
        context_blocks.append(f"Title: {title}\nSource: {src}\nURL: {url}\n---\n{content}")
    context_str = "\n\n".join(context_blocks)

    system = (
        "You are RAGBot, a precise assistant. Use only the provided context and the current question to answer. "
        "If an 'Aggregations' document is present, treat its numeric values as ground truth. "
        "If the answer is not in the context, just suggest a basic answer."
    )
    user = (
        f"Question: {query}\n\n"
        + f"Context:\n{context_str}\n\nProvide a helpful answer."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


