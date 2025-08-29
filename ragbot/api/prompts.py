from typing import Dict, List


def build_answer_prompt(query: str, passages: List[Dict]) -> List[Dict[str, str]]:
    context_blocks = []
    for i, p in enumerate(passages, start=1):
        title = p.get("title", "")
        src = p.get("source", "")
        url = p.get("url", "")
        content = p.get("content", "")
        context_blocks.append(f"[Document {i}]\nTitle: {title}\nSource: {src}\nURL: {url}\n---\n{content}")
    context_str = "\n\n".join(context_blocks)

    system = (
        "You are RAGBot, a precise assistant. Use the provided context to answer. "
        "If the answer is not in the context, say you don't know. Be concise, "
        "cite sources with [Document N] when relevant."
    )
    user = (
        f"Question: {query}\n\n"
        f"Context:\n{context_str}\n\n"
        "Provide a helpful answer."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


