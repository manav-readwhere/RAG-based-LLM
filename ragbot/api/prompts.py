from typing import Dict, List, Optional


def _format_history(messages: List[Dict[str, str]], max_chars: int = 2000) -> str:
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        lines.append(f"{role}: {content}")
        if sum(len(x) for x in lines) > max_chars:
            break
    return "\n".join(lines)


def build_answer_prompt(query: str, passages: List[Dict], history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    context_blocks = []
    for i, p in enumerate(passages, start=1):
        title = p.get("title", "")
        src = p.get("source", "")
        url = p.get("url", "")
        content = p.get("content", "")
        context_blocks.append(f"[Document {i}]\nTitle: {title}\nSource: {src}\nURL: {url}\n---\n{content}")
    context_str = "\n\n".join(context_blocks)

    system = (
        "You are RAGBot, a precise assistant. Use the provided context and the recent conversation to answer. "
        "If the answer is not in the context, say you don't know. Be concise, cite sources with [Document N]."
    )
    history_str = _format_history(history or [])
    user = (
        f"Question: {query}\n\n"
        + (f"Conversation so far:\n{history_str}\n\n" if history_str else "")
        + f"Context:\n{context_str}\n\nProvide a helpful answer."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


