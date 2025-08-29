from typing import List

from .openai_client import embed_text as _embed_text, embed_texts as _embed_texts


def embed_text(text: str) -> List[float]:
    return _embed_text(text)


def embed_many(texts: List[str]) -> List[List[float]]:
    return _embed_texts(texts)


