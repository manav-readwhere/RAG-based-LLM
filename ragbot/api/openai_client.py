import json
from typing import Dict, Any, Generator, List, Optional

import httpx

from .env import load_env


OPENAI_BASE_URL = "https://api.openai.com/v1"


def _auth_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def embed_text(text: str, model: Optional[str] = None) -> List[float]:
    env = load_env()
    api_key = env["OPENAI_API_KEY"]
    model_name = model or env["EMBED_MODEL"]
    url = f"{OPENAI_BASE_URL}/embeddings"
    payload = {"model": model_name, "input": text}
    with httpx.Client(timeout=60) as client:
        resp = client.post(url, headers=_auth_headers(api_key), json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]


def embed_texts(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    env = load_env()
    api_key = env["OPENAI_API_KEY"]
    model_name = model or env["EMBED_MODEL"]
    url = f"{OPENAI_BASE_URL}/embeddings"
    payload = {"model": model_name, "input": texts}
    with httpx.Client(timeout=300) as client:
        resp = client.post(url, headers=_auth_headers(api_key), json=payload)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]


def _chat_stream(messages: List[Dict[str, str]], temperature: float) -> Generator[str, None, None]:
    env = load_env()
    api_key = env["OPENAI_API_KEY"]
    model_name = env["GEN_MODEL"]
    url = f"{OPENAI_BASE_URL}/chat/completions"
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }

    with httpx.Client(timeout=None) as client:
        with client.stream("POST", url, headers=_auth_headers(api_key), json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if not line.startswith("data:") and not line.startswith("data "):
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                    continue
                data = line.split(":", 1)[1].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content


def _chat_text(messages: List[Dict[str, str]], temperature: float) -> str:
    env = load_env()
    api_key = env["OPENAI_API_KEY"]
    model_name = env["GEN_MODEL"]
    url = f"{OPENAI_BASE_URL}/chat/completions"
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }

    with httpx.Client(timeout=120) as client:
        resp = client.post(url, headers=_auth_headers(api_key), json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def chat_completion(messages: List[Dict[str, str]], stream: bool = True, temperature: float = 0.2) -> Generator[str, None, None] | str:
    if stream:
        return _chat_stream(messages, temperature)
    else:
        return _chat_text(messages, temperature)


