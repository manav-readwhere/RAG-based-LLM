import os
from typing import Dict, Any

from dotenv import load_dotenv


REQUIRED_KEYS = [
    "OPENAI_API_KEY",
    "ES_HOST",
    "ES_USERNAME",
    "ES_PASSWORD",
    "ES_INDEX",
    "EMBED_MODEL",
    "GEN_MODEL",
]


def load_env() -> Dict[str, Any]:
    load_dotenv(override=False)
    env = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "REDACTED"),
        "ES_HOST": os.getenv("ES_HOST", "http://localhost:9200"),
        "ES_USERNAME": os.getenv("ES_USERNAME", "elastic"),
        "ES_PASSWORD": os.getenv("ES_PASSWORD", "changeme"),
        "ES_INDEX": os.getenv("ES_INDEX", "kb_docs"),
        "EMBED_MODEL": os.getenv("EMBED_MODEL", "text-embedding-3-large"),
        "GEN_MODEL": os.getenv("GEN_MODEL", "gpt-4o"),
    }
    missing = [key for key in REQUIRED_KEYS if not env.get(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    return env


def load_env_es() -> Dict[str, Any]:
    load_dotenv(override=False)
    env = {
        "ES_HOST": os.getenv("ES_HOST", "http://localhost:9200"),
        "ES_USERNAME": os.getenv("ES_USERNAME", "elastic"),
        "ES_PASSWORD": os.getenv("ES_PASSWORD", "changeme"),
        "ES_INDEX": os.getenv("ES_INDEX", "kb_docs"),
        "EMBED_MODEL": os.getenv("EMBED_MODEL", "text-embedding-3-large"),
    }
    for key in ["ES_HOST", "ES_USERNAME", "ES_PASSWORD", "ES_INDEX"]:
        if not env.get(key):
            raise RuntimeError(f"Missing required environment variable: {key}")
    return env


def embedding_dimensions(model_name: str) -> int:
    if model_name == "text-embedding-3-large":
        return 3072
    if model_name == "text-embedding-3-small":
        return 1536
    return 3072


