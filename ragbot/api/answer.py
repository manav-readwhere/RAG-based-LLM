from typing import Dict, Generator, List

from .openai_client import chat_completion


def answer_question(messages: List[Dict[str, str]]) -> Generator[str, None, None]:
    for token in chat_completion(messages, stream=True):
        yield token


