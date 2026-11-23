"""Shared test helpers: a scripted fake LLM and a fake retriever, so the LangGraph workflow
can be tested deterministically without API keys or network access."""
import hashlib
import math
import re

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage


class FakeLLM:
    """Stands in for the chat model. Returns scripted AIMessages in order and records
    every list of messages it was called with."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("FakeLLM was called more times than scripted")
        return self.responses.pop(0)


class FakeRetriever:
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else [Document(page_content="NovaLoop offers 20 days of paid time off.")]
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.docs


class KeywordEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings (hashed into 256 dims). No model download,
    but similar wording still gives similar vectors, so retrieval can be tested offline."""

    DIM = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def tool_call(name, args, call_id="call_1"):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def ai_tool_calls(*calls):
    return AIMessage(content="", tool_calls=list(calls))


@pytest.fixture
def answers_file(tmp_path):
    return tmp_path / "answers.txt"
