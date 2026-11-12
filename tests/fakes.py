"""Подмены без сети: эмбеддинги по буквенным триграммам, модель-«попугай» и модель анализа запроса."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

from curator.memory import Turn
from curator.router import AnalysisResult, QueryAnalysis

DIM = 512


class TrigramEmbeddings(Embeddings):
    """Похожие тексты дают похожие векторы: счётчик буквенных триграмм по 512 корзинам, длина вектора 1."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        out = [0.0] * self.dim
        t = " " + " ".join(text.lower().replace("ё", "е").split()) + " "
        for i in range(len(t) - 2):
            h = int(hashlib.md5(t[i:i + 3].encode("utf-8")).hexdigest()[:8], 16)
            out[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in out)) or 1.0
        return [v / norm for v in out]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._vector(text)


def usage(input_tokens: int, output_tokens: int) -> dict[str, int]:
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}


class FakeChat(BaseChatModel):
    """Отвечает заданным текстом и запоминает промпты. fail=True — модель «не отвечает»."""

    reply: str = "Вот что сказано в материалах курса [1]."
    fail: bool = False
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        self.calls.append(list(messages))
        if self.fail:
            raise TimeoutError("модель не ответила")
        message = AIMessage(content=self.reply, usage_metadata=usage(1000, 200))  # pyright: ignore[reportArgumentType]
        return ChatResult(generations=[ChatGeneration(message=message)])

    def prompt_text(self, call: int = -1) -> str:
        return "\n\n".join(str(m.content) for m in self.calls[call])


class FakeRouterLLM(BaseChatModel):
    """Модель анализа запроса: with_structured_output отдаёт заданный разбор (или падает, или «не парсится»)."""

    analysis: QueryAnalysis | None = None
    fail: bool = False
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-router"

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        raise NotImplementedError("анализ — только через with_structured_output")

    def with_structured_output(self, schema: Any, *, include_raw: bool = False,  # pyright: ignore[reportIncompatibleMethodOverride]
                               **kwargs: Any) -> Runnable[Any, Any]:
        def run(messages: list[BaseMessage]) -> dict[str, Any]:
            self.calls.append(list(messages))
            if self.fail:
                raise ConnectionError("нет сети")
            raw = AIMessage(content="", usage_metadata=usage(300, 40))  # pyright: ignore[reportArgumentType]
            error = None if self.analysis else ValueError("не тот JSON")
            return {"raw": raw, "parsed": self.analysis, "parsing_error": error}

        return RunnableLambda(run)


class FixedAnalyzer:
    """Анализатор с заранее заданным ответом — чтобы тест куратора не зависел от правил."""

    def __init__(self, **fields: Any):
        base: dict[str, Any] = {"intent": "material", "module": None, "topic": None, "resource_types": [],
                                "level_hint": None, "uses_student_data": False, "module_overview": False}
        base.update(fields)
        self.analysis = QueryAnalysis(**base)
        self.calls: list[tuple[str, list[Turn]]] = []

    def analyze(self, question: str, history: Sequence[Turn]) -> AnalysisResult:
        self.calls.append((question, list(history)))
        return AnalysisResult(self.analysis.model_copy(), "llm", 300, 40)
