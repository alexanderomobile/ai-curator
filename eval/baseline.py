"""Наивный RAG — точка сравнения для куратора.

Тот же индекс Chroma и та же модель эмбеддингов, что у куратора, а всё остальное убрано: поиск top-5 по тексту
вопроса без фильтров метаданных (и без порога близости), нет анализа запроса, данных LMS, few-shot, ограничений
в коде и блока «Источники». Системный промпт — одна строка, модель gpt-4.1-mini, temperature 0.2, тот же потолок
длины ответа, что у куратора. Реплики диалога передаются модели как есть (история чата есть у любого чат-бота),
в поиск идёт только текущий вопрос. Уровень, модуль и студент из интерфейса наивному RAG неизвестны.

    python -m eval.baseline "Когда дедлайн второго ДЗ?"
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from curator.config import ConfigError, Settings, load_settings
from curator.factory import make_embeddings
from curator.ingest import index_status, open_store
from curator.retriever import VectorSearch

SYSTEM_PROMPT = "Ты помощник курса. Отвечай по контексту."
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.2
TOP_K = 5
CHUNK_SEPARATOR = "\n\n---\n\n"


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts = cast(list[Any], content)
        return "".join(p if isinstance(p, str) else str(cast(dict[str, Any], p).get("text", "")) for p in parts)
    return str(content)


def usage_tokens(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        return 0, 0
    data = cast(dict[str, Any], usage)
    return int(data.get("input_tokens", 0) or 0), int(data.get("output_tokens", 0) or 0)


def finish_reason(message: Any) -> str:
    """Почему модель остановилась: stop — закончила сама, length — упёрлась в потолок длины (ответ обрезан)."""
    meta = getattr(message, "response_metadata", None) or {}
    return str(cast(dict[str, Any], meta).get("finish_reason") or "") if isinstance(meta, dict) else ""


def hit_sources(found: Sequence[tuple[Document, float]]) -> list[dict[str, Any]]:
    """Разделы, попавшие в контекст (без повторов): в ответе наивного RAG их нет, но проверкам они нужны."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for doc, distance in found:
        meta = doc.metadata
        key = (str(meta.get("doc_id", "")), str(meta.get("section", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append({"kind": str(meta.get("resource_type", "")), "id": key[0], "section": key[1],
                    "title": str(meta.get("title", "")), "score": round(1.0 - float(distance), 3)})
    return out


@dataclass
class BaselineAnswer:
    text: str
    sources: list[dict[str, Any]] = field(default_factory=list)  # что было в контексте, а не что показано
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    finish_reason: str = ""


class BaselineRAG:
    def __init__(self, store: VectorSearch, llm: BaseChatModel, *, top_k: int = TOP_K,
                 price: Callable[[int, int], float] | None = None, model: str = MODEL):
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.price = price
        self.model = model

    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        return self.store.similarity_search_with_score(question, k=self.top_k)

    @staticmethod
    def messages(question: str, history: Sequence[tuple[str, str]], docs: Sequence[Document]) -> list[BaseMessage]:
        context = CHUNK_SEPARATOR.join(d.page_content for d in docs)
        out: list[BaseMessage] = [SystemMessage(f"{SYSTEM_PROMPT}\n\nКонтекст:\n\n{context}")]
        for q, a in history:
            out.append(HumanMessage(q))
            out.append(AIMessage(a))
        out.append(HumanMessage(question))
        return out

    def answer(self, question: str, history: Sequence[tuple[str, str]] = ()) -> BaselineAnswer:
        found = self.retrieve(question)
        reply = self.llm.invoke(self.messages(question, history, [d for d, _ in found]))
        prompt_tokens, completion_tokens = usage_tokens(reply)
        cost = self.price(prompt_tokens, completion_tokens) if self.price else 0.0
        return BaselineAnswer(message_text(reply).strip(), hit_sources(found), prompt_tokens, completion_tokens, cost,
                              finish_reason(reply))


def make_baseline_llm(settings: Settings) -> BaseChatModel:
    """Модель и температура закреплены (как у куратора по умолчанию); таймаут, повторы и потолок длины — его же."""
    return ChatOpenAI(model=MODEL, temperature=TEMPERATURE, max_completion_tokens=settings.max_tokens,
                      timeout=settings.llm_timeout_s, max_retries=2, api_key=SecretStr(settings.openai_api_key))


def build_baseline(settings: Settings) -> BaselineRAG:
    if not settings.openai_api_key:
        raise ConfigError("Нет OPENAI_API_KEY: задайте его в .env в корне проекта.")
    status = index_status(settings.kb_dir, settings.index_dir, settings.embed_model, settings.chunk_size,
                          settings.chunk_overlap)
    if not status.usable or status.manifest is None or status.manifest.get("embed_model") != settings.embed_model:
        raise ConfigError(status.message)
    store = open_store(settings.index_dir, str(status.manifest["collection"]), make_embeddings(settings))
    return BaselineRAG(store, make_baseline_llm(settings), price=settings.cost_usd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.baseline", description="Наивный RAG для сравнения")
    parser.add_argument("question")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]
    try:
        baseline = build_baseline(load_settings())
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    answer = baseline.answer(args.question)
    print(answer.text)
    print("\n---")
    print("в контексте: " + ", ".join(f"{s['id']} «{s['section']}» ({s['score']})" for s in answer.sources))
    print(f"токены: {answer.prompt_tokens}+{answer.completion_tokens}; ${answer.cost_usd:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
