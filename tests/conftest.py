"""Общие фикстуры. Сеть не нужна: ключ OpenAI подменён, адрес API ведёт в никуда — случайный настоящий вызов
упадёт сразу, а не потратит деньги. Индекс фикстурной базы знаний строится один раз на сессию."""
from __future__ import annotations

import os

os.environ["OPENAI_API_KEY"] = "sk-test-not-a-real-key"
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"
for _var in ("LMS_URL", "CURATOR_NOW", "CURATOR_DATA_DIR", "CURATOR_INDEX_DIR", "CURATOR_JOURNAL", "SHOW_LIVE_TEXTS",
             "CURATOR_SOURCE", "CHAT_MODEL", "EMBED_MODEL", "MIN_SIMILARITY", "USER_HASH_SALT",
             "CURATOR_TRUSTED_PROXIES"):
    os.environ.pop(_var, None)

from collections.abc import Callable  # noqa: E402
from dataclasses import replace  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

from curator.config import MSK, Settings  # noqa: E402
from curator.curator import Curator  # noqa: E402
from curator.fewshot import FewShotLibrary, load_examples  # noqa: E402
from curator.ingest import build_index, open_store  # noqa: E402
from curator.journal import Journal  # noqa: E402
from curator.kb import KnowledgeBase, load_kb  # noqa: E402
from curator.limits import DemoLimits  # noqa: E402
from curator.lms import FileBackend, LMSClient  # noqa: E402
from curator.memory import DialogMemory  # noqa: E402
from curator.prompts import PromptLibrary  # noqa: E402
from curator.recommender import Recommender, load_resources  # noqa: E402
from curator.retriever import Retriever  # noqa: E402
from curator.router import Analyzer, RuleAnalyzer  # noqa: E402
from tests.fakes import FakeChat, TrigramEmbeddings  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 12, 14, 0, tzinfo=MSK)  # четверг; до дедлайна hw1 (15.11 23:59) — 3 дня
EMBED_MODEL = "fake-trigram"
# Порог для триграммных эмбеддингов: у вопросов по теме фикстур близость 0,3-0,6, у бессмыслицы — около 0,05.
MIN_SIMILARITY = 0.2


@pytest.fixture(scope="session")
def fixture_index(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    index_dir = tmp_path_factory.mktemp("index")
    manifest = build_index(FIXTURES / "kb", index_dir, TrigramEmbeddings(), embed_model=EMBED_MODEL,
                           chunk_size=1200, chunk_overlap=150)
    return index_dir, manifest


@pytest.fixture
def kb() -> KnowledgeBase:
    return load_kb(FIXTURES / "kb")


@pytest.fixture
def settings(tmp_path: Path, fixture_index: tuple[Path, dict[str, Any]]) -> Settings:
    return Settings(openai_api_key="sk-test", data_dir=FIXTURES, index_dir=fixture_index[0],
                    journal_path=tmp_path / "journal.db", exports_dir=tmp_path / "exports", now_override=NOW,
                    user_hash_salt="test-salt", embed_model=EMBED_MODEL, min_similarity=MIN_SIMILARITY)


@pytest.fixture
def retriever(fixture_index: tuple[Path, dict[str, Any]]) -> Retriever:
    index_dir, manifest = fixture_index
    store = open_store(index_dir, manifest["collection"], TrigramEmbeddings())
    return Retriever(store, top_k=5, min_similarity=MIN_SIMILARITY, min_filtered_hits=2)


BuildCurator = Callable[..., tuple[Curator, FakeChat, Journal]]


@pytest.fixture
def build_curator(settings: Settings, kb: KnowledgeBase, retriever: Retriever) -> BuildCurator:
    def _build(*, chat: FakeChat | None = None, analyzer: Analyzer | None = None, limits: DemoLimits | None = None,
               **overrides: Any) -> tuple[Curator, FakeChat, Journal]:
        s = replace(settings, **overrides) if overrides else settings
        chat = chat or FakeChat()
        journal = Journal(s.journal_path, s.retention_days)
        examples, _ = load_examples(s.fewshot_path)
        lecture_topics = {d.doc_id: d.topic for d in kb.documents if d.resource_type == "lecture"}
        curator = Curator(
            settings=s, analyzer=analyzer or RuleAnalyzer(list(kb.topics()), lecture_topics), retriever=retriever,
            llm=chat, lms=LMSClient(FileBackend(s.lms_dir)), kb=kb,
            recommender=Recommender(load_resources(s.resources_path), kb), fewshot=FewShotLibrary(examples),
            prompts=PromptLibrary(), memory=DialogMemory(turns=2), journal=journal, limits=limits,
            kb_version="test",
        )
        return curator, chat, journal

    return _build
