"""Поиск: фильтр анализа — приоритет, а не стена; фильтр интерфейса — жёсткий; эмбеддинг вопроса — один раз;
поиск по вектору в Chroma возвращает расстояние; выдача честно говорит, что в неё попало вне фильтра."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from curator.ingest import open_store
from curator.retriever import MODULE_BONUS, PREFER_BONUS, Retriever, SearchFilter
from tests.fakes import TrigramEmbeddings


class _Unit(Embeddings):
    """Вектор не важен: близость у фейкового хранилища задана заранее."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [1.0]


def _matches(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Подмножество синтаксиса where в Chroma: $and, $in и равенство."""
    if not where:
        return True
    if "$and" in where:
        return all(_matches(meta, part) for part in where["$and"])
    key, cond = next(iter(where.items()))
    if isinstance(cond, dict) and "$in" in cond:
        return meta.get(key) in cond["$in"]
    return meta.get(key) == cond


class FakeStore:
    """Хранилище с заданной близостью фрагментов; возвращает расстояние (1 - близость), как Chroma."""

    def __init__(self, chunks: list[tuple[str, dict[str, Any], float]]):
        self.chunks = chunks
        self._embeddings = _Unit()
        self.queries: list[dict[str, Any] | None] = []

    @property
    def embeddings(self) -> Embeddings | None:
        return self._embeddings

    def similarity_search_with_score(self, query: str, k: int = 4, filter: dict[str, str] | None = None,  # noqa: A002
                                     where_document: dict[str, str] | None = None,
                                     **kwargs: Any) -> list[tuple[Document, float]]:
        raise AssertionError("поиск должен идти по готовому вектору")

    def similarity_search_by_vector_with_relevance_scores(
            self, embedding: list[float], k: int = 4, filter: dict[str, str] | None = None,  # noqa: A002
            where_document: dict[str, str] | None = None, **kwargs: Any) -> list[tuple[Document, float]]:
        self.queries.append(filter)
        found = [(Document(page_content=cid, metadata=meta, id=cid), 1.0 - sim) for cid, meta, sim in self.chunks
                 if _matches(meta, filter)]
        return sorted(found, key=lambda pair: pair[1])[:k]


def _meta(module: int, rtype: str, topic: str) -> dict[str, Any]:
    return {"doc_id": f"{rtype}-{module}-{topic}", "module": module, "resource_type": rtype, "topic": topic,
            "section": topic}


CHUNKS = [
    ("in-filter", _meta(3, "lecture", "charts"), 0.40),          # прошёл фильтр анализа целиком
    ("same-module", _meta(3, "lecture", "seaborn"), 0.41),       # только модуль (и тип)
    ("other-close", _meta(4, "lecture", "defense"), 0.50),       # чужой модуль, заметно ближе
    ("other-near", _meta(4, "lecture", "report"), 0.425),        # чужой модуль, чуть ближе
    ("faq", _meta(0, "faq", "schedule"), 0.39),
    ("below", _meta(3, "lecture", "charts"), 0.28),              # ниже порога даже с прибавкой
]
PREFER = SearchFilter(3, ("lecture",), "charts")


def test_bonus_values_are_small_and_ordered():
    assert 0 < MODULE_BONUS < PREFER_BONUS <= 0.1


def test_analysis_filter_is_priority_not_wall():
    result = Retriever(FakeStore(CHUNKS), 5, 0.30, 2).search("вопрос", prefer=PREFER)
    ids = [h.chunk_id for h in result.hits]
    # 0.50 (чужой модуль) > 0.40+0.05 (фильтр анализа) > 0.39+0.02 (FAQ: модуль 0 проходит фильтр модуля)
    # = 0.41 (тот же модуль, другая тема: +0.02 → 0.43) ...
    assert ids[0] == "other-close" and ids[1] == "in-filter"
    assert ids == ["other-close", "in-filter", "same-module", "other-near", "faq"]
    assert "below" not in ids  # порог — по близости без прибавки
    assert [h.score for h in result.hits][:2] == [0.50, 0.40]  # близость в выдаче — без прибавки
    assert result.top_score == 0.5
    assert result.relaxed and result.outside == 4
    assert result.filter_requested == PREFER and result.filter_used == SearchFilter()
    assert result.steps[-1] == "в выдаче вне фильтра «модуль 3, тип lecture, тема charts»: 4 из 5"


def test_close_in_filter_fragment_stays_first():
    """Фрагмент вне фильтра, ближе меньше чем на прибавку, первым не становится."""
    chunks = [("in", _meta(3, "lecture", "charts"), 0.48), ("out", _meta(4, "lecture", "x"), 0.52)]
    assert [h.chunk_id for h in Retriever(FakeStore(chunks), 5, 0.3, 2).search("q", prefer=PREFER).hits] == \
        ["in", "out"]


def test_ui_filter_is_hard():
    store = FakeStore(CHUNKS)
    result = Retriever(store, 5, 0.30, 2).search("вопрос", SearchFilter(module=3), PREFER)
    assert {h.metadata["module"] for h in result.hits} <= {3, 0}  # модуль 4 не пролез, хотя ближе
    assert [h.chunk_id for h in result.hits][:2] == ["in-filter", "same-module"]
    assert result.filter_hard == SearchFilter(module=3)
    assert all(q is not None for q in store.queries)  # без фильтра модуля не искали


def test_ui_filter_relaxes_when_nothing_found():
    result = Retriever(FakeStore(CHUNKS), 5, 0.30, 2).search("вопрос", SearchFilter(resource_types=("guide",)))
    assert result.hits and result.relaxed
    assert result.steps[0] == "тип guide: 0"


def test_no_preference_means_plain_similarity_order():
    result = Retriever(FakeStore(CHUNKS), 5, 0.30, 2).search("вопрос")
    assert [h.chunk_id for h in result.hits] == ["other-close", "other-near", "same-module", "in-filter", "faq"]
    assert not result.relaxed and all(h.bonus == 0 for h in result.hits)
    assert result.filter_used == SearchFilter() and result.outside == 0


def test_all_in_filter_is_not_relaxed():
    chunks = [("a", _meta(3, "lecture", "charts"), 0.6), ("b", _meta(3, "lecture", "charts"), 0.5)]
    result = Retriever(FakeStore(chunks), 5, 0.3, 2).search("q", prefer=PREFER)
    assert not result.relaxed and result.outside == 0 and result.filter_used == PREFER


def test_each_query_embedded_once_for_all_steps():
    store = FakeStore(CHUNKS)
    Retriever(store, 5, 0.30, 2).search(["вопрос", "прошлый вопрос\nвопрос"], prefer=PREFER)
    assert store._embeddings.calls == 2  # pyright: ignore[reportPrivateUsage]
    assert len(store.queries) == 8  # 4 шага × 2 формулировки — по готовым векторам


# --- настоящая Chroma: поиск по вектору возвращает расстояние, а не близость.

def test_chroma_vector_search_returns_cosine_distance(fixture_index: tuple[Path, dict[str, Any]]):
    index_dir, manifest = fixture_index
    embeddings = TrigramEmbeddings()
    store = open_store(index_dir, manifest["collection"], embeddings)
    question = "Можно ли перенести дедлайн?"
    vector = embeddings.embed_query(question)
    by_vector = store.similarity_search_by_vector_with_relevance_scores(vector, k=3)
    by_text = store.similarity_search_with_score(question, k=3)
    assert [d.id for d, _ in by_vector] == [d.id for d, _ in by_text]
    assert [round(s, 6) for _, s in by_vector] == [round(s, 6) for _, s in by_text]
    best, distance = by_vector[0]
    cosine = sum(a * b for a, b in zip(vector, embeddings.embed_documents([best.page_content])[0], strict=True))
    assert math.isclose(1.0 - distance, cosine, abs_tol=1e-4)  # меньше — ближе: это расстояние
    assert distance < by_vector[-1][1]


def test_real_store_search_embeds_question_once(fixture_index: tuple[Path, dict[str, Any]]):
    index_dir, manifest = fixture_index
    embeddings = TrigramEmbeddings()
    retriever = Retriever(open_store(index_dir, manifest["collection"], embeddings), 5, 0.2, 2)
    result = retriever.search("Можно ли перенести дедлайн?",
                              prefer=SearchFilter(2, ("lecture",), "pandas-filtering"))
    assert embeddings.calls == 1
    # Анализ ошибся (модуль 2, лекции), но ответ — в FAQ: он в выдаче, и это отражено.
    assert result.hits[0].section == "Можно ли перенести дедлайн?"
    assert result.relaxed and result.outside >= 1


@pytest.mark.parametrize("flt, meta, expected", [
    (SearchFilter(2), {"module": 0}, True), (SearchFilter(2), {"module": 3}, False),
    (SearchFilter(None, ("faq",)), {"resource_type": "faq"}, True),
    (SearchFilter(None, (), "t"), {"topic": "u"}, False), (SearchFilter(), {}, True),
])
def test_filter_matches_like_where(flt: SearchFilter, meta: dict[str, Any], expected: bool):
    assert flt.matches(meta) is expected
    assert _matches(meta, flt.where()) is expected


def test_merged_keeps_own_fields():
    hard = SearchFilter(None, ("faq",), None)
    assert hard.merged(SearchFilter(2, ("lecture",), "t")) == SearchFilter(2, ("faq",), "t")
