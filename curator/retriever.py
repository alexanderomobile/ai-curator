"""Поиск по базе знаний с фильтром по метаданным (модуль, тип ресурса, тема) и порогом близости.

Фильтры двух видов:
- **жёсткий** (`flt`) — то, что студент сам выбрал в интерфейсе (модуль, тип материалов, тема). Выдача только
  в его пределах; если в них нашлось меньше min_filtered_hits фрагментов выше порога, он ослабляется по шагам:
  без темы → только модуль → без фильтра;
- **предпочтение** (`prefer`) — модуль, тип и тема из анализа запроса. Это приоритет, а не стена: анализ
  ошибается (вопрос о графиках к защите — модуль 4, а не модуль 3 «Визуализация»), и тогда нужный фрагмент
  лежит за его фильтром. Кандидаты — поиск с предпочтением, с его ослабленными шагами и без него (в пределах
  жёсткого фильтра); порядок — по близости с небольшой прибавкой прошедшим предпочтение (PREFER_BONUS)
  и меньшей — прошедшим его только по модулю (MODULE_BONUS). В выдачу — top_k лучших.

Общие документы (FAQ, обзор курса) помечены module=0 и проходят фильтр любого модуля. Фрагменты ниже порога
близости отбрасываются (порог — по самой близости, без прибавки): если не осталось ни одного — «в материалах курса
нет».

Эмбеддинг каждой формулировки вопроса считается один раз, все шаги ищут по готовому вектору: число вызовов
эмбеддингов не растёт с числом шагов. В Chroma коллекция с косинусным расстоянием: поиск по вектору возвращает
расстояние (меньше — ближе), близость = 1 - расстояние (это закреплено тестом).

RetrievalResult честно говорит, что попало в выдачу вне запрошенного фильтра: relaxed — есть такие фрагменты,
outside — сколько их, filter_used — самый узкий фильтр из цепочки ослабления, в который укладывается вся выдача,
steps — сколько нашёл каждый шаг и итог.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Прибавки к близости при ранжировании (в единицах косинусной близости, как MIN_SIMILARITY). Разброс близости
# у фрагментов одного вопроса — десятые доли, у соседних по смыслу разделов — 0,02-0,10: прибавка 0,05 оставляет
# первыми фрагменты из фильтра анализа, пока фрагмент вне его заметно ближе не оказался. Фрагменту, прошедшему
# фильтр только по модулю (без типа и темы), — меньшая прибавка: он ближе к цели анализа, чем чужой модуль.
PREFER_BONUS = 0.05
MODULE_BONUS = 0.02


class VectorSearch(Protocol):
    """Что нужно от векторного хранилища (langchain_chroma.Chroma)."""

    @property
    def embeddings(self) -> Embeddings | None: ...

    def similarity_search_with_score(self, query: str, k: int = ..., filter: dict[str, str] | None = ...,  # noqa: A002
                                     where_document: dict[str, str] | None = ...,
                                     **kwargs: Any) -> list[tuple[Document, float]]: ...

    def similarity_search_by_vector_with_relevance_scores(
            self, embedding: list[float], k: int = ..., filter: dict[str, str] | None = ...,  # noqa: A002
            where_document: dict[str, str] | None = ..., **kwargs: Any) -> list[tuple[Document, float]]: ...


@dataclass(frozen=True)
class SearchFilter:
    module: int | None = None
    resource_types: tuple[str, ...] = ()
    topic: str | None = None

    def where(self) -> dict[str, Any] | None:
        """Фильтр в синтаксисе Chroma: одно условие — как есть, несколько — через $and."""
        conditions: list[dict[str, Any]] = []
        if self.module is not None:
            conditions.append({"module": {"$in": [self.module, 0]}})
        if self.resource_types:
            conditions.append({"resource_type": {"$in": list(self.resource_types)}})
        if self.topic:
            conditions.append({"topic": self.topic})
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        """Проходит ли фрагмент с такими метаданными этот фильтр — то же, что where() в Chroma."""
        if self.module is not None and metadata.get("module") not in (self.module, 0):
            return False
        if self.resource_types and metadata.get("resource_type") not in self.resource_types:
            return False
        return not (self.topic and metadata.get("topic") != self.topic)

    def merged(self, other: SearchFilter) -> SearchFilter:
        """Этот фильтр, дополненный полями other там, где у этого пусто (поля этого важнее)."""
        return SearchFilter(self.module if self.module is not None else other.module,
                            self.resource_types or other.resource_types, self.topic or other.topic)

    def relaxed(self) -> list[SearchFilter]:
        """Шаги ослабления: полный фильтр → без темы → только модуль → без фильтра (повторы убираются)."""
        steps = [self, SearchFilter(self.module, self.resource_types), SearchFilter(self.module), SearchFilter()]
        unique: list[SearchFilter] = []
        for step in steps:
            if step not in unique:
                unique.append(step)
        return unique

    def describe(self) -> str:
        parts = []
        if self.module is not None:
            parts.append(f"модуль {self.module}")
        if self.resource_types:
            parts.append("тип " + "/".join(self.resource_types))
        if self.topic:
            parts.append(f"тема {self.topic}")
        return ", ".join(parts) or "без фильтра"


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    score: float  # косинусная близость, 1 — совпадение (без прибавки за фильтр)
    metadata: dict[str, Any]
    bonus: float = 0.0  # прибавка за фильтр анализа — только для порядка выдачи

    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id", ""))

    @property
    def section(self) -> str:
        return str(self.metadata.get("section", ""))

    @property
    def rank(self) -> float:
        return self.score + self.bonus


@dataclass
class RetrievalResult:
    hits: list[Hit]
    filter_requested: SearchFilter  # интерфейс + анализ: что хотели найти
    filter_used: SearchFilter  # самый узкий фильтр цепочки ослабления, в который укладывается вся выдача
    relaxed: bool = False  # в выдаче есть фрагменты вне запрошенного фильтра
    steps: list[str] = field(default_factory=list)
    filter_hard: SearchFilter = field(default_factory=SearchFilter)  # выбор в интерфейсе
    outside: int = 0  # сколько фрагментов выдачи вне запрошенного фильтра

    @property
    def top_score(self) -> float | None:
        """Лучшая близость в выдаче (без прибавок за фильтр)."""
        return round(max(h.score for h in self.hits), 3) if self.hits else None


class Retriever:
    def __init__(self, store: VectorSearch, top_k: int, min_similarity: float, min_filtered_hits: int, *,
                 prefer_bonus: float = PREFER_BONUS, module_bonus: float = MODULE_BONUS):
        self.store = store
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.min_filtered_hits = min_filtered_hits
        self.prefer_bonus = prefer_bonus
        self.module_bonus = module_bonus

    def _vectors(self, queries: Sequence[str]) -> list[list[float]]:
        embeddings = self.store.embeddings
        if embeddings is None:
            raise RuntimeError("у хранилища нет модели эмбеддингов")
        return [embeddings.embed_query(q) for q in queries]

    def _search(self, vectors: Sequence[list[float]], flt: SearchFilter) -> dict[str, Hit]:
        """Все формулировки вопроса: у фрагмента — лучшая близость из всех запросов."""
        where = cast("dict[str, str] | None", flt.where())
        best: dict[str, Hit] = {}
        for vector in vectors:
            found = self.store.similarity_search_by_vector_with_relevance_scores(vector, k=self.top_k, filter=where)
            for doc, distance in found:
                similarity = 1.0 - float(distance)  # коллекция с косинусным расстоянием
                if similarity < self.min_similarity:
                    continue
                chunk_id = str(doc.id or "")
                if chunk_id not in best or similarity > best[chunk_id].score:
                    best[chunk_id] = Hit(chunk_id, doc.page_content, similarity, dict(doc.metadata))
        return best

    def _bonus(self, hit: Hit, wanted: SearchFilter, hard: SearchFilter) -> float:
        if wanted == hard:  # предпочтений сверх интерфейса нет — всем поровну
            return 0.0
        if wanted.matches(hit.metadata):
            return self.prefer_bonus
        if wanted.module is not None and hard.module is None and SearchFilter(wanted.module).matches(hit.metadata):
            return self.module_bonus
        return 0.0

    def search(self, query: str | Sequence[str], flt: SearchFilter | None = None,
               prefer: SearchFilter | None = None) -> RetrievalResult:
        """query — вопрос или несколько формулировок (уточнение само и вместе с прошлым вопросом);
        flt — жёсткий фильтр (интерфейс), prefer — предпочтение (анализ запроса)."""
        queries = [query] if isinstance(query, str) else list(dict.fromkeys(query))
        hard = flt or SearchFilter()
        soft = prefer or SearchFilter()
        requested = hard.merged(soft)
        vectors = self._vectors(queries)
        steps: list[str] = []
        collected: dict[str, Hit] = {}
        chain: list[SearchFilter] = []
        for hard_step in hard.relaxed():
            # Предпочтение и его ослабленные шаги в пределах шага жёсткого фильтра; последний — сам жёсткий шаг.
            chain = list(dict.fromkeys(hard_step.merged(s) for s in soft.relaxed()))
            found: dict[str, Hit] = {}
            for level in chain:
                hits = self._search(vectors, level)
                steps.append(f"{level.describe()}: {len(hits)}")
                for chunk_id, hit in hits.items():
                    found.setdefault(chunk_id, hit)
            for chunk_id, hit in found.items():
                collected.setdefault(chunk_id, hit)
            if len(found) >= self.min_filtered_hits:
                return self._result(found, requested, hard, hard_step, chain, steps)
        # Ни один шаг не дал достаточно: лучшее из всего найденного (последний шаг — без фильтра).
        return self._result(collected, requested, hard, SearchFilter(), chain or [SearchFilter()], steps)

    def _result(self, found: Mapping[str, Hit], requested: SearchFilter, hard: SearchFilter, hard_step: SearchFilter,
                chain: Sequence[SearchFilter], steps: list[str]) -> RetrievalResult:
        wanted = hard_step.merged(requested)
        ranked = sorted((replace(h, bonus=self._bonus(h, wanted, hard_step)) for h in found.values()),
                        key=lambda h: (h.rank, h.score), reverse=True)[: self.top_k]
        outside = sum(1 for h in ranked if not requested.matches(h.metadata))
        covering = [*chain, SearchFilter()]
        used = next((f for f in covering if all(f.matches(h.metadata) for h in ranked)), SearchFilter())
        if outside:
            steps.append(f"в выдаче вне фильтра «{requested.describe()}»: {outside} из {len(ranked)}")
        return RetrievalResult(ranked, requested, used, relaxed=outside > 0, steps=steps, filter_hard=hard,
                               outside=outside)
