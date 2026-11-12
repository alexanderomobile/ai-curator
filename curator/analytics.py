"""Аналитика журнала: KPI и разрезы для дашборда (app/views/1_Аналитика.py) и отчёта (curator.report).

Всё считается из обезличенных записей журнала, на чистом Python: записей — тысячи, а не миллионы, и так результат
проще проверять тестами. Разрезы возвращаются списками словарей — дашборд превращает их в таблицы и графики.

Тексты вопросов отдаются только там, где их можно показывать: source=eval/demo всегда, source=live — только если
это явно разрешено (SHOW_LIVE_TEXTS у дашборда, --include-live-texts у отчёта).
«Пробел в контенте» — вопрос, на который поиск не нашёл материалов (refusal=no_context): сигнал дописать базу.

Период считается от настоящего времени (wall_now), а не от CURATOR_NOW: журнал пишет настоящее время, а CURATOR_NOW
застывает на демо-дате только для расчёта сроков курса. Выборка ограничена с двух сторон: [сейчас − N дней, сейчас).
«Ответ по материалам» — только вопросы без отказа: отказ по правилам (оценка, перенос) ответом не считается, даже
если в старых записях журнала у него стоит answered=1.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .config import MSK
from .journal import Journal

INTENT_LABELS = {
    "deadline": "Дедлайны", "difficulty": "Сложность материала", "material": "Вопрос по материалу",
    "recommendation": "Рекомендации", "organizational": "Организационные", "grading_request": "Просьба оценить",
    "schedule_change": "Просьба перенести", "off_topic": "Не по курсу", "": "—",
}
REFUSAL_LABELS = {
    "grading_request": "Оценку ставит преподаватель", "schedule_change": "Сроки не меняются куратором",
    "off_topic": "Не по курсу", "small_talk": "Приветствие / благодарность", "no_context": "Нет в материалах",
    "output_guard": "Ответ модели заменён (ограничение)", "llm_error": "Модель не ответила",
    "search_error": "Поиск по материалам не ответил",
    "limit_too_long": "Лимит: длинный вопрос", "limit_per_hour": "Лимит: вопросов в час",
    "limit_hourly": "Лимит: общий в час", "limit_daily": "Лимит: дневной", "limit_empty": "Пустой вопрос",
}
RESTRICTION_REFUSALS = ("grading_request", "schedule_change", "off_topic", "output_guard")
TEXT_SOURCES = ("eval", "demo")
NO_TOPIC = "(тема не определена)"


@dataclass(frozen=True)
class Row:
    id: int
    ts: datetime
    session: str
    source: str
    question: str
    intent: str
    module: int | None
    topic: str
    level: str
    refusal: str
    answered: bool
    rating: int | None
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    lectures: tuple[str, ...]
    analysis_method: str

    @property
    def day(self) -> date:
        return self.ts.astimezone(MSK).date()

    @property
    def is_limit(self) -> bool:
        return self.refusal.startswith("limit_")

    @property
    def answered_by_materials(self) -> bool:
        """Ответ опирается на материалы курса: отказы (в том числе по правилам) сюда не входят."""
        return self.answered and not self.refusal

    def text_visible(self, show_live: bool) -> bool:
        return self.source in TEXT_SOURCES or (show_live and self.source == "live")


def _ts(value: str) -> datetime:
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _lectures(raw: Any) -> tuple[str, ...]:
    try:
        items = json.loads(raw) if isinstance(raw, str) else []
    except ValueError:
        return ()
    if not isinstance(items, list):
        return ()
    return tuple(str(i.get("id")) for i in items if isinstance(i, dict) and i.get("kind") == "lecture")


def to_rows(records: Iterable[dict[str, Any]]) -> list[Row]:
    rows = []
    for r in records:
        rows.append(Row(
            id=int(r.get("id") or 0), ts=_ts(str(r["ts"])), session=str(r.get("session") or ""),
            source=str(r.get("source") or "live"), question=str(r.get("question") or ""),
            intent=str(r.get("intent") or ""), module=int(r["module"]) if r.get("module") is not None else None,
            topic=str(r.get("topic") or ""), level=str(r.get("level") or ""), refusal=str(r.get("refusal") or ""),
            answered=bool(r.get("answered")), rating=int(r["rating"]) if r.get("rating") is not None else None,
            latency_ms=int(r.get("latency_ms") or 0), prompt_tokens=int(r.get("prompt_tokens") or 0),
            completion_tokens=int(r.get("completion_tokens") or 0), cost_usd=float(r.get("cost_usd") or 0.0),
            lectures=_lectures(r.get("sources")), analysis_method=str(r.get("analysis_method") or ""),
        ))
    return rows


def wall_now() -> datetime:
    """Настоящее время в МСК — граница периода дашборда и отчёта (тесты подменяют эту функцию)."""
    return datetime.now(MSK)


def load(journal: Journal, days: int, now: datetime | None = None, sources: tuple[str, ...] = ()) -> list[Row]:
    """Записи за [now − days, now); now по умолчанию — настоящее время."""
    moment = now or wall_now()
    return to_rows(journal.fetch(moment - timedelta(days=days), sources, until=moment))


def questions(rows: Iterable[Row]) -> list[Row]:
    """Вопросы, дошедшие до куратора (без отказов по лимитам демо)."""
    return [r for r in rows if not r.is_limit]


def quantile(values: list[float], q: float) -> float | None:
    """Линейная интерполяция между соседними значениями (как numpy.quantile по умолчанию)."""
    if not values:
        return None
    data = sorted(values)
    pos = (len(data) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return data[low] + (data[high] - data[low]) * (pos - low)


def _share(part: int, total: int) -> float | None:
    return part / total if total else None


def kpis(rows: list[Row]) -> dict[str, Any]:
    q = questions(rows)
    total = len(q)
    rated = [r.rating for r in q if r.rating is not None]
    latency = [float(r.latency_ms) for r in q if r.latency_ms > 0]
    cost = sum(r.cost_usd for r in q)
    return {
        "total": total,
        "sessions": len({r.session for r in q}),
        "answered_share": _share(sum(r.answered_by_materials for r in q), total),
        "no_context": sum(r.refusal == "no_context" for r in q),
        "refusals": sum(r.refusal in RESTRICTION_REFUSALS for r in q),
        "limited": sum(r.is_limit for r in rows),
        "rated": len(rated),
        "positive_share": _share(sum(v > 0 for v in rated), len(rated)),
        "median_latency_ms": quantile(latency, 0.5),
        "p95_latency_ms": quantile(latency, 0.95),
        "tokens": sum(r.prompt_tokens + r.completion_tokens for r in q),
        "cost_usd": cost,
        "cost_per_question": cost / total if total else None,
        "rules_share": _share(sum(r.analysis_method == "rules" for r in q), total),
    }


def _group(rows: Iterable[Row], key: Callable[[Row], Any]) -> dict[Any, list[Row]]:
    groups: dict[Any, list[Row]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    return groups


def _negative(rows: list[Row]) -> int:
    return sum(1 for r in rows if r.rating is not None and r.rating < 0)


def by_intent(rows: list[Row]) -> list[dict[str, Any]]:
    out = []
    for intent, group in _group(questions(rows), lambda r: r.intent).items():
        out.append({"intent": intent, "label": INTENT_LABELS.get(intent, intent), "questions": len(group),
                    "answered": _share(sum(r.answered_by_materials for r in group), len(group)),
                    "negative": _negative(group)})
    return sorted(out, key=lambda d: (-d["questions"], d["intent"]))


def _module_key(r: Row) -> str:
    return str(r.module) if r.module is not None else "—"


def by_module(rows: list[Row]) -> list[dict[str, Any]]:
    out = [{"module": module, "questions": len(group), "no_context": sum(r.refusal == "no_context" for r in group)}
           for module, group in _group(questions(rows), _module_key).items()]
    return sorted(out, key=lambda d: d["module"])


def by_topic(rows: list[Row], titles: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Темы: вопросов, без ответа, «трудно» (difficulty), 👎. titles — slug темы → понятное название."""
    out = []
    for topic, group in _group(questions(rows), lambda r: r.topic or NO_TOPIC).items():
        modules = Counter(r.module for r in group if r.module is not None)
        out.append({"topic": topic, "title": (titles or {}).get(topic, ""),
                    "module": modules.most_common(1)[0][0] if modules else None, "questions": len(group),
                    "no_context": sum(r.refusal == "no_context" for r in group),
                    "difficulty": sum(r.intent == "difficulty" for r in group), "negative": _negative(group)})
    return sorted(out, key=lambda d: (-d["questions"], d["topic"]))


OUTCOMES = ("Ответ по материалам", "Нет в материалах", "Отказ по правилам", "Прочее")


def outcome(r: Row) -> str:
    """Один исход на вопрос — для графика по дням с накоплением (категории не пересекаются)."""
    if r.refusal == "no_context":
        return OUTCOMES[1]
    if r.refusal in RESTRICTION_REFUSALS:
        return OUTCOMES[2]
    if r.answered_by_materials:
        return OUTCOMES[0]
    return OUTCOMES[3]


def by_day_outcome(rows: list[Row]) -> list[dict[str, Any]]:
    """Длинная таблица день × исход; нули включены, чтобы слои шли в одном порядке каждый день."""
    q = questions(rows)
    days = sorted({r.day for r in q})
    counts = Counter((r.day, outcome(r)) for r in q)
    return [{"day": d, "outcome": o, "questions": counts.get((d, o), 0)} for d in days for o in OUTCOMES]


def by_day(rows: list[Row]) -> list[dict[str, Any]]:
    out = []
    for day, group in _group(questions(rows), lambda r: r.day).items():
        out.append({"day": day, "questions": len(group), "answered": sum(r.answered_by_materials for r in group),
                    "no_context": sum(r.refusal == "no_context" for r in group),
                    "refused": sum(r.refusal in RESTRICTION_REFUSALS for r in group),
                    "cost_usd": sum(r.cost_usd for r in group)})
    return sorted(out, key=lambda d: d["day"])


def content_gaps(rows: list[Row], show_live: bool = False, examples: int = 3) -> list[dict[str, Any]]:
    """Вопросы без ответа по темам и модулям — что дописать в базу знаний."""
    gaps = [r for r in questions(rows) if r.refusal == "no_context"]
    out = []
    for (topic, module), group in _group(gaps, lambda r: (r.topic or NO_TOPIC, _module_key(r))).items():
        texts = list(dict.fromkeys(r.question for r in group if r.text_visible(show_live)))[:examples]
        hidden = sum(not r.text_visible(show_live) for r in group)
        if hidden:
            texts.append(f"(ещё {hidden} — тексты живых посетителей скрыты)")
        out.append({"topic": topic, "module": module, "questions": len(group), "examples": texts})
    return sorted(out, key=lambda d: (-d["questions"], d["topic"]))


def hard_lectures(rows: list[Row], titles: dict[str, str] | None = None, top: int = 10) -> list[dict[str, Any]]:
    """Лекции, на которые чаще всего ссылались ответы на «мне трудно» (difficulty)."""
    counts: Counter[str] = Counter()
    negative: Counter[str] = Counter()
    for r in questions(rows):
        if r.intent != "difficulty":
            continue
        for lecture in set(r.lectures):
            counts[lecture] += 1
            if r.rating is not None and r.rating < 0:
                negative[lecture] += 1
    out = [{"lecture": lec, "title": (titles or {}).get(lec, ""), "questions": n, "negative": negative[lec]}
           for lec, n in counts.items()]
    return sorted(out, key=lambda d: (-d["questions"], d["lecture"]))[:top]


def refusals(rows: list[Row]) -> list[dict[str, Any]]:
    counts = Counter(r.refusal for r in rows if r.refusal)
    return [{"refusal": k, "label": REFUSAL_LABELS.get(k, k), "count": v}
            for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def ratings(rows: list[Row]) -> list[dict[str, Any]]:
    out = []
    for intent, group in _group([r for r in questions(rows) if r.rating is not None], lambda r: r.intent).items():
        up = sum(1 for r in group if (r.rating or 0) > 0)
        down = len(group) - up
        out.append({"intent": intent, "label": INTENT_LABELS.get(intent, intent), "up": up, "down": down,
                    "positive_share": up / len(group)})
    return sorted(out, key=lambda d: d["intent"])


def latency(rows: list[Row]) -> dict[str, float | None]:
    values = [float(r.latency_ms) for r in questions(rows) if r.latency_ms > 0]
    return {"p50": quantile(values, 0.5), "p90": quantile(values, 0.9), "p95": quantile(values, 0.95),
            "max": max(values) if values else None}


def latencies(rows: list[Row]) -> list[float]:
    return [r.latency_ms / 1000 for r in questions(rows) if r.latency_ms > 0]


def questions_table(rows: list[Row], show_live: bool, limit: int = 50) -> list[dict[str, Any]]:
    """Последние вопросы с текстом — только те, что можно показывать."""
    visible = sorted((r for r in rows if r.text_visible(show_live)), key=lambda r: r.ts, reverse=True)[:limit]
    return [{"время (МСК)": r.ts.astimezone(MSK).strftime("%d.%m %H:%M"), "источник": r.source,
             "сценарий": INTENT_LABELS.get(r.intent, r.intent), "модуль": r.module, "тема": r.topic,
             "отказ": REFUSAL_LABELS.get(r.refusal, r.refusal), "оценка": r.rating, "вопрос": r.question}
            for r in visible]
