"""Few-shot: пары «хороший / плохой ответ» из data/fewshot/examples.yaml для промпта.

У каждой пары — сценарий (intent или scenario), вопрос, уровень, плохой ответ, чем он плох, хороший ответ
и метки материалов (sources), на которые хороший ответ ссылается номерами [1], [2].
В промпт идут 2 пары того же сценария (сначала того же уровня) и одна пара no_answer: она учит модель
говорить «в материалах этого нет», когда материалы отвечают лишь на часть вопроса.

Пара может быть помечена условиями (tags, например `revision` — у студента задание на доработке): такая пара
идёт в промпт только при этих условиях и тогда — первой в своём сценарии. Без условий выбор прежний.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCENARIOS = ("deadline", "difficulty", "recommendation", "organizational", "grading_request", "schedule_change",
             "off_topic", "no_answer")
# Для сценария material отдельных пар может не быть: ближе всего к нему difficulty.
_FALLBACK = {"material": ("material", "difficulty")}
FIELDS = ("question", "bad_answer", "why_bad", "good_answer")
MIN_EXAMPLES = 16


@dataclass(frozen=True)
class Example:
    scenario: str
    question: str
    level: str
    bad_answer: str
    why_bad: str
    good_answer: str
    sources: tuple[str, ...] = ()  # метки материалов, на которые ссылается good_answer через [1], [2]
    tags: tuple[str, ...] = ()  # условия, при которых пара уместна (revision — у студента есть доработка)

    def sources_text(self) -> str:
        return "; ".join(f"[{n}] {s}" for n, s in enumerate(self.sources, 1)) or "—"


def _items(data: Any) -> list[Any]:
    if isinstance(data, dict):
        for key in ("examples", "pairs", "items"):
            if isinstance(data.get(key), list):
                return list(data[key])
        return []
    return list(data) if isinstance(data, list) else []


def parse_examples(data: Any) -> tuple[list[Example], list[str]]:
    examples: list[Example] = []
    problems: list[str] = []
    for n, item in enumerate(_items(data), 1):
        if not isinstance(item, dict):
            problems.append(f"пример {n}: не словарь")
            continue
        scenario = str(item.get("scenario") or item.get("intent") or "").strip()
        missing = [f for f in FIELDS if not str(item.get(f) or "").strip()]
        if not scenario:
            missing.insert(0, "scenario")
        if missing:
            problems.append(f"пример {n}: нет полей {', '.join(missing)}")
            continue
        raw_sources = item.get("sources") or []
        sources = tuple(str(s).strip() for s in raw_sources) if isinstance(raw_sources, list) else ()
        raw_tags = item.get("tags") or []
        tags = tuple(str(t).strip() for t in raw_tags if str(t).strip()) if isinstance(raw_tags, list) else ()
        examples.append(Example(scenario, str(item["question"]).strip(), str(item.get("level") or "all").strip(),
                                str(item["bad_answer"]).strip(), str(item["why_bad"]).strip(),
                                str(item["good_answer"]).strip(), sources, tags))
    return examples, problems


def load_examples(path: Path) -> tuple[list[Example], list[str]]:
    if not path.is_file():
        return [], [f"нет файла {path}"]
    return parse_examples(yaml.safe_load(path.read_text(encoding="utf-8")))


def check_examples(examples: list[Example]) -> list[str]:
    """Требования BRIEF: минимум 16 пар, каждый из 8 сценариев представлен."""
    problems = []
    if len(examples) < MIN_EXAMPLES:
        problems.append(f"пар {len(examples)}, нужно не меньше {MIN_EXAMPLES}")
    have = {e.scenario for e in examples}
    absent = [s for s in SCENARIOS if s not in have]
    if absent:
        problems.append(f"нет сценариев: {', '.join(absent)}")
    return problems


class FewShotLibrary:
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def _pick(self, scenarios: tuple[str, ...], level: str, k: int, tags: frozenset[str]) -> list[Example]:
        pool = [e for e in self.examples if e.scenario in scenarios and set(e.tags) <= tags]
        pool.sort(key=lambda e: (scenarios.index(e.scenario), not e.tags, e.level not in (level, "all")))
        return pool[:k]

    def select(self, intent: str, level: str, k: int = 2, with_no_answer: bool = True,
               tags: frozenset[str] | set[str] = frozenset()) -> list[Example]:
        """tags — условия ответа (например, {"revision"}): пары с этими условиями идут первыми, пары с другими
        условиями не идут вовсе."""
        wanted = frozenset(tags)
        chosen = self._pick(_FALLBACK.get(intent, (intent,)), level, k, wanted)
        if with_no_answer and intent != "no_answer":
            chosen += self._pick(("no_answer",), level, 1, wanted)
        return chosen
