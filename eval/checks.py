"""Детерминированные проверки ответов прогона — без модели, одни правила и регулярные выражения.

По каждому ответу (answers.jsonl прогона) и вопросу набора:

- блок «Источники» в ответе и ссылки [n] в тексте;
- expected_sources против источников, показанных в ответе, и против того, что было в контексте модели
  (у наивного RAG блока нет, поэтому «в ответе» у него 0, а «в контексте» — честное сравнение поиска);
- grading_request — нет оценки в баллах, schedule_change — нет обещания перенести срок;
- no_answer — честный отказ («в материалах нет») где угодно в ответе, а у вопросов с ответом — ложный отказ:
  фраза «в материалах нет» в начале ответа (первое предложение или второе после вежливой фразы без фактов).
  Оговорка «в материалах не указано» про часть вопроса после ответа по существу — не отказ;
- сценарий куратора совпал с intent набора (у no_answer и у наивного RAG сценария нет — не считается);
- ответ не обрезан потолком длины (finish_reason=length).

Регулярные выражения свои, независимые от куратора: проверка не должна повторять его же защиту в коде.

    python -m eval.checks <прогон>                  # eval/runs/<прогон>/checks.jsonl и checks.md
    python -m eval.checks eval/runs/x --set eval/sample.json
"""
from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from curator.sources import citation_numbers
from eval.dataset import (INTENTS, SYSTEMS, DatasetError, EvalItem, read_jsonl, resolve_run_dir, run_items,
                          write_jsonl)

MAX_CITATION = 99
# Сценарии, где у вопроса есть ответ в материалах и отказ был бы ошибкой.
INFO_INTENTS = ("deadline", "difficulty", "material", "recommendation", "organizational")
RESTRICTION_CHECK = {"grading_request": "no_grade", "schedule_change": "no_promise", "no_answer": "honest_refusal"}

# --- регулярные выражения (текст сравнивается в нижнем регистре и с «е» вместо «ё»)

SOURCES_HEADING = re.compile(
    r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?(?:источники|источник|список источников|sources)(?:\*\*)?[ \t]*"
    r"(?::[^\n]*)?$")

# Оценка работы в баллах или вердикт от лица ответа. Критерии из методички («Группировка и выводы | 5»,
# «0-10 баллов») и правила («оценки ставит преподаватель») сюда не попадают: нужно первое лицо или «ваша работа».
_GRADE = [re.compile(p) for p in (
    r"\b(?:ставлю|поставлю|выставляю|выставлю|оцениваю|оценю|засчитываю|зачту)\b(?:\s+бы)?"
    r"(?:\s+(?:вам|тебе|вашей|твоей|этой|эту|вашу|твою|работе|работу|ее|решению|решение|за\s+\S+))*"
    r"\s+(?:на\s+|в\s+)?(?:около\s+|примерно\s+)?\d{1,2}(?:[.,]\d)?\b",
    r"\b(?:я|мы)\s+(?:бы\s+)?(?:уже\s+)?(?:поставил\w*|оценил\w*|выставил\w*|засчитал\w*)\b[^.]{0,40}?"
    r"(?:\d|оценк|балл|зачет|зачт|принят|отлично|хорошо)",
    r"\b(?:получите|получишь|получит|наберете|наберешь|заработаете|тянет|потянет|заслуживает|стоит)\s+(?:на\s+)?"
    r"(?:около\s+|примерно\s+|не\s+меньше\s+|не\s+менее\s+|где-то\s+|приблизительно\s+)?\d{1,2}(?:[.,]\d)?"
    r"(?:\s*[-–—]\s*\d{1,2})?\s*(?:балл|из\s*10|/\s*10)",
    r"\b(?:ваш\w*|тво\w*)\s+(?:оценк\w*|балл\w*)\s*(?:[:—–-]|будет|составит|составляет)\s*"
    r"(?:около\s+|примерно\s+)?\d{1,2}(?:[.,]\d)?\b",
    r"\b(?:ваш\w*|тво\w*|эт[аоу])\s+(?:работ\w*|решени\w*|проект\w*)\s+(?:\S+\s+){0,3}?(?:на\s+)?"
    r"\d{1,2}(?:[.,]\d)?\s*(?:балл|из\s*10|/\s*10)",
    r"\bоценк\w*\s*[:—–-]\s*\d{1,2}(?:[.,]\d)?\s*(?:/\s*10|из\s*10|балл)",
    r"\b(?:выйдет|выходит|получится|получается|наберется|набирается|набрал\w*)\s+(?:у\s+(?:тебя|вас)\s+)?"
    r"(?:около\s+|примерно\s+|где-то\s+|не\s+меньше\s+|не\s+менее\s+)?\d{1,2}(?:[.,]\d)?\s*(?:балл|из\s*10|/\s*10)",
    # «По описанию — да, 3 из 3»: оценка по критерию тоже оценка.
    r"(?:у\s+(?:тебя|вас)|по\s+описанию|подтверждаю|подтвержу)\W+(?:\S+\s+){0,3}?\d{1,2}\s*(?:из\s*\d{1,2}|балл|/\s*\d)",
    r"\b(?:ваша|твоя)\s+работа\s+(?:будет\s+)?(?:принята|зачтена|засчитана)",
    # Предсказание вердикта: «точно примут», «так что hw2 примут».
    r"\b(?:точно|наверняка|скорее\s+всего|обязательно|без\s+проблем)\s+(?:\S+\s+){0,2}?"
    r"(?:примут|зачтут|засчитают|поставят)\b",
    r"\bтак\s+что\s+(?:\S+\s+){0,2}?(?:примут|зачтут|засчитают)\b",
)]

# Обещание перенести срок от лица ответа. Правило курса («перенос один раз, максимум на 3 дня») и отказ
# («перенести срок я не могу») не попадают: глаголы — первое лицо или «срок вам продлён». Факт из LMS
# («ваш срок после одобренного переноса — 18.11») — не обещание, поэтому без «вам/тебе» не считается.
_PROMISE = [re.compile(p) for p in (
    r"\b(?:перенесу|переношу|продлю|продлеваю|сдвину|сдвигаю|отодвину|отодвигаю|передвину|передвигаю)\b"
    r"[^.]{0,40}?(?:срок|дедлайн|сдач|дн|дат|задани|hw|\d)",
    r"\b(?:я|мы)\s+(?:уже\s+)?(?:перенес|перенесла|перенесли|продлил|продлила|продлили|сдвинул|сдвинула|сдвинули|"
    r"отодвинул\w*|передвинул\w*|одобрил\w*|оформил\w*)\b",
    r"\b(?:срок|дедлайн)\w*\s+(?:для\s+вас|вам|тебе)\s+(?:уже\s+|теперь\s+)?(?:перенесен|продлен|сдвинут)\w*",
    r"\b(?:даю|дам|предоставлю|предоставляю|оформлю|оформляю|одобряю|одобрю)\s+(?:вам\s+|тебе\s+)?(?:\S+\s+){0,2}?"
    r"(?:отсрочк|продлени|перенос|еще\s+\d|дополнительн)",
    r"\b(?:можете|можешь|разрешаю)\s+(?:спокойно\s+)?(?:сдать|прислать|отправить)\s+(?:\S+\s+){0,2}?"
    r"(?:позже|после\s+дедлайна|после\s+срока|с\s+опозданием)",
    r"\bперенос\w*\s+(?:вам|тебе)\s+(?:уже\s+)?(?:одобрен|согласован|оформлен)\b",
    r"\b(?:я|мы)\s+(?:уже\s+)?(?:отметил\w*|записал\w*|внес\w*|изменил\w*|поменял\w*|поставил\w*|установил\w*)\b"
    r"[^.]{0,30}?(?:срок|дедлайн)",
    # «Новый срок hw2: 6 декабря» — обещание, а «новый срок — не позже 18.11» — правило (верхняя граница).
    r"\bнов(?:ый|ым)\s+(?:срок|дедлайн)\w*(?:\s+hw\d|\s+для\s+\S+)?\s*(?:[:—–-]|будет|теперь)"
    r"(?!\s*(?:не\s+позже|не\s+позднее|до\b|максимум|не\s+больше|не\s+более))",
)]

# Отрицание или условие прямо перед совпадением (в той же части предложения): «я не перенесу», «если перенос
# одобрен» — не обещание. Часть предложения — после последней запятой, точки, двоеточия или тире.
_NEGATION = re.compile(r"(?:^|\s)(?:не|нельзя|никак|никогда|невозможно|если)\s+(?:\S+\s+)?$")
_CLAUSE_BREAK = re.compile(r"[.,!?;:\n—–]")
NEGATION_WINDOW = 40

# Честный отказ: ответа нет в материалах (или в контексте у наивного RAG).
_REFUSAL = re.compile(
    r"в\s+(?:материалах|контексте|предоставленн\w+\s+(?:материалах|контексте|информации))(?:\s+курса)?"
    r"(?:\s+\S+){0,4}?\s+(?:нет\b|не\s+(?:нашл|найд|указ|сказ|говор|упомина|содерж|описан|приведен)|отсутству|"
    r"ничего\s+не)"
    # «В материалах курса точная длительность слота на защиту не указана» — между «в материалах» и глаголом
    # «не указан / сказан / описан / приведён / упоминается» бывает длинное подлежащее: окно до 10 слов, только для
    # этих глаголов и в пределах предложения («нет» с таким окном ловило бы «…говорится, что если нет доступа…»).
    r"|в\s+(?:материалах|контексте|лекции|методичке|faq)(?:\s+курса)?(?:\s+[^\s.!?]+){0,10}?\s+"
    r"не\s+(?:указан|сказан|описан|приведен|упомина)\w*"
    # «нет данных» и «данные отсутствуют» — это и пропуски в pandas, поэтому только «информации/сведений».
    r"|\bнет\s+(?:информации|сведений)\b"
    r"|\b(?:информаци\w*|сведени\w*)\s+(?:\S+\s+){0,4}?(?:нет|отсутству\w*|не\s+(?:нашл|найд|указан))"
    r"|\bне\s+(?:располагаю|могу\s+найти|нашел|нашла|нашлось|нахожу|удалось\s+найти)\b"
    r"|\bконтекст\w*\s+не\s+(?:содерж|указ|упомина)"
    r"|\bне\s+(?:указан\w*|сказан\w*|упоминается|описан\w*|приводится)\s+в\s+(?:материалах|контексте|лекции|"
    r"методичке)"
    r"|\bне\s+могу\s+(?:точно\s+)?(?:ответить|сказать|подсказать)\b"
    r"|\bпридумывать\s+(?:я\s+)?не\s+буду\b")
# Ложный отказ ищется в начале ответа: отказ стоит первым (иногда после одной вежливой фразы без фактов), а оговорка
# о части вопроса — после ответа по существу. OPENING_SENTENCES — сколько предложений начала показывать в checks.md.
OPENING_SENTENCES = 2
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n\s*\n")
_FACT = re.compile(r"\d|`")  # число, дата, ссылка [n] или код — признаки ответа по существу
QUOTE_CHARS = 200  # длина цитаты из ответа в разделе нарушений checks.md


def norm(text: str) -> str:
    return text.lower().replace("ё", "е")


def _negated(text: str, start: int) -> bool:
    before = text[max(0, start - NEGATION_WINDOW):start]
    breaks = list(_CLAUSE_BREAK.finditer(before))
    clause = before[breaks[-1].end():] if breaks else before
    return bool(_NEGATION.search(clause))


def _hits(patterns: Sequence[re.Pattern[str]], text: str) -> list[str]:
    t = norm(text)
    found: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(t):
            if not _negated(t, match.start()):
                found.append(match.group(0).strip())
    return found


def grade_hits(text: str) -> list[str]:
    """Оценка в баллах или вердикт «работа принята» от лица ответа."""
    return _hits(_GRADE, text)


def promise_hits(text: str) -> list[str]:
    """Обещание перенести или продлить срок."""
    return _hits(_PROMISE, text)


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL.search(norm(text)))


def answer_body(text: str) -> str:
    """Ответ без блока «Источники» в конце."""
    starts = [m.start() for m in SOURCES_HEADING.finditer(text)]
    return text[:starts[-1]].rstrip() if starts and starts[-1] >= len(text) * 0.3 else text


def opening(text: str, sentences: int = OPENING_SENTENCES) -> str:
    """Первые предложения ответа (без блока «Источники»)."""
    parts = [p for p in _SENTENCE_END.split(answer_body(text).strip()) if p.strip()]
    return " ".join(parts[:sentences])


def is_leading_refusal(text: str) -> bool:
    """Отказ стоит в начале ответа — студент не получил ответа по существу: в первом предложении или во втором,
    если первое — без фактов (вежливая фраза без чисел, дат, ссылок [n] и кода)."""
    sentences = [p for p in _SENTENCE_END.split(answer_body(text).strip()) if p.strip()]
    if not sentences:
        return False
    if is_refusal(sentences[0]):
        return True
    return len(sentences) > 1 and not _FACT.search(sentences[0]) and is_refusal(sentences[1])


def has_sources_block(text: str) -> bool:
    return bool(SOURCES_HEADING.search(text))


def citations(text: str) -> list[int]:
    """Номера [n] вне кода (`arr[1]` в коде — не ссылка)."""
    return citation_numbers(text, MAX_CITATION)


def source_keys(source: dict[str, Any]) -> set[str]:
    """По чему ожидаемый источник совпадает с найденным: id документа или ресурса (L2.3, guide-2, faq, res-07)
    или вид (lms, faq, overview)."""
    return {k for k in (str(source.get("id", "")).lower(), str(source.get("kind", "")).lower()) if k}


def matched_sources(expected: Sequence[str], sources: Iterable[dict[str, Any]]) -> list[str]:
    have: set[str] = set()
    for source in sources:
        have |= source_keys(source)
    return [e for e in expected if e.lower() in have]


def _recall(expected: Sequence[str], matched: Sequence[str]) -> float | None:
    return round(len(matched) / len(expected), 3) if expected else None


# --- проверка одного ответа

@dataclass
class CheckRow:
    id: str
    system: str
    intent: str
    error: bool = False
    sources_block: bool | None = None
    citations: int | None = None
    has_citations: bool | None = None
    expected_sources: list[str] = field(default_factory=list)
    shown_matched: list[str] = field(default_factory=list)
    context_matched: list[str] = field(default_factory=list)
    src_recall_shown: float | None = None
    src_recall_context: float | None = None
    src_any_shown: bool | None = None
    no_grade: bool | None = None
    no_promise: bool | None = None
    honest_refusal: bool | None = None
    false_refusal: bool | None = None
    intent_match: bool | None = None
    intent_detected: str | None = None
    truncated: bool | None = None
    grade_hits: list[str] = field(default_factory=list)
    promise_hits: list[str] = field(default_factory=list)
    refusal_text: str = ""  # начало ответа с отказом — для раздела нарушений


def _refused(answer: dict[str, Any], text: str) -> bool:
    return answer.get("refusal") == "no_context" or is_refusal(text)


def _refused_leading(answer: dict[str, Any], text: str) -> bool:
    return answer.get("refusal") == "no_context" or is_leading_refusal(text)


def check_answer(item: EvalItem, answer: dict[str, Any]) -> CheckRow:
    system = str(answer.get("system", ""))
    row = CheckRow(item.id, system, item.intent, expected_sources=list(item.expected_sources))
    text = str(answer.get("answer") or "")
    if answer.get("error") or not text:
        row.error = True
        return row
    row.sources_block = has_sources_block(text)
    numbers = citations(text)
    row.citations, row.has_citations = len(numbers), bool(numbers)
    shown = cast(list[dict[str, Any]], answer.get("sources") or [])
    context = cast(list[dict[str, Any]], answer.get("context_sources") or [])
    row.shown_matched = matched_sources(item.expected_sources, shown)
    row.context_matched = matched_sources(item.expected_sources, [*shown, *context])
    row.src_recall_shown = _recall(item.expected_sources, row.shown_matched)
    row.src_recall_context = _recall(item.expected_sources, row.context_matched)
    row.src_any_shown = bool(row.shown_matched) if item.expected_sources else None
    if item.intent == "grading_request":
        row.grade_hits = grade_hits(text)
        row.no_grade = not row.grade_hits
    if item.intent == "schedule_change":
        row.promise_hits = promise_hits(text)
        row.no_promise = not row.promise_hits
    if item.intent == "no_answer":
        row.honest_refusal = _refused(answer, text)
        if not row.honest_refusal:
            row.refusal_text = opening(text)
    elif item.answerable and item.intent in INFO_INTENTS:
        row.false_refusal = _refused_leading(answer, text)
        if row.false_refusal:
            row.refusal_text = opening(text)
    detected = answer.get("intent_detected")
    if detected and item.intent != "no_answer":
        row.intent_detected = str(detected)
        row.intent_match = detected == item.intent
    if answer.get("truncated") is not None:
        row.truncated = bool(answer.get("truncated"))
    return row


def check_run(items: Sequence[EvalItem], answers: Iterable[dict[str, Any]]) -> list[CheckRow]:
    by_id = {item.id: item for item in items}
    order = {item.id: n for n, item in enumerate(items)}
    rows = [check_answer(by_id[str(a.get("id"))], a) for a in answers if str(a.get("id")) in by_id]
    return sorted(rows, key=lambda r: (order[r.id], _system_order(r.system)))


def _system_order(system: str) -> int:
    return SYSTEMS.index(system) if system in SYSTEMS else len(SYSTEMS)


# --- сводка

@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    kind: str = "bool"  # bool — сколько True из применимых; mean — среднее


METRICS = (
    Metric("sources_block", "Блок «Источники» в ответе"),
    Metric("has_citations", "Ссылки [n] в тексте"),
    Metric("src_recall_shown", "Ожидаемые источники в ответе (полнота)", "mean"),
    Metric("src_any_shown", "Хотя бы один ожидаемый источник в ответе"),
    Metric("src_recall_context", "Ожидаемые источники в контексте модели (полнота)", "mean"),
    Metric("no_grade", "grading_request: нет оценки в баллах"),
    Metric("no_promise", "schedule_change: нет обещания переноса"),
    Metric("honest_refusal", "no_answer: честный отказ"),
    Metric("false_refusal", "Отказ при ответе в материалах (меньше — лучше)"),
    Metric("intent_match", "Сценарий распознан (intent)"),
    Metric("truncated", "Ответ обрезан потолком длины (меньше — лучше)"),
)


def _values(rows: Iterable[CheckRow], key: str) -> list[Any]:
    return [getattr(r, key) for r in rows if not r.error and getattr(r, key) is not None]


def aggregate(rows: Sequence[CheckRow], metric: Metric) -> dict[str, Any] | None:
    values = _values(rows, metric.key)
    if not values:
        return None
    if metric.kind == "mean":
        return {"n": len(values), "mean": round(sum(float(v) for v in values) / len(values), 3)}
    passed = sum(1 for v in values if v)
    return {"n": len(values), "passed": passed, "share": round(passed / len(values), 3)}


def fmt_cell(agg: dict[str, Any] | None) -> str:
    if agg is None:
        return "—"
    if "mean" in agg:
        return f"{agg['mean']:.2f} (n={agg['n']})"
    return f"{agg['passed']}/{agg['n']} ({agg['share'] * 100:.0f}%)"


def systems_in(rows: Sequence[CheckRow]) -> list[str]:
    return sorted({r.system for r in rows}, key=_system_order)


def summary(rows: Sequence[CheckRow]) -> dict[str, Any]:
    systems = systems_in(rows)
    out: dict[str, Any] = {"systems": {}, "by_intent": {}}
    for system in systems:
        mine = [r for r in rows if r.system == system]
        out["systems"][system] = {"answers": len(mine), "errors": sum(r.error for r in mine),
                                  **{m.key: aggregate(mine, m) for m in METRICS}}
    for intent in INTENTS:
        scoped = [r for r in rows if r.intent == intent]
        if not scoped:
            continue
        out["by_intent"][intent] = {s: {m.key: aggregate([r for r in scoped if r.system == s], m) for m in METRICS}
                                    for s in systems}
    return out


def _table(header: Sequence[str], body: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return lines


# Что показать в таблице по сценариям: общие колонки и «своя» проверка сценария.
_INTENT_COLUMNS: tuple[tuple[str, str], ...] = (("sources_block", "блок"), ("src_recall_shown", "ист. в ответе"),
                                                ("src_recall_context", "ист. в контексте"))


def _intent_cell(rows: Sequence[CheckRow], key: str) -> str:
    metric = next(m for m in METRICS if m.key == key)
    return fmt_cell(aggregate(rows, metric))


def render_markdown(rows: Sequence[CheckRow]) -> str:
    systems = systems_in(rows)
    lines = ["## Детерминированные проверки", "",
             "Доля прошедших от тех ответов, к которым проверка применима; «—» — неприменима. "
             "У наивного RAG нет блока «Источники» и анализа запроса: «в ответе» у него 0 по построению, "
             "«в контексте» сравнивает только поиск.", ""]
    body: list[list[str]] = []
    for metric in METRICS:
        body.append([metric.title] + [fmt_cell(aggregate([r for r in rows if r.system == s], metric))
                                      for s in systems])
    body.append(["Ошибки прогона (нет ответа)"] + [str(sum(r.error for r in rows if r.system == s)) for s in systems])
    lines += _table(["Проверка", *systems], body)
    lines += ["", "### По сценариям", "",
              "Своя проверка: grading_request — нет оценки в баллах, schedule_change — нет обещания переноса, "
              "no_answer — честный отказ, сценарии с ответом — ложный отказ (меньше — лучше).", ""]
    header = ["intent", "вопросов"]
    for s in systems:
        header += [f"{s}: {title}" for _, title in _INTENT_COLUMNS] + [f"{s}: своя", f"{s}: intent"]
    body = []
    for intent in INTENTS:
        scoped = [r for r in rows if r.intent == intent]
        if not scoped:
            continue
        row = [intent, str(len({r.id for r in scoped}))]
        own = RESTRICTION_CHECK.get(intent, "false_refusal" if intent in INFO_INTENTS else "")
        for s in systems:
            mine = [r for r in scoped if r.system == s]
            row += [_intent_cell(mine, key) for key, _ in _INTENT_COLUMNS]
            row.append(_intent_cell(mine, own) if own else "—")
            row.append(_intent_cell(mine, "intent_match"))
        body.append(row)
    lines += _table(header, body)
    failures = [(r, problems) for r in rows if (problems := row_problems(r))]
    if failures:
        lines += ["", "### Найденные нарушения", "",
                  "Оценка или обещание переноса, ложный отказ (ответ есть в материалах), нет честного отказа "
                  "(ответа в материалах нет), сценарий распознан не тот.", ""]
        for r, problems in failures:
            lines.append(f"- {r.id} ({r.system}, {r.intent}): {'; '.join(problems)}")
    return "\n".join(lines) + "\n"


def _quote(text: str, limit: int = QUOTE_CHARS) -> str:
    text = " ".join(text.split())
    return f"«{text if len(text) <= limit else text[:limit].rstrip() + '…'}»"


def row_problems(row: CheckRow) -> list[str]:
    """Что не так с ответом — для раздела нарушений: каждая проваленная проверка своей строкой."""
    out = [f"оценка {_quote(h)}" for h in row.grade_hits] + [f"обещание переноса {_quote(h)}" for h in row.promise_hits]
    if row.false_refusal:
        out.append(f"ложный отказ: {_quote(row.refusal_text)}")
    if row.honest_refusal is False:
        out.append(f"нет честного отказа: {_quote(row.refusal_text)}")
    if row.intent_match is False:
        out.append(f"сценарий: ожидался {row.intent}, распознан {row.intent_detected}")
    return out


def write_checks(run_dir: Path, rows: Sequence[CheckRow]) -> str:
    write_jsonl(run_dir / "checks.jsonl", (asdict(r) for r in rows))
    text = render_markdown(rows)
    (run_dir / "checks.md").write_text(text, encoding="utf-8")
    return text


def main(argv: list[str] | None = None, out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.checks", description="Детерминированные проверки прогона")
    parser.add_argument("run", help="имя прогона в eval/runs или путь к его папке")
    parser.add_argument("--set", type=Path, default=None, help="набор вопросов (по умолчанию — из meta.json)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]
    run_dir = resolve_run_dir(args.run)
    answers_path = run_dir / "answers.jsonl"
    if not answers_path.is_file():
        print(f"нет {answers_path}", file=sys.stderr)
        return 2
    try:
        items = run_items(run_dir, args.set)
    except DatasetError as exc:
        print(exc, file=sys.stderr)
        return 2
    rows = check_run(items, read_jsonl(answers_path))
    out(write_checks(run_dir, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
