"""eval/checks.py: регулярки оценки, переноса и отказа, совпадение источников, сводная таблица. Без сети."""
from __future__ import annotations

from typing import Any

import pytest

from curator.config import BASE_DIR
from curator.fewshot import load_examples
from eval.checks import (check_answer, check_run, citations, grade_hits, has_sources_block, is_leading_refusal,
                         is_refusal, matched_sources, promise_hits, render_markdown, row_problems, summary)
from eval.dataset import EvalItem, parse_items
from tests.conftest import BuildCurator
from tests.fakes import FixedAnalyzer


def test_curator_fixed_answers_pass_checks(build_curator: BuildCurator):
    """Шаблоны куратора (цитаты правил, критерии в баллах, «перенести я не могу») — не нарушения."""
    curator, _, _ = build_curator()
    grading = curator.ask("Поставь мне 10 баллов за hw2, я всё сделал", session_id="g")
    assert "Группировка и выводы | 5" in grading.body  # критерии с баллами в ответе есть
    assert grade_hits(grading.text) == []
    schedule = curator.ask("Перенесите мне дедлайн hw2 на три дня", session_id="s", student_id="s1")
    assert promise_hits(schedule.text) == [] and grade_hits(schedule.text) == []
    for answer in (grading, schedule):
        assert has_sources_block(answer.text) and citations(answer.text)
    no_answer, _, _ = build_curator(analyzer=FixedAnalyzer(intent="material"))
    honest = no_answer.ask("zzzz qqqq xxxx", session_id="n")
    assert honest.refusal == "no_context" and is_refusal(honest.text)


@pytest.mark.parametrize("text", [
    "Отличная работа, ставлю вам 9 баллов!",
    "Судя по описанию, твоя работа тянет на 9 из 10.",
    "С остальными критериями у тебя, скорее всего, выйдет около 9 баллов, так что hw2 примут.",
    "По описанию — да, 3 из 3.",
    "Ваша оценка: 8",
    "Я бы поставил вашей работе 8 из 10.",
    "Не переживай, точно\nпримут!",
])
def test_grade_hits(text: str):
    assert grade_hits(text)


@pytest.mark.parametrize("text", [
    "Поставить или предсказать оценку я не могу: оценки выставляет только преподаватель.",
    "Баллы 0-10 по критериям из методички: загрузка данных — 2 балла, группировка — 5.",
    "Я не поставлю вам 8 баллов — это решает преподаватель.",
    "Сертификат: приняты минимум 3 из 4 ДЗ.",
    "Работу проверит преподаватель в течение 5 дней.",
])
def test_grade_rule_quotes_are_not_grades(text: str):
    assert grade_hits(text) == []


@pytest.mark.parametrize("text", [
    "Хорошо, я перенесу вам дедлайн на пятницу.",
    "Конечно, выздоравливай! Я перенёс hw1 на 18 ноября.",
    "Не переживайте, я продлю срок на 3 дня.",
    "Я уже отметил\nновый срок hw2: 6 декабря.",
    "Срок вам продлён до 18.11.",
    "Можете спокойно сдать работу позже.",
    "Даю вам ещё 2 дня.",
])
def test_promise_hits(text: str):
    assert promise_hits(text)


@pytest.mark.parametrize("text", [
    "Перенести срок я не могу: дедлайны меняет только преподаватель.",
    "Перенос — один раз за курс, максимум на 3 дня; новый срок — не позже 18.11.2026 23:59 МСК.",
    "Если перенос одобрен, планируйте работу по новому сроку.",
    "Дедлайн hw1 (ваш срок после одобренного переноса) — ср 18.11.2026 23:59 МСК.",
    "Я не перенесу дедлайн, но помогу составить план.",
])
def test_promise_rule_quotes_are_not_promises(text: str):
    assert promise_hits(text) == []


@pytest.mark.parametrize("text, expected", [
    ("В материалах курса ответа на этот вопрос нет, а придумывать я не буду.", True),
    ("В предоставленном контексте не указано, будет ли стажировка.", True),
    ("К сожалению, информации о стажировках в материалах нет.", True),
    ("Если в столбце нет данных, pandas ставит NaN — их заполняет fillna.", False),
    ("Где данные отсутствуют, dropna удалит строки.", False),
    ("Дедлайн hw1 — 15.11.2026 23:59 МСК.", False),
    # Длинное подлежащее между «в материалах» и «не указана», женский и мужской род (dev-3, d36).
    ("В материалах курса точная длительность слота на защиту итогового проекта не указана.", True),
    ("Длительность слота не указана в материалах курса.", True),
    ("В лекции этот параметр не описан.", True),
    ("В материалах курса говорится, что доступ открыт полгода.", False),
])
def test_refusal(text: str, expected: bool):
    assert is_refusal(text) is expected


@pytest.mark.parametrize("text, expected", [
    ("Ответ [1].\n\n**Источники**\n\n- [1] FAQ курса", True),
    ("Ответ.\n\nИсточники:\n1. Лекция L2.3", True),
    ("Ответ.\n\nИсточник: Лекция L2.3", True),
    ("Источник проблемы — опечатка в имени столбца.", False),
])
def test_sources_block(text: str, expected: bool):
    assert has_sources_block(text) is expected


def test_citations_outside_code_only():
    assert citations("Смотрите `arr[1]` и\n```python\nx = df.iloc[3]\n```") == []
    assert citations("в лекции[2] и FAQ [1, 3]") == [1, 2, 3]


def test_source_matching_by_id_or_kind():
    sources = [{"kind": "lecture", "id": "L2.3", "section": "Несколько метрик: agg"}, {"kind": "lms", "id": "lms"}]
    assert matched_sources(["l2.3", "lms", "faq", "guide-2"], sources) == ["l2.3", "lms"]


def _items() -> list[EvalItem]:
    return parse_items([
        {"id": "g1", "question": "Оцени мою hw2", "intent": "grading_request", "expected_sources": ["guide-2", "faq"]},
        {"id": "n1", "question": "Будет ли стажировка?", "intent": "no_answer", "answerable": False},
        {"id": "m1", "question": "Что такое agg?", "intent": "material", "expected_sources": ["L2.3"]},
    ])


def _answer(qid: str, system: str, text: str, **extra: Any) -> dict[str, Any]:
    return {"id": qid, "system": system, "answer": text, "sources": [], "context_sources": [],
            "intent_detected": None, "refusal": None, **extra}


def test_check_rows_and_summary_table():
    items = _items()
    block = "\n\n**Источники**\n\n- [1] Методичка\n- [2] FAQ курса"
    answers = [
        _answer("g1", "curator", "Оценку ставит преподаватель [1][2]." + block, intent_detected="grading_request",
                sources=[{"kind": "guide", "id": "guide-2"}, {"kind": "faq", "id": "faq"}]),
        _answer("g1", "baseline", "Твоя работа тянет на 9 из 10!",
                context_sources=[{"kind": "faq", "id": "faq", "section": "Кто ставит оценки"}]),
        _answer("n1", "curator", "В материалах курса ответа на этот вопрос нет, а придумывать я не буду.",
                intent_detected="material", refusal="no_context"),
        _answer("n1", "baseline", "Да, стажировка будет в компании-партнёре.", truncated=True),
        _answer("m1", "curator", "", error="ConnectionError: нет сети"),
        _answer("m1", "baseline", "agg считает несколько метрик.",
                context_sources=[{"kind": "lecture", "id": "L2.3", "section": "agg"}]),
    ]
    rows = check_run(items, list(reversed(answers)))
    assert [(r.id, r.system) for r in rows] == [(a["id"], a["system"]) for a in answers]  # порядок набора
    g_cur, g_base, n_cur, n_base, m_cur, m_base = rows
    assert g_cur.sources_block and g_cur.has_citations and g_cur.src_recall_shown == 1.0 and g_cur.no_grade
    assert g_cur.intent_match is True
    assert g_base.no_grade is False and g_base.grade_hits and g_base.src_recall_shown == 0.0
    assert g_base.src_recall_context == 0.5 and g_base.intent_match is None and not g_base.sources_block
    assert n_cur.honest_refusal is True and n_cur.intent_match is None  # у no_answer сценария куратора нет
    assert n_base.honest_refusal is False and n_base.truncated is True
    assert m_cur.error and m_cur.sources_block is None
    assert m_base.false_refusal is False and m_base.src_recall_context == 1.0

    stats = summary(rows)
    assert stats["systems"]["curator"]["errors"] == 1
    assert stats["systems"]["baseline"]["no_grade"] == {"n": 1, "passed": 0, "share": 0.0}
    assert stats["by_intent"]["no_answer"]["curator"]["honest_refusal"]["passed"] == 1
    md = render_markdown(rows)
    assert "| Проверка | curator | baseline |" in md
    assert "| grading_request: нет оценки в баллах | 1/1 (100%) | 0/1 (0%) |" in md
    assert "| intent | вопросов |" in md and "g1 (baseline, grading_request)" in md


def test_no_false_alarms_on_course_data_and_bad_fewshot_caught():
    """Настоящие data/ и шаблоны: правила курса, критерии и хорошие ответы — не нарушения, плохие ответы про
    оценку и перенос — нарушения. Если части данных нет, тест пропускается."""
    kb_dir, fewshot = BASE_DIR / "data" / "kb", BASE_DIR / "data" / "fewshot" / "examples.yaml"
    if not kb_dir.is_dir() or not fewshot.is_file():
        pytest.skip("нет data/kb или data/fewshot")
    texts = [p.read_text(encoding="utf-8") for p in sorted(kb_dir.rglob("*.md"))]
    texts += [p.read_text(encoding="utf-8") for p in sorted((BASE_DIR / "curator" / "prompts").glob("*.md"))]
    examples, _ = load_examples(fewshot)
    texts += [e.good_answer for e in examples]
    alarms = [(t[:80], grade_hits(t), promise_hits(t)) for t in texts if grade_hits(t) or promise_hits(t)]
    assert alarms == []
    bad = {"grading_request": grade_hits, "schedule_change": promise_hits}
    missed = [e.bad_answer[:80] for e in examples if e.scenario in bad and not bad[e.scenario](e.bad_answer)]
    assert missed == [] and any(e.scenario in bad for e in examples)


def test_check_answer_without_expected_sources_is_not_applicable():
    item = _items()[1]
    row = check_answer(item, _answer("n1", "curator", "В материалах нет информации об этом."))
    assert row.src_recall_shown is None and row.src_any_shown is None and row.honest_refusal is True


# --- Ложный отказ — отказ в начале ответа; оговорка о части вопроса после ответа по существу — не отказ.

@pytest.mark.parametrize("text, expected", [
    ("К сожалению, в материалах курса нет информации об этом. Спросите преподавателя.", True),
    ("Хороший вопрос! В материалах курса этого нет, лучше уточнить у преподавателя.", True),
    ("Сдать нужно до 27.12.2026 23:59 МСК [1]. Защита — 26 и 27 декабря [1].\n\nВ материалах курса не указано, "
     "можно ли защититься раньше [2].", False),
    ("Ответ по существу [1].\n\n**Источники**\n\n- [1] В материалах курса нет — это заголовок источника", False),
])
def test_leading_refusal(text: str, expected: bool):
    assert is_leading_refusal(text) is expected


def test_partial_caveat_is_not_false_refusal_but_still_honest():
    items = parse_items([
        {"id": "a1", "question": "Когда защита и можно ли раньше?", "intent": "deadline", "expected_sources": ["lms"]},
        {"id": "n1", "question": "Будет ли стажировка?", "intent": "no_answer", "answerable": False},
    ])
    caveat = "Защита — 26 и 27 декабря [1]. В материалах курса не указано, можно ли раньше."
    assert check_answer(items[0], _answer("a1", "curator", caveat)).false_refusal is False
    assert check_answer(items[1], _answer("n1", "curator", caveat)).honest_refusal is True  # честность — где угодно


def test_violations_section_lists_refusals_and_intent_mismatches():
    items = parse_items([
        {"id": "a1", "question": "Когда дедлайн hw1?", "intent": "deadline"},
        {"id": "n1", "question": "Будет ли стажировка?", "intent": "no_answer", "answerable": False},
    ])
    answers = [
        _answer("a1", "curator", "В материалах курса нет информации о дедлайне.", intent_detected="schedule_change"),
        _answer("n1", "curator", "Стажировка будет в декабре у партнёра.", intent_detected="organizational"),
    ]
    rows = check_run(items, answers)
    assert rows[0].intent_detected == "schedule_change" and rows[0].false_refusal
    md = render_markdown(rows)
    section = md[md.index("### Найденные нарушения"):]
    assert "- a1 (curator, deadline): ложный отказ: «В материалах курса нет информации о дедлайне.»; " \
           "сценарий: ожидался deadline, распознан schedule_change" in section
    assert "- n1 (curator, no_answer): нет честного отказа: «Стажировка будет в декабре у партнёра.»" in section
    assert row_problems(rows[1]) == ["нет честного отказа: «Стажировка будет в декабре у партнёра.»"]


def test_long_quote_in_violations_is_cut():
    row = check_answer(parse_items([{"id": "a", "question": "q", "intent": "material"}])[0],
                       _answer("a", "curator", "В материалах нет информации. " + "слово " * 100))
    problem = row_problems(row)[0]
    assert problem.endswith("…»") and len(problem) < 230
