"""Что код кладёт в контекст сам и как держит сроки: фильтры анализа — предпочтение, задание модуля (цель, данные,
шаги), модуль целиком (обзор + LMS), доработка (правило вместо «осталось N»), статус задания в шаблонах отказов,
недавние события, вопрос не о курсе без находок поиска; R3 — вопрос о правилах после анализа моделью,
задание в организационном вопросе, длительность в рекомендациях, тема курса в отказе off_topic."""
from __future__ import annotations

import re
from typing import Any

import pytest

from curator.curator import (CODE_PLACEHOLDER, GUIDE_SECTION_CHARS, Curator,  # pyright: ignore[reportPrivateUsage]
                             _guide_excerpt, _Query, remove_sentences, revision_deadline_violations,
                             strip_unsupported_durations)
from curator.kb import KnowledgeBase
from curator.memory import Turn
from curator.prompts import PromptLibrary
from curator.retriever import Hit, RetrievalResult, SearchFilter
from curator.router import LLMAnalyzer, QueryAnalysis, RuleAnalyzer
from curator.sources import Context, SourceRef
from tests.conftest import BuildCurator
from tests.fakes import FakeChat, FakeRouterLLM, FixedAnalyzer


def ask(curator: Curator, question: str, **kwargs: Any):
    kwargs.setdefault("session_id", "session-ctx")
    return curator.ask(question, **kwargs)


def _labels(chat: FakeChat) -> list[str]:
    return re.findall(r"^\[\d+\] (.+)$", str(chat.calls[-1][1].content), re.MULTILINE)


# --- R2-1: фильтры анализа — предпочтение, интерфейса — жёсткие.

def test_wrong_analysis_filter_does_not_hide_answer(build_curator: BuildCurator):
    analyzer = FixedAnalyzer(intent="material", module=1, topic="python-variables", resource_types=["lecture"])
    curator, _, journal = build_curator(analyzer=analyzer)
    answer = ask(curator, "Можно ли перенести дедлайн?")
    assert answer.retrieval is not None
    assert answer.retrieval.filter_hard == SearchFilter()
    assert answer.retrieval.filter_requested == SearchFilter(1, ("lecture",), "python-variables")
    assert any(h.section == "Можно ли перенести дедлайн?" for h in answer.retrieval.hits)  # FAQ вне фильтра анализа
    row = journal.fetch()[0]
    assert row["relaxed"] == 1 and row["filter"] == answer.retrieval.filter_used.describe()


def test_ui_filters_are_hard_and_module_named_in_question_is_soft(build_curator: BuildCurator):
    curator, _, _ = build_curator(analyzer=FixedAnalyzer(intent="material", module=1))
    ui = ask(curator, "Как отфильтровать строки по условию?", module=2, resource_type="lecture")
    assert ui.retrieval is not None and ui.retrieval.filter_hard == SearchFilter(2, ("lecture",))
    assert {h.metadata["resource_type"] for h in ui.retrieval.hits} == {"lecture"}
    named = ask(curator, "Когда дедлайн hw1?", module=2, session_id="other")  # модуль назван прямо — перебил UI
    assert named.module == 1 and named.retrieval is not None and named.retrieval.filter_hard.module is None


# --- R2-2: вопрос о задании модуля — цель, данные, шаги методички.

def test_assignment_question_adds_guide_goal_and_steps(build_curator: BuildCurator):
    analyzer = FixedAnalyzer(intent="material", module=2, resource_types=["guide", "lecture"])
    curator, chat, _ = build_curator(analyzer=analyzer)
    ask(curator, "Что именно требуется сделать в работе про кофейню?")
    labels = _labels(chat)
    assert "Методичка к ДЗ hw2 «Продажи кофейни», раздел «Цель ДЗ»" in labels
    assert "Методичка к ДЗ hw2 «Продажи кофейни», раздел «Шаги»" in labels
    prompt = chat.prompt_text()
    assert "### Шаг 2. Группировка" in prompt
    assert prompt.count("Посчитать выручку кофейни по точкам") == 1  # находка поиска не дублирует раздел


ABOUT_HW2 = Turn("Расскажите про второй модуль", "Модуль 2 — pandas; задание модуля — hw2 «Продажи кофейни».")
ABOUT_MODULE = Turn("Расскажите про второй модуль", "Модуль 2 — pandas: лекции L2.1-L2.3.")
ABOUT_REVIEW = Turn("Когда ближайший вебинар?", "Вебинар — в среду в 19:00 МСК, разбор ДЗ — в пятницу в 18:00.")


def _query(question: str, **fields: Any) -> _Query:
    base: dict[str, Any] = {"intent": "material", "question": question, "history": [], "module": 2, "topic": None,
                            "hard": SearchFilter(), "prefer": SearchFilter(), "asked_types": (), "level": "beginner",
                            "student": None, "uses_student_data": False}
    return _Query(**(base | fields))


@pytest.mark.parametrize("question, fields, expected", [
    ("Что сдавать в hw2?", {"asked_types": ("lecture",)}, True),  # слова о задании
    ("Объясни ещё раз про домашку", {"intent": "difficulty"}, True),
    ("Какие тут данные?", {"topic": "hw2-coffee-sales"}, True),  # тема методички
    ("Какие тут данные?", {"asked_types": ("guide", "lecture")}, True),  # основной тип — методичка
    ("Чем loc отличается от iloc?", {"asked_types": ("lecture", "guide")}, False),  # основной тип — лекция
    ("Что сдавать в hw2?", {"intent": "recommendation"}, False),  # не тот сценарий
    ("Что сдавать в hw2?", {"module": None}, False),  # модуль неизвестен
    # R3-3: в организационном вопросе и вопросе о сроках — только если спрашивают, что в задании сделать.
    ("Что нужно сделать в hw2?", {"intent": "organizational"}, True),
    ("Какие шаги в задании по кофейне?", {"intent": "deadline"}, True),
    ("Как сдать hw2 ссылкой на Colab?", {"intent": "organizational"}, False),
    ("Что делать, если не успеваю сдать hw2?", {"intent": "deadline"}, False),
    ("Дедлайн hw2 уже прошёл, что теперь делать?", {"intent": "deadline"}, False),
    ("Что нужно сделать, чтобы получить сертификат?", {"intent": "organizational"}, False),  # не о задании
    # Короткое уточнение — о задании, если о нём была прошлая реплика.
    ("а что там нужно сделать?", {"intent": "organizational", "history": [ABOUT_HW2]}, True),
    ("а что там нужно сделать?", {"asked_types": ("lecture",), "history": [ABOUT_HW2]}, True),
    ("а что там нужно сделать?", {"intent": "organizational", "history": [ABOUT_MODULE]}, False),
    # «Разбор ДЗ», «защита итогового проекта» — события, а не задание.
    ("Что нужно сделать к разбору ДЗ?", {"intent": "organizational"}, False),
    ("а что там нужно сделать?", {"intent": "organizational", "history": [ABOUT_REVIEW]}, False),
    ("а что там нужно сделать?", {"intent": "organizational"}, False),
])
def test_assignment_detection(build_curator: BuildCurator, question: str, fields: dict[str, Any], expected: bool):
    curator, _, _ = build_curator()
    assert curator._is_about_assignment(_query(question, **fields)) is expected  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("fields, expected", [
    ({"module_overview": True, "topic": "pandas-filtering", "intent": "recommendation"}, True),
    ({"topic": None}, True), ({"topic": None, "intent": "organizational"}, True),
    ({"topic": None, "intent": "difficulty"}, False), ({"topic": "pandas-filtering"}, False),
    ({"module_overview": True, "module": None}, False),
])
def test_module_overview_detection(fields: dict[str, Any], expected: bool):
    assert Curator._is_module_overview(_query("вопрос", **fields)) is expected  # pyright: ignore[reportPrivateUsage]


def test_long_guide_section_drops_code_then_cuts():
    text = "Шаг 1. Прочитайте файл.\n\n```python\n" + "x = 1\n" * 600 + "```\n\nШаг 2. Посчитайте выручку."
    short = _guide_excerpt(text)
    assert CODE_PLACEHOLDER in short and "Шаг 2. Посчитайте выручку." in short and "x = 1" not in short
    assert _guide_excerpt("Короткий раздел.\n\n```python\nx = 1\n```").endswith("```")  # влезает — код остаётся
    long = _guide_excerpt("Абзац текста. " * 400)
    assert len(long) <= GUIDE_SECTION_CHARS + 2 and long.endswith("…")


# --- R2-3: модуль целиком — раздел обзора и данные LMS о задании модуля.

@pytest.mark.parametrize("fields", [{"module_overview": True, "topic": "pandas-filtering"}, {"topic": None}])
def test_module_overview_question_gets_overview_and_lms(build_curator: BuildCurator, fields: dict[str, Any]):
    analyzer = FixedAnalyzer(intent="material", module=2, **fields)
    curator, chat, _ = build_curator(analyzer=analyzer)
    answer = ask(curator, "Расскажите, что ждёт во втором модуле")
    labels = _labels(chat)
    assert "Обзор курса «Python для анализа данных», раздел «Модуль 2. Pandas: работа с таблицами (16-29 ноября " \
           "2026)»" in labels
    prompt = chat.prompt_text()
    assert "hw2 «Продажи кофейни» (модуль 2): откроется пн 16.11.2026 00:00 МСК" in prompt
    assert answer.lms_used


def test_specific_topic_question_gets_no_module_overview(build_curator: BuildCurator):
    curator, chat, _ = build_curator(analyzer=FixedAnalyzer(intent="material", module=2, topic="pandas-filtering"))
    answer = ask(curator, "Чем loc отличается от iloc?")
    assert not any("Модуль 2." in label for label in _labels(chat)) and not answer.lms_used


# --- R2-4: доработка — правило из FAQ, few-shot с условием, выходная проверка срока.

def test_revision_student_gets_rule_instead_of_time_left(build_curator: BuildCurator):
    analyzer = FixedAnalyzer(intent="deadline", module=1, uses_student_data=True)
    curator, chat, _ = build_curator(analyzer=analyzer)
    ask(curator, "Сколько у меня времени на исправления по первой работе?", student_id="s3")
    prompt = chat.prompt_text()
    hw1 = next(line for line in prompt.splitlines() if line.startswith("- hw1 "))
    assert "осталось" not in hw1 and "к доработке не относится" in hw1
    assert "5 дней с момента решения преподавателя" in hw1
    assert "FAQ: «Сколько времени даётся на доработку?»" in _labels(chat)
    assert "Работу вернули на исправление, сдавать до дедлайна задания?" in prompt  # few-shot с условием revision


def test_tagged_fewshot_absent_without_revision(build_curator: BuildCurator):
    curator, chat, _ = build_curator(analyzer=FixedAnalyzer(intent="deadline", module=1, uses_student_data=True))
    ask(curator, "Когда дедлайн hw1?", student_id="s1")
    assert "Работу вернули на исправление" not in chat.prompt_text()


def test_revision_deadline_in_model_answer_is_replaced_by_rule(build_curator: BuildCurator):
    reply = ("Ошибка в запятой: замените её на точку [2]. Исправьте и пришлите работу до 15.11.2026 23:59 МСК [1]. "
             "Если что-то непонятно — спрашивайте.")
    curator, _, journal = build_curator(chat=FakeChat(reply=reply),
                                        analyzer=FixedAnalyzer(intent="deadline", module=1, uses_student_data=True))
    answer = ask(curator, "Что делать с доработкой?", student_id="s3")
    assert "до 15.11.2026" not in answer.body
    assert answer.body.startswith("Ошибка в запятой: замените её на точку")
    assert "Если что-то непонятно — спрашивайте." in answer.body
    assert "Срок доработки hw1 считается по своему правилу [" in answer.body
    assert "5 дней с момента решения преподавателя" in answer.body
    assert answer.guard == "output:revision_deadline" and answer.refusal == "" and answer.answered
    assert journal.fetch()[0]["guard"] == "output:revision_deadline"
    assert any(i.source.section == "Сколько времени даётся на доработку?" for i in answer.sources)


def test_revision_rule_added_when_question_is_about_revision_without_lms(build_curator: BuildCurator):
    """Вопрос о доработке в сценарии «трудно» (данных LMS в промпте нет) — правило и проверка срока всё равно есть."""
    reply = "Замените запятую на точку [1]. Исправленную работу пришлите до 15.11 [1]."
    curator, chat, _ = build_curator(chat=FakeChat(reply=reply), analyzer=FixedAnalyzer(intent="difficulty", module=1))
    answer = ask(curator, "Работу вернули на доработку из-за ошибки с float, что исправить?", student_id="s3")
    assert not answer.lms_used and "FAQ: «Сколько времени даётся на доработку?»" in _labels(chat)
    assert answer.guard == "output:revision_deadline" and "до 15.11" not in answer.body
    plain, chat2, _ = build_curator(analyzer=FixedAnalyzer(intent="difficulty", module=1))
    ask(plain, "Чем loc отличается от iloc?", student_id="s3")
    assert not any("доработку" in label for label in _labels(chat2))  # вопрос не о доработке и без LMS — без правила


def test_revision_guard_off_for_student_without_revision(build_curator: BuildCurator):
    reply = "Исправьте ошибку и сдайте работу до 15.11.2026 23:59 МСК [1]."
    curator, _, _ = build_curator(chat=FakeChat(reply=reply),
                                  analyzer=FixedAnalyzer(intent="deadline", module=1, uses_student_data=True))
    answer = ask(curator, "Когда сдавать?", student_id="s1")
    assert "до 15.11.2026 23:59" in answer.body and answer.guard == ""


@pytest.mark.parametrize("text", [
    "Исправленную работу пришлите до 15.11.2026.",
    "На доработку у вас осталось 3 дня 9 часов.",
    "Вы успеете исправить и сдать работу до дедлайна.",
    "Сдать доработку нужно к 18 ноября.",
    "На исправление есть 2 дня.",
    "Работу нужно сдать снова до вс 15.11.2026 23:59 МСК.",
])
def test_revision_deadline_violations_found(text: str):
    assert revision_deadline_violations(text, {"hw1"}) == [text]


@pytest.mark.parametrize("text", [
    "На исправление есть 5 дней с момента решения преподавателя.",
    "Исходный дедлайн 15.11 к доработке не относится.",
    "Отсчёт идёт от решения преподавателя, а не до дедлайна задания.",
    "Вопросы по доработке можно принести на разбор в пт 13.11.2026 18:00 МСК.",
    "hw2 нужно сдать до 29.11.2026 23:59 МСК — исправьте ошибки заранее.",
    "Дедлайн hw1 — 15.11.2026 23:59 МСК, осталось 3 дня.",
    "```python\n# исправить до 15.11\nx = 1\n```",
])
def test_revision_rule_statements_are_not_violations(text: str):
    assert revision_deadline_violations(text, {"hw1"}) == []


def test_remove_sentences_keeps_code_and_rest():
    text = "Первое. Исправьте до 15.11.\n\n```python\nx = 1\n```\n\nТретье."
    assert remove_sentences(text, ["Исправьте до 15.11."]) == "Первое.\n\n```python\nx = 1\n```\n\nТретье."


# --- R2-6: статус задания в шаблонах отказов.

@pytest.mark.parametrize("student, expected", [
    ("s4", "По данным LMS статус hw1 «Анализ списка покупок» у вас — «на проверке» ["),
    ("s2", "По данным LMS статус hw1 «Анализ списка покупок» у вас — «принято» ["),
    ("s3", "По данным LMS hw1 «Анализ списка покупок» у вас — «на доработку» ["),
])
def test_grading_template_shows_student_status(build_curator: BuildCurator, student: str, expected: str):
    curator, chat, _ = build_curator()
    answer = ask(curator, "Поставь мне 10 баллов за hw1", student_id=student)
    assert chat.calls == [] and expected in answer.body
    if student == "s3":
        assert "5 дней с момента решения преподавателя" in answer.body


def test_grading_template_without_student_has_no_status(build_curator: BuildCurator):
    curator, _, _ = build_curator()
    assert "По данным LMS статус" not in ask(curator, "Поставь мне 10 баллов за hw1").body


def test_schedule_template_shows_status_and_revision_rule(build_curator: BuildCurator):
    curator, _, _ = build_curator()
    accepted = ask(curator, "Перенесите мне дедлайн hw1 на три дня", student_id="s2")
    assert "Статус задания у вас по данным LMS — «принято» [" in accepted.body
    revision = ask(curator, "Перенесите мне дедлайн hw1 на три дня", student_id="s3", session_id="s3")
    assert "Задание у вас по данным LMS — «на доработку» [" in revision.body
    assert "исходный дедлайн задания к доработке не относится" in revision.body.lower()
    assert "Подать заявку на перенос по правилу ещё можно" not in revision.body
    assert "осталось" not in revision.body
    assert "в комментариях к работе в LMS" in revision.body
    # В FAQ только «прочитайте комментарии преподавателя»: что в них видна дата решения, материалы не говорят.
    assert "видна в комментари" not in revision.body and "Дату решения" not in revision.body
    assert "только он" not in accepted.body  # преподаватель курса — Ирина Соколова; в шаблоне — «преподаватель»


@pytest.mark.parametrize("question, events", [
    ("Перенесите разбор ДЗ на субботу", True),  # «ДЗ» — часть названия события, а не срок
    ("Сдвиньте защиту итогового проекта на январь", True),
    ("Перенесите дедлайн hw2 и разбор на неделю", False),  # есть и срок — сценарий о дедлайне
])
def test_event_names_with_deadline_words_are_events(build_curator: BuildCurator, question: str, events: bool):
    curator, _, _ = build_curator()
    answer = ask(curator, question)
    assert answer.refusal == "schedule_change"
    assert answer.body.startswith("Изменить расписание я не могу") is events


# --- R2-7: недавние события и формат сдачи в фактах LMS.

def test_deadline_prompt_has_recent_events_and_submit_format(build_curator: BuildCurator):
    curator, chat, _ = build_curator(analyzer=FixedAnalyzer(intent="deadline", uses_student_data=True))
    ask(curator, "Что мне сдавать и до когда?", student_id="s1")
    prompt = chat.prompt_text()
    assert "Прошедшие события за последние 3 дня:\n- ср 11.11.2026 19:00 МСК — Вебинар 2. Функции и модули" in prompt
    assert "Формат сдачи: Ссылка на Google Colab или файл .ipynb в LMS." in prompt


def test_recommendation_scope_follows_top_hit_from_other_module():
    """Анализ назвал модуль 3, а лучший фрагмент — лекция модуля 4: каталог ресурсов — по модулю 4."""
    def result(module: int, topic: str) -> RetrievalResult:
        hit = Hit("c", "t", 0.5, {"module": module, "resource_type": "lecture", "topic": topic})
        return RetrievalResult([hit], SearchFilter(3, ("lecture",), "data-storytelling"), SearchFilter())

    scope = Curator._recommendation_scope  # pyright: ignore[reportPrivateUsage]
    assert scope(3, "data-storytelling", result(4, "project-defense")) == (4, "project-defense")
    assert scope(3, "data-storytelling", result(3, "seaborn-chart-choice")) == (3, "data-storytelling")
    assert scope(None, None, result(4, "project-defense")) == (4, "project-defense")


# --- R2-5 (г): вопрос не о курсе, по которому поиск ничего не нашёл.

def test_leisure_request_without_hits_becomes_off_topic(build_curator: BuildCurator):
    curator, chat, journal = build_curator(analyzer=FixedAnalyzer(intent="recommendation"))
    answer = ask(curator, "Посоветуй сериал на выходные, хочу отвлечься")
    assert chat.calls == [] and answer.intent == "off_topic" and answer.refusal == "off_topic"
    assert not any(i.source.kind == "resource" for i in answer.sources)
    assert journal.fetch()[0]["intent"] == "off_topic"


def test_unscoped_recommendation_gets_no_generic_resources(build_curator: BuildCurator):
    curator, chat, _ = build_curator(analyzer=FixedAnalyzer(intent="recommendation"))
    answer = ask(curator, "zzzz qqqq xxxx")
    assert chat.calls == [] and answer.refusal == "no_context"
    assert not any(i.source.kind == "resource" for i in answer.sources)


# --- R3-1: анализ моделью назвал grading_request вопрос о правилах — отвечает модель, прогноз оценки всё равно ловится.

def _model_says(kb: KnowledgeBase, intent: str, topic: str | None = None) -> LLMAnalyzer:
    analysis = QueryAnalysis(intent=intent, module=None, topic=topic,  # pyright: ignore[reportArgumentType]
                             resource_types=["guide", "faq"], level_hint=None, uses_student_data=False,
                             module_overview=False)
    return LLMAnalyzer(FakeRouterLLM(analysis=analysis), PromptLibrary(), modules="- 1. Основы Python",
                       topics=kb.topics(), fallback=RuleAnalyzer(list(kb.topics())))


def test_rules_question_misread_as_grading_is_answered_by_model(build_curator: BuildCurator, kb: KnowledgeBase):
    curator, chat, journal = build_curator(chat=FakeChat(reply="Нужно, чтобы были приняты 3 задания из 4 [1]."),
                                           analyzer=_model_says(kb, "grading_request", "certificate"))
    answer = ask(curator, "Нужна ли защита для сертификата, если все домашки приняты?")
    assert answer.intent == "organizational" and answer.refusal == "" and len(chat.calls) == 1
    assert answer.guard == "grading_request->organizational" and answer.topic == "certificate"
    assert "FAQ: «Как получить сертификат?»" in _labels(chat)
    assert journal.fetch()[0]["guard"] == "grading_request->organizational"


def test_grade_forecast_still_refused_after_rules_downgrade(build_curator: BuildCurator, kb: KnowledgeBase):
    """Безопасность не просела: прогноз своей оценки — отказ без модели; прогноз в ответе модели — замена шаблоном."""
    curator, chat, _ = build_curator(analyzer=_model_says(kb, "grading_request"))
    refused = ask(curator, "Какой балл мне поставят за работу?")
    assert refused.refusal == "grading_request" and chat.calls == []
    assert "Поставить или предсказать оценку я не могу" in refused.body
    leaky, _, _ = build_curator(chat=FakeChat(reply="По критериям вам поставят 8 из 10 [1]."),
                                analyzer=_model_says(kb, "grading_request", "grading"))
    replaced = ask(leaky, "Как считаются баллы за проект по критериям?", session_id="leaky")
    assert replaced.refusal == "output_guard" and "8 из 10" not in replaced.body
    assert "Поставить или предсказать оценку я не могу" in replaced.body


# --- R3-3: «что сделать в задании» в организационном сценарии — цель, данные и шаги методички.

def test_task_question_in_organizational_scenario_gets_guide_sections(build_curator: BuildCurator):
    curator, chat, _ = build_curator(analyzer=FixedAnalyzer(intent="organizational", module=2))
    ask(curator, "Что нужно сделать в hw2 и в каком виде сдавать?")
    labels = _labels(chat)
    assert "Методичка к ДЗ hw2 «Продажи кофейни», раздел «Цель ДЗ»" in labels
    assert "Методичка к ДЗ hw2 «Продажи кофейни», раздел «Шаги»" in labels
    ask(curator, "Как сдать hw2 ссылкой на Colab?", session_id="submit")
    assert "Методичка к ДЗ hw2 «Продажи кофейни», раздел «Цель ДЗ»" not in _labels(chat)


def test_followup_about_assignment_gets_guide_sections(build_curator: BuildCurator):
    curator, chat, _ = build_curator(analyzer=FixedAnalyzer(intent="material", module=2, resource_types=["lecture"]))
    curator.memory.add("follow", ABOUT_HW2)
    ask(curator, "а что там нужно сделать?", session_id="follow")
    assert "Методичка к ДЗ hw2 «Продажи кофейни», раздел «Шаги»" in _labels(chat)


def test_material_prompt_asks_for_goal_data_and_steps():
    scenario = PromptLibrary().scenario("material")
    assert "перескажи по методичке все три части" in scenario and "цель, данные" in scenario and "шаги" in scenario
    assert "Пересказать по методичке, что нужно сделать в задании" in PromptLibrary().system(today="чт")


# --- R3-4: длительность в рекомендации — только из материалов.

def _duration_ctx() -> Context:
    ctx = Context()
    ctx.add(SourceRef("lecture", "L2.1", "Series и DataFrame", "Для продвинутых", module=2),
            "Тип category экономит память.")
    ctx.add(SourceRef("resource", "res-18", "pandas: индексация", url="https://pandas.pydata.org/x"),
            "pandas: индексация (документация, ~60 мин) — https://pandas.pydata.org/x\nloc, iloc, маски.")
    ctx.add(SourceRef("resource", "res-10", "Kaggle Learn: Python", url="https://www.kaggle.com/learn/python"),
            "Kaggle Learn: Python (курс, ~300 мин) — https://www.kaggle.com/learn/python\nУпражнения.")
    return ctx


@pytest.mark.parametrize("text, expected", [
    ("1. **Раздел «Для продвинутых»** [1] — про память и Arrow. Займёт около 30 минут.",
     "1. **Раздел «Для продвинутых»** [1] — про память и Arrow."),
    ("- Раздел L2.1 [1] — полезно для больших таблиц, около 40 минут [1].",
     "- Раздел L2.1 [1] — полезно для больших таблиц [1]."),
    ("- Раздел L2.1 (около 15 мин) [1] пригодится.", "- Раздел L2.1 [1] пригодится."),
])
def test_invented_durations_are_cut(text: str, expected: str):
    assert strip_unsupported_durations(text, _duration_ctx()) == (expected, True)


@pytest.mark.parametrize("text", [
    "2. **Документация pandas** [2] — про loc и iloc, около 60 минут, на английском.",
    "3. **Kaggle Learn: Python** [3] — упражнения в браузере, примерно 5 часов.",  # 5 часов = ~300 мин
    "Время по каталогу — около 60 минут.",  # без ссылки — сверяется с ресурсами каталога
    "```python\n# около 30 минут на загрузку\n```",
])
def test_durations_from_materials_are_kept(text: str):
    assert strip_unsupported_durations(text, _duration_ctx()) == (text, False)


def test_recommendation_answer_loses_invented_duration(build_curator: BuildCurator):
    reply = ("1. **Раздел «Если сложно»** [1] — повторите маски. Займёт около 17 минут.\n"
             "2. Потом — пример из лекции [1].")
    curator, _, journal = build_curator(chat=FakeChat(reply=reply), analyzer=FixedAnalyzer(
        intent="recommendation", module=2, topic="pandas-filtering"))
    answer = ask(curator, "Что почитать про маски в pandas?", level_mode="beginner")
    assert "17 минут" not in answer.body
    assert answer.body.startswith("1. **Раздел «Если сложно»** [1] — повторите маски.\n")
    assert "2. Потом — пример из лекции [1]." in answer.body
    assert answer.guard == "output:duration" and journal.fetch()[0]["guard"] == "output:duration"


def test_recommendation_prompt_limits_time_to_catalog_resources():
    scenario = PromptLibrary().scenario("recommendation")
    assert "Время называй только у дополнительного ресурса" in scenario
    assert "У разделов лекций и методичек длительности в материалах нет" in scenario


# --- R3-8: просьба не по курсу с темой курса — в отказе строка про эту тему.

def test_off_topic_with_course_topic_offers_that_topic(build_curator: BuildCurator):
    curator, chat, _ = build_curator()
    answer = ask(curator, "Посоветуй музыку для фона, весь день сижу над переменными и типами")
    assert answer.refusal == "off_topic" and chat.calls == []
    assert "А вот с темой «Переменные и типы данных» [2] помогу" in answer.body
    assert [i.source.ref_id for i in answer.sources] == ["course-overview", "L1.1"]
    plain = ask(curator, "Какая погода будет в Москве на выходных?", session_id="weather")
    assert "А вот с темой" not in plain.body and "\n\n\n" not in plain.body
    hello = ask(curator, "Привет!", session_id="hello")
    assert hello.refusal == "small_talk" and "А вот с темой" not in hello.body


# --- R3-7: вопрос об опоздании — правило FAQ о несданной в срок работе в контексте.

@pytest.mark.parametrize("question, added", [
    ("Что будет, если я не успею сдать hw2 к новому сроку?", True),
    ("Дедлайн hw2 уже прошёл, что теперь делать?", True),
    ("Когда дедлайн hw2?", False),
])
def test_lateness_question_gets_late_work_rule(build_curator: BuildCurator, question: str, added: bool):
    """Код кладёт правило сразу после данных LMS, до находок поиска (в маленькой базе фикстур поиск может найти
    его и сам — но ниже)."""
    curator, chat, _ = build_curator(analyzer=FixedAnalyzer(intent="deadline", module=2))
    ask(curator, question)
    labels = _labels(chat)
    assert labels[0].startswith("LMS:")
    assert (labels[1] == "FAQ: «Что будет, если я не успею сдать ДЗ к дедлайну?»") is added
