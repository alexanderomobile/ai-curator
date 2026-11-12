"""Оркестратор: лимиты → обезличивание → анализ запроса → сценарий → материалы → ответ → «Источники» → журнал.

Сценарии grading_request, schedule_change и off_topic — фиксированные: ответ собирает код из шаблонов
(prompts/refusal_*.md) и материалов курса, модель не вызывается, поэтому оценку или перенос она «пообещать»
не может. Для остальных сценариев отвечает модель по материалам; если в её ответе всё же нашлось обещание оценки
или переноса (проверка регулярными выражениями), ответ заменяется фиксированным сценарием.

Сбои внешних сервисов студент видит как понятный ответ, а не как исключение: LMS недоступна — ответ без данных
LMS, не ответил поиск (эмбеддинги OpenAI) — шаблон search_error, не ответила модель — список материалов.
Такие ответы тоже пишутся в журнал.

Модуль, выбранный в интерфейсе, важнее модуля из анализа запроса; подменить его может только модуль, названный
в вопросе прямо («hw3», «модуль 2», «L2.2»). Выбор в интерфейсе (модуль, тип, тема) — жёсткий фильтр поиска,
модуль, тип и тема из анализа — предпочтение: фрагмент вне них попадёт в ответ, если он заметно ближе к вопросу
(curator/retriever.py).

Часть материалов код кладёт в контекст сам, не надеясь на поиск по смыслу:
- вопрос о задании модуля (что сделать, данные, шаги; в организационном вопросе и вопросе о сроках — только если
  спрашивают, что в задании сделать) — цель, данные и шаги из методички модуля;
- вопрос о модуле целиком — раздел обзора курса про модуль и данные LMS о его задании;
- у студента задание на доработке — правило доработки из FAQ (и в данных LMS вместо «осталось N»);
- вопрос о том, что будет при опоздании («не успею», «дедлайн прошёл») — правило FAQ о несданной в срок работе.
Если модель всё же назвала дату или «осталось N» как срок доработки, такое предложение вырезается, а в конец
ответа добавляется правило (выходная проверка revision). В рекомендации длительность «около N минут», которой нет
в материалах, на которые ссылается тот же пункт ответа, вырезается (выходная проверка duration): время есть только
у ресурсов каталога, у разделов лекций его нет.

Отказ off_topic, в котором упомянута тема курса («посоветуй фильм, устал от группировки»), предлагает помощь
именно с этой темой — со ссылкой на её лекцию.

Источники в ответе нумеруются подряд, в порядке первого упоминания: [1], [2], [3] — без пропусков, даже если модели
давали больше материалов.

Все зависимости приходят снаружи (curator.factory собирает настоящие, тесты — фейковые без сети).
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .config import Settings
from .fewshot import FewShotLibrary
from .journal import Journal, Record
from .kb import KnowledgeBase, Section
from .limits import DemoLimits
from .lms import (REVISION_DAYS, STATUS_LABELS, Assignment, LMSClient, LMSError, Student, deadline_status,
                  event_when, fmt_dt, revision_rule)
from .memory import DialogMemory, Turn
from .privacy import redact, user_hash
from .prompts import PromptLibrary, clean_question, quote, tidy
from .recommender import Recommender
from .retriever import RetrievalResult, Retriever, SearchFilter
from .router import (INTENT_RESOURCE_TYPES, AnalysisResult, Analyzer, detect_topic, explicit_module, leisure_request,
                     off_topic_marker, small_talk_kind)
from .sources import (Context, ContextItem, SourceRef, add_hits, citation_numbers, process_citations, renumber,
                      sources_block, strip_model_sources, used_items)

__all__ = ["Curator", "CuratorAnswer", "clean_question", "output_violation"]

log = logging.getLogger("curator")

WEEKDAYS_FULL = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")
LEVELS = ("beginner", "advanced")
LEVEL_MODES = ("auto", "beginner", "advanced")
HISTORY_ANSWER_CHARS = 1200
EXCERPT_CHARS = 600
CRITERIA_CHARS = 1400
FOLLOWUP_WORDS = 6
# Сколько дополнительных ресурсов и разделов лекций добавлять по сценарию. Для рекомендаций сценарий требует
# 2-3 пункта: в материалы идут 2 ресурса и 1 раздел лекции, иначе модель перечисляет всё, что ей дали.
RESOURCES_BY_INTENT = {"recommendation": 2, "difficulty": 2, "material": 1}
SECTIONS_BY_INTENT = {"recommendation": 1, "difficulty": 1}
# Сколько найденных фрагментов давать модели по сценарию (остальным — все top_k найденных).
HITS_BY_INTENT = {"recommendation": 2}
LMS_INTENTS = ("deadline", "organizational")
# Вопрос о задании модуля: код сам добавляет цель, данные и шаги методички (как критерии в отказе об оценке).
# 3500 знаков вмещают самый длинный из этих разделов в методичках курса («Данные» guide-4 — 3438 знаков) целиком,
# с кодом: тогда находки поиска из того же раздела не дублируют его в промпте. Три раздела — до ~5100 знаков
# (1300-1600 токенов). Раздел длиннее (база выросла) сначала сжимается — блоки кода заменяются пометкой, — потом
# обрезается по абзацу.
ASSIGNMENT_INTENTS = ("material", "difficulty")
# В организационном вопросе и вопросе о сроках слова «ДЗ», «задание» есть почти всегда («как сдать ДЗ», «когда
# дедлайн задания»): разделы методички добавляются, только если спрашивают, что в задании сделать (_ASSIGNMENT_TASK).
ASSIGNMENT_TASK_INTENTS = ("organizational", "deadline")
ASSIGNMENT_SECTION_KINDS = ("goal", "data", "steps")
GUIDE_SECTION_CHARS = 3500
# Вопрос о модуле целиком: раздел обзора курса про модуль и данные LMS о его задании.
OVERVIEW_INTENTS = ("material", "organizational")
# Вопрос о том, что будет при опоздании: код кладёт в контекст правило FAQ о несданной в срок работе (_add_late_rule).
LATE_RULE_INTENTS = ("deadline", "organizational")
# Где выходная проверка вырезает длительность, которой нет в материалах (strip_unsupported_durations): в рекомендациях
# «что изучить» модель называет время к каждому пункту. В ответах по материалу «около 30 минут» бывает фактом
# из данных (длительность поездки в датасете), поэтому там проверки нет.
DURATION_CHECK_INTENTS = ("recommendation",)
# Отказ по лимиту: в журнал — начало вопроса. Обезличивается окно побольше, чтобы почта или телефон на границе
# обрезки не попали в журнал наполовину; окно ограничено, чтобы огромная вставка не гонялась через регулярки.
LIMITED_LOG_CHARS = 300
LIMITED_SCAN_CHARS = 3000

_EVENT_WORDS = re.compile(r"вебинар|разбор|защит|заняти|расписан|эфир|встреч", re.IGNORECASE)
_DEADLINE_WORDS = re.compile(r"дедлайн|срок|\bдз\b|домашк|задани|сдач|сдать|проект|hw[1-3]", re.IGNORECASE)
# Названия событий со словами сроков внутри: «разбор ДЗ», «защита проекта» — это событие, а не дедлайн.
_EVENT_PHRASES = re.compile(r"разбор\w*\s+(?:дз|д/з|домашн\w*(?:\s+задани\w*)?|домашек|домашки)\b|"
                            r"защит\w*\s+(?:итогов\w*\s+)?проект\w*", re.IGNORECASE)
# Вопрос о задании: слова «ДЗ», «домашка», «hw2», «итоговый проект», «задание».
_ASSIGNMENT_WORDS = re.compile(r"\bдз\b|д/з|домашк|домашн\w*\s+(?:задани|работ)|\bhw\s*-?\d|\bfinal\b|"
                               r"итогов\w*\s+проект|задани", re.IGNORECASE)
# Вопрос о содержании задания: что сделать, что требуется, какие шаги и данные, с чего начать.
# «Что теперь делать?» (опоздал, не понимаю) — не о содержании задания, поэтому глагол — «сделать», а не «делать».
_ASSIGNMENT_TASK = re.compile(r"что\s+(?:\w+\s+){0,5}?(?:сделать|выполнить|требуется|требуют)\b|"
                              r"(?:в\s+ч[её]м|какая|какие)\s+(?:\w+\s+)?(?:суть|задача|цель|шаги|требовани\w*)\b|"
                              r"какие\s+(?:\w+\s+)?данные|с\s+чего\s+(?:\w+\s+)?начать", re.IGNORECASE)
# «Не успею», «опоздаю», «просрочил», «дедлайн прошёл» — вопрос о том, что будет при опоздании. Без правила FAQ
# о несданной в срок работе модель выводила последствия из соседних вопросов (там — доработка уже сданной работы)
# и обещала доработку тому, кто работу к сроку не сдал.
_LATE_WORDS = re.compile(r"не\s+успе\w*|опозда\w*|просроч\w*|после\s+(?:дедлайна|срока)|"
                         r"(?:дедлайн|срок)\w*\s+(?:\w+\s+){0,2}?прош\w*", re.IGNORECASE)
_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
CODE_PLACEHOLDER = "(пример кода — в методичке)"


def _ci(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# Обещания оценки или переноса в ответе модели. Цитаты правил («оценки ставит преподаватель», «за группировку
# по критериям — 5 баллов», «если заявку одобрят, новый срок появится в LMS») сюда не попадают — это проверено
# тестами в обе стороны.
_NOT = r"(?:(?!не\b)\S+\s+)?"  # одно слово между подлежащим и сказуемым, но не «не»
_OUTPUT_GRADING = [
    # «ставлю вам 9», «засчитываю вашу работу», «ставлю вам пятёрку за hw1»: до двух слов-дополнений между глаголом
    # и предметом оценки.
    _ci(r"\b(?:ставлю|поставлю|выставляю|выставлю|засчитываю|засчитаю|зачту|приму)\s+"
        r"(?:(?:вам|тебе|вашу|твою|эту|ваше|твое|ваш|твой)\s+|за\s+\S+\s+){0,2}"
        r"(?:\d{1,2}\b|балл|оценк|зачет|работ|дз|домашк|проект|пят[её]рк|четв[её]рк)"),
    _ci(r"\bоцениваю\s+(?:вашу|твою|эту)\s+работу\b"),
    # «я бы поставил вам 9», «мы оценили вашу работу» — с предметом оценки; «я бы поставил легенду справа»,
    # «мы поставили ascending=False», «мы оценили медиану» — объяснение кода, не оценка.
    _ci(r"\b(?:я|мы)\s+(?:бы\s+)?(?:поставил|поставила|поставили|оценил|оценила|оценили|засчитал|засчитала|"
        r"засчитали|принял|приняла|приняли)\s+(?:бы\s+)?(?:вам\s+|тебе\s+)?(?:(?:вашу|твою|эту|вашей|твоей|этой|ваш|"
        r"твой)\s+)?"
        r"(?:\d|балл|оценк|зачет|работ|дз|домашк|проект|пят[её]рк|четв[её]рк|на\s+\d)"),
    _ci(r"\bоценил[аи]?\s+бы\b"),
    _ci(r"\bваш\w*\s+(?:оценк\w*|балл\w*)\s*(?:[:—-]|будет|составит)\s*\d"),
    _ci(r"\bоценк\w*\s*[:—-]\s*\d{1,2}\s*(?:/|из)\s*10\b"),
    _ci(r"(?:\bваш\w*|\bвам\b|\bтебе\b|\bтво\w*|\bработ\w*)[^.!?\n]{0,30}\b\d{1,2}\s*(?:/|из)\s*10\b"),
    _ci(r"\bтянет\s+на\s+\d|\bзаслуживает\s+(?:\d|оценк|высш)"),
    _ci(r"\b(?:вам|тебе)\s+постав(?:ят|ит|лю|им)\s+\d|\bпостав(?:ят|ит|лю|им)\s+(?:вам|тебе)\s+\d"),
    _ci(r"\bза\s+(?:вашу|твою|свою|эту)\s+(?:работу|домашку|дз|проект)[^.!?\n]{0,40}\b\d{1,2}\s*"
        r"(?:балл|из\s*10|/\s*10)"),
    _ci(r"\b(?:получите|получишь|наберете|наберешь)\b[^.!?\n]{0,25}\b\d{1,2}\s*(?:балл\w*|из\s*10|/\s*10)"
        r"[^.!?\n]{0,25}\bза\s+(?:вашу|твою|свою|эту)\s+(?:работу|домашку|дз|проект)"),
]
# «Работа принята» — нарушение, если только по данным LMS работа студента действительно не принята: та, что названа
# в предложении («ваша работа hw2 принята» — сверяется hw2), а если задание не названо — хоть одна.
_OUTPUT_ACCEPTED = [
    _ci(rf"\b(?:ваш\w*|тво\w*)\s+(?:работ\w*|дз|домашк\w*|проект\w*|задани\w*)\s+{_NOT}(?:уже\s+)?"
        r"(?:принят|засчитан|зачтен)"),
    _ci(rf"(?:^|[.!?:]\s+)(?:работ\w*|дз|домашк\w*)\s+{_NOT}(?:уже\s+)?(?:принят|засчитан|зачтен)\w*\s*[,.!—-]"),
]
_OUTPUT_SCHEDULE = [
    _ci(r"\b(?:я\s+)?(?:переношу|продлеваю|сдвигаю|перенесу|продлю|сдвину|отодвину)\s+"
        r"(?:вам|тебе|ваш|твой|дедлайн|срок|сдачу)"),
    _ci(r"\b(?:я|мы)\s+(?:уже\s+)?(?:перенес|перенесла|перенесли|продлил|продлила|продлили|сдвинул|сдвинула|"
        r"сдвинули|отодвинул|отодвинула)\b"),
    # «Готово: продлила срок hw1», «Продлила срок hw1 до 18.11» — без «я», глагол в начале предложения или после
    # «готово». «Преподаватель уже продлил срок», «Ирина Соколова продлила срок», «куратор не перенёс срок» —
    # у глагола своё подлежащее, это не обещание куратора.
    _ci(r"(?m)(?:^|[.!?:]\s*|\bготово\W+)(?:уже\s+)?(?:продлил|продлила|перенес|перенесла|сдвинул|сдвинула|"
        r"отодвинул|отодвинула)\s+(?:вам\s+|тебе\s+)?(?:срок|дедлайн|сдач)"),
    _ci(r"\b(?:одобряю|согласую|разрешаю)\s+(?:вам\s+|тебе\s+)?(?:перенос|продлени|отсрочк)"),
]
# «Ваш дедлайн перенесён на 18.11», «новый срок — 18.11» — нарушение, если в LMS у студента нет одобренного переноса
# (того задания, что названо в предложении, если оно названо).
_OUTPUT_MOVED = [
    _ci(r"\b(?:дедлайн|срок)\w*\s+(?:(?:hw[1-3]|final|сдачи|по\s+\S+|для\s+вас|вам)\s+){0,2}(?:уже\s+)?"
        r"(?:перенесен|продлен|сдвинут|отодвинут)\w*\s+(?:на|до)\b"),
    _ci(r"\bваш\w*\s+нов\w*\s+(?:срок|дедлайн)|\bнов\w*\s+(?:срок|дедлайн)\w*\s*[:—-]\s*\d"),
]

# Срок доработки в ответе модели. Даты решения преподавателя в LMS нет, поэтому конкретная дата или «осталось N»
# как срок доработки — всегда выдумка (чаще всего — исходный дедлайн задания). Проверяется предложение, где речь
# о доработке и где срок к ней привязан: «исправить и сдать до 15.11», «на доработку осталось 3 дня», «успеете
# исправить до дедлайна». Не нарушение: правило («есть 5 дней с решения преподавателя»), отрицание связи («исходный
# дедлайн к доработке не относится»), дата события («разбор в пт 13.11 — принесите вопросы по доработке») и сроки
# другого задания, которое не на доработке («hw2 сдайте до 29.11»).
_MONTHS = r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
_DATE = rf"(?:\d{{1,2}}[./]\d{{1,2}}(?:[./]\d{{2,4}})?|\d{{1,2}}\s+{_MONTHS})"
_REVISION_WORDS = _ci(r"доработ\w*|исправ\w*|пересда\w*|(?:сда\w*|отправ\w*|присла\w*|пришл\w*|загруз\w*)\s+"
                      r"(?:\w+\s+){0,2}?(?:снова|повторно|заново|еще\s+раз)")
_REVISION_NOT_LINKED = _ci(r"\bне\s+(?:относится|относятся|связан\w*|зависит|привязан\w*|считается\s+от|от\s+исходн)|"
                           r"\bа\s+не\b")
_REVISION_DATE = _ci(rf"\b(?:до|к|не\s+позже|не\s+позднее)\s+(?:(?:пн|вт|ср|чт|пт|сб|вс)\w*,?\s+)?{_DATE}")
_REVISION_LEFT = _ci(r"\b(?:осталось|остается|остаются|в\s+запасе)\s+(?:еще\s+|всего\s+)?\d+\s*(?:дн|день|час|сут)")
_REVISION_HAVE = _ci(r"\b(?:есть|дается|даётся|будет)\s+(?:еще\s+|у\s+вас\s+)?(\d+)\s*(?:дн|день|сут)")
_REVISION_TO_DEADLINE = _ci(r"\bдо\s+(?:этого|того|указанного|исходного|основного|общего|вашего|текущего)?\s*"
                            r"(?:срока|дедлайна)\b")
_ASSIGNMENT_ID = _ci(r"\b(hw\d|final)\b")
# Предложение кончается на «.», «!», «?» перед пробелом или концом строки; точка внутри даты («15.11.2026»)
# и числа («94.5») предложение не обрывает.
_SENTENCE = re.compile(r"(?:[^.!?\n]|[.!?]+(?=[^\s.!?]))*(?:[.!?]+(?=\s|$)|\n|$)")

# Длительность в рекомендации: «около 40 минут», «займёт примерно 2 часа», «(~25 мин)». В материалах она есть только
# у ресурсов каталога («~40 мин» из resources.csv), у разделов лекций и методичек её нет. Модель переносила число
# соседнего ресурса или из примера на раздел лекции — такое число вырезается (выходная проверка duration).
_DURATION = _ci(r"(?:[ \t]*(?P<open>\()[ \t]*|[ \t]*[,;—–][ \t]*|[ \t]+)?"
                r"(?:(?:это|на\s+это|на\s+него|на\s+неё|на\s+нее)\s+)?"
                r"(?:(?:занимает|займ[её]т|уйд[её]т|потребует(?:ся)?|понадобится)\s+)?"
                r"(?:около|примерно|приблизительно|порядка|~)\s*(?P<n>\d{1,3})(?:\s*[-–]\s*\d{1,3})?\s*"
                r"(?P<unit>минут\w*|мин\b\.?|час\w*)(?(open)\))")
_MINUTES = _ci(r"(\d{1,3})\s*(?:мин|минут)")
_CITE_MARK = re.compile(r"\[[\d,;\s–—-]+\]")
_WORD = re.compile(r"[^\W\d_]")


class _SearchUnavailable(Exception):
    """Поиск по базе знаний не ответил (эмбеддинги, Chroma)."""


@dataclass
class CuratorAnswer:
    text: str  # тело ответа и блок «Источники»
    body: str
    sources_block: str
    sources: list[ContextItem]
    intent: str
    module: int | None
    topic: str | None
    level: str
    level_source: str
    refusal: str = ""
    answered: bool = True
    cited: bool = False
    analysis_method: str = ""
    guard: str = ""
    journal_id: int | None = None
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    retrieval: RetrievalResult | None = None
    lms_used: bool = False
    student_id: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class _Draft:
    body: str
    context: Context
    refusal: str = ""
    answered: bool = True
    lms_used: bool = False
    empty_sources: str = ""
    retrieval: RetrievalResult | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    guard: str = ""
    from_model: bool = False  # тело ответа написала модель (а не шаблон)
    intent: str = ""  # сценарий, если код его уточнил (вопрос не о курсе, а поиск ничего не нашёл)


@dataclass
class _Query:
    """Всё, что нужно ответу модели по материалам: вопрос, фильтры поиска, уровень, студент."""

    intent: str
    question: str
    history: Sequence[Turn]
    module: int | None
    topic: str | None
    hard: SearchFilter  # выбор в интерфейсе — жёсткий фильтр поиска
    prefer: SearchFilter  # интерфейс + анализ запроса — предпочтение при ранжировании
    asked_types: tuple[str, ...]  # типы из интерфейса или анализа, без типов сценария по умолчанию
    level: str
    student: Student | None
    uses_student_data: bool
    module_overview: bool = False


def _mentions_assignment(text: str) -> bool:
    """Речь о задании, а не о событии с теми же словами: «разбор ДЗ», «защита итогового проекта» не считаются."""
    return bool(_ASSIGNMENT_WORDS.search(_EVENT_PHRASES.sub(" ", text)))


def _has_revision(student: Student | None) -> bool:
    return student is not None and "revision" in student.assignments.values()


def _sentence_at(text: str, pos: int) -> str:
    """Предложение текста, в котором стоит позиция pos."""
    for match in _SENTENCE.finditer(text):
        if match.start() <= pos < match.end():
            return match.group(0)
    return text


def _named_assignments(match: re.Match[str]) -> set[str]:
    """Задания, названные в самой фразе («ваша работа hw1 принята»), а если в ней нет — в её предложении."""
    named = {m.lower() for m in _ASSIGNMENT_ID.findall(match.group(0))}
    return named or {m.lower() for m in _ASSIGNMENT_ID.findall(_sentence_at(match.string, match.start()))}


def _unconfirmed(patterns: Sequence[re.Pattern[str]], text: str, confirmed: set[str], any_confirmed: bool) -> bool:
    """Есть ли фраза-факт («работа принята», «срок перенесён»), которую данные LMS студента не подтверждают.
    Если в предложении названо задание — факт сверяется с этим заданием (confirmed — задания, где он верен);
    если не названо — достаточно, что он верен хоть для одного задания студента (any_confirmed)."""
    for pattern in patterns:
        for match in pattern.finditer(text):
            named = _named_assignments(match)
            if not (named <= confirmed if named else any_confirmed):
                return True
    return False


def output_violation(text: str, student: Student | None = None) -> str | None:
    """Обещание оценки или переноса в ответе модели. student — студент, чьи данные LMS модель видела: если работа
    у него действительно принята или перенос одобрен, «ваша работа принята» и «ваш срок перенесён на …» — факт.
    Факт сверяется с заданием, названным в предложении: у s2 принята hw1, поэтому «Ваша работа hw2 принята» —
    нарушение, а «Ваша работа hw1 принята» — нет."""
    norm = text.replace("ё", "е").replace("Ё", "Е")
    accepted = {a.lower() for a, status in student.assignments.items() if status == "accepted"} if student else set()
    moved = {student.extension.assignment_id.lower()} if student is not None and student.extension else set()
    if any(p.search(norm) for p in _OUTPUT_GRADING):
        return "grading_request"
    if _unconfirmed(_OUTPUT_ACCEPTED, norm, accepted, bool(accepted)):
        return "grading_request"
    if any(p.search(norm) for p in _OUTPUT_SCHEDULE):
        return "schedule_change"
    if _unconfirmed(_OUTPUT_MOVED, norm, moved, bool(moved)):
        return "schedule_change"
    return None


def _revision_sentence_bad(sentence: str, revision_ids: set[str]) -> bool:
    norm = sentence.replace("ё", "е").replace("Ё", "Е")
    if not _REVISION_WORDS.search(norm) or _REVISION_NOT_LINKED.search(norm):
        return False
    named = {m.lower() for m in _ASSIGNMENT_ID.findall(norm)}
    if named and not named & revision_ids:
        return False  # срок другого задания, не того, что на доработке
    if _REVISION_DATE.search(norm) or _REVISION_LEFT.search(norm) or _REVISION_TO_DEADLINE.search(norm):
        return True
    have = _REVISION_HAVE.search(norm)
    return have is not None and int(have.group(1)) != REVISION_DAYS


def revision_deadline_violations(text: str, revision_ids: set[str]) -> list[str]:
    """Предложения ответа (вне кода), где срок доработки назван датой, остатком «осталось N» или исходным
    дедлайном. revision_ids — задания студента в статусе «на доработку»."""
    found: list[str] = []
    for part in _CODE_BLOCK.split(text):
        for match in _SENTENCE.finditer(part):
            sentence = match.group(0)
            if sentence.strip() and _revision_sentence_bad(sentence, revision_ids):
                found.append(sentence.strip())
    return found


def remove_sentences(text: str, sentences: Sequence[str]) -> str:
    """Текст без этих предложений (код не трогается); пустые абзацы схлопываются."""
    pieces: list[str] = []
    last = 0
    for code in _CODE_BLOCK.finditer(text):
        pieces.append(_drop(text[last:code.start()], sentences))
        pieces.append(code.group(0))
        last = code.end()
    pieces.append(_drop(text[last:], sentences))
    return "".join(pieces).strip()


def _drop(part: str, sentences: Sequence[str]) -> str:
    for sentence in sentences:
        part = re.sub(rf"[ \t]*{re.escape(sentence)}", "", part)
    part = re.sub(r"[ \t]+\n", "\n", part)
    return re.sub(r"\n{3,}", "\n\n", part)


def strip_unsupported_durations(text: str, ctx: Context) -> tuple[str, bool]:
    """Вырезать из ответа (вне кода) длительность «около N минут», которой нет в материалах. Опора — материалы,
    на которые ссылается та же строка ответа (пункт списка, абзац); строка без ссылок сверяется с ресурсами
    каталога. Число часов сравнивается в минутах («около 5 часов» = «~300 мин»). Если от предложения ничего,
    кроме длительности, не осталось («Займёт около 30 минут.»), оно убирается целиком."""
    texts = {item.n: item.text for item in ctx.items}
    resources = [item.text for item in ctx.items if item.source.kind == "resource"]
    changed = False

    def _allowed(line: str) -> set[int]:
        cited = citation_numbers(line, len(ctx))
        pool = [texts[n] for n in cited if n in texts] if cited else resources
        return {int(m) for t in pool for m in _MINUTES.findall(t)}

    def _fix_line(line: str) -> str:
        nonlocal changed
        if not _DURATION.search(line):
            return line
        allowed = _allowed(line)

        def _sub(match: re.Match[str]) -> str:
            n = int(match.group("n"))
            minutes = n * 60 if match.group("unit").lower().startswith("час") else n
            return match.group(0) if minutes in allowed else ""

        sentences = [m.group(0) for m in _SENTENCE.finditer(line) if m.group(0)]
        if "".join(sentences) != line:  # разбор на предложения не покрыл строку — не трогаем
            return line
        out: list[str] = []
        cut = False
        for sentence in sentences:
            fixed = _DURATION.sub(_sub, sentence)
            if fixed != sentence:
                cut = True
                if not _WORD.search(_CITE_MARK.sub("", fixed)):
                    continue  # от предложения осталась только ссылка и точка
            out.append(fixed)
        if not cut:
            return line
        changed = True
        return "".join(out).rstrip()

    def _fix(part: str) -> str:
        return "\n".join(_fix_line(line) for line in part.split("\n"))

    pieces: list[str] = []
    last = 0
    for code in _CODE_BLOCK.finditer(text):
        pieces.append(_fix(text[last:code.start()]))
        pieces.append(code.group(0))
        last = code.end()
    pieces.append(_fix(text[last:]))
    return "".join(pieces), changed


def today_text(settings: Settings) -> str:
    now = settings.now()
    return f"{WEEKDAYS_FULL[now.weekday()]}, {now:%d.%m.%Y %H:%M}"


def _excerpt(text: str, limit: int) -> str:
    """Начало раздела по границе абзаца или строки: цитата не обрывается на полуслове."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text.rfind("\n\n", 0, limit)
    if cut < limit // 3:
        cut = text.rfind("\n", 0, limit)
    if cut < limit // 3:
        cut = text.rfind(" ", 0, limit)
    return text[:cut].rstrip() + " …"


def _guide_excerpt(text: str, limit: int = GUIDE_SECTION_CHARS) -> str:
    """Раздел методички для контекста: целиком, если влезает; иначе блоки кода заменяются пометкой (суть шагов —
    в тексте), и если всё ещё длинно — обрезается по абзацу."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return _excerpt(_CODE_BLOCK.sub(CODE_PLACEHOLDER, text), limit)


def _usage(message: Any) -> tuple[int, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        return 0, 0
    data = cast(dict[str, Any], usage)
    return int(data.get("input_tokens", 0) or 0), int(data.get("output_tokens", 0) or 0)


def _content(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        parts = cast(list[Any], content)
        return "".join(p if isinstance(p, str) else str(cast(dict[str, Any], p).get("text", ""))
                       for p in parts)
    return str(content)


class Curator:
    def __init__(self, *, settings: Settings, analyzer: Analyzer, retriever: Retriever, llm: BaseChatModel,
                 lms: LMSClient, kb: KnowledgeBase, recommender: Recommender, fewshot: FewShotLibrary,
                 prompts: PromptLibrary, memory: DialogMemory, journal: Journal | None = None,
                 limits: DemoLimits | None = None, kb_version: str = ""):
        self.settings = settings
        self.analyzer = analyzer
        self.retriever = retriever
        self.llm = llm
        self.lms = lms
        self.kb = kb
        self.recommender = recommender
        self.fewshot = fewshot
        self.prompts = prompts
        self.memory = memory
        self.journal = journal
        self.limits = limits
        self.kb_version = kb_version

    # --- источники, которые нужны фиксированным сценариям

    def _faq(self, topic: str, prefer: Sequence[str] = ()) -> tuple[SourceRef, Section] | None:
        found = self.kb.faq(topic)
        if not found:
            return None
        for word in prefer:
            for doc, section in found:
                if word in section.title.lower():
                    return SourceRef("faq", doc.doc_id, doc.title, section.title), section
        doc, section = found[0]
        return SourceRef("faq", doc.doc_id, doc.title, section.title), section

    def _overview(self) -> SourceRef | None:
        doc = next((d for d in self.kb.documents if d.resource_type == "overview"), None)
        if doc is None:
            return None
        section = next((s for s in doc.sections if "куратор" in s.title.lower()), None)
        return SourceRef("overview", doc.doc_id, doc.title, section.title if section else "")

    def _lms_source(self, now_text: str) -> SourceRef:
        return SourceRef("lms", "lms", f"LMS: задания, сроки и расписание (данные на {now_text} МСК)")

    def _student(self, student_id: str | None) -> Student | None:
        try:
            return self.lms.student(student_id)
        except LMSError as exc:
            log.warning("LMS: студент недоступен: %s", exc)
            return None

    # --- фиксированные сценарии

    def _off_topic(self, question: str) -> _Draft:
        ctx = Context()
        overview = self._overview()
        ref = ctx.ref(overview, "") if overview else ""
        kind = small_talk_kind(question)
        topic = "" if kind else self._course_topic_line(ctx, question)
        body = self.prompts.block("refusal_off_topic", kind or "main", ref=ref, topic=topic)
        return _Draft(tidy(body), ctx, refusal="small_talk" if kind else "off_topic", answered=False)

    def _course_topic_line(self, ctx: Context, question: str) -> str:
        """Строка отказа о теме курса, которая упомянута в просьбе не по курсу («посоветуй фильм, устал от
        группировки»): тему находят правила (detect_topic), название и ссылка — лекция этой темы из базы."""
        slug = detect_topic(question, frozenset(self.kb.topics()))
        doc = next((d for d in self.kb.documents if d.topic == slug and d.resource_type == "lecture"), None) \
            if slug else None
        if doc is None:
            return ""
        ref = ctx.ref(SourceRef("lecture", doc.doc_id, doc.title, module=doc.module), "")
        return self.prompts.block("refusal_off_topic", "topic", title=doc.title, ref=ref)

    def _status_line(self, template: str, ctx: Context, assignment: Assignment, student: Student | None) -> str:
        """Строка шаблона со статусом задания у студента по данным LMS (у доработки — с правилом доработки)."""
        status = student.assignments.get(assignment.id, "") if student else ""
        if not status:
            return ""
        label = STATUS_LABELS.get(status, status)
        name = f"{assignment.id} «{assignment.title}»"
        fact = f"{assignment.id}: статус у студента — {label}"
        if status == "revision":
            fact += f"; {revision_rule()}"
        ref = ctx.ref(self._lms_source(fmt_dt(self.settings.now())), fact)
        if status == "revision":
            return self.prompts.block(template, "status_revision", assignment=name, ref=ref, rule=revision_rule())
        return self.prompts.block(template, "status", assignment=name, status=label, ref=ref)

    def _grading(self, module: int | None, student: Student | None = None) -> _Draft:
        p = self.prompts
        ctx = Context()
        now = self.settings.now()
        faq = self._faq("grading", prefer=("кто ставит", "оценива"))
        rule = ""
        if faq:
            source, section = faq
            excerpt = _excerpt(section.text, EXCERPT_CHARS)
            rule = p.block("refusal_grading", "rule", ref=ctx.ref(source, section.text), excerpt=quote(excerpt))
        status = ""
        if student is not None and module is not None:
            try:
                assignments = self.lms.assignments(module)
            except LMSError:
                assignments = []
            if assignments:
                status = self._status_line("refusal_grading", ctx, assignments[0], student)
        steps: list[str] = []
        guide = self.kb.guide(module) if module is not None else None
        criteria = next((s for s in guide.sections if s.kind == "criteria"), None) if guide else None
        if guide and criteria:
            source = SourceRef("guide", guide.doc_id, guide.title, criteria.title, module=guide.module)
            steps.append(p.block("refusal_grading", "criteria_step", ref=ctx.ref(source, criteria.text),
                                 criteria=_excerpt(criteria.text, CRITERIA_CHARS)))
        else:
            steps.append(p.block("refusal_grading", "no_module_step"))
        try:
            reviews = self.lms.upcoming(now, module, 1, types=("review",))
        except LMSError:
            reviews = []
        if reviews:
            event = reviews[0]
            facts = f"{fmt_dt(event.starts_at)} МСК — {event.title}"
            ref = ctx.ref(self._lms_source(fmt_dt(now)), facts)
            steps.append(p.block("refusal_grading", "review_step", when=f"{fmt_dt(event.starts_at)} МСК", ref=ref))
        steps.append(p.block("refusal_grading", "help_step"))
        body = p.block("refusal_grading", "main", rule=rule, status=status, steps="\n".join(steps))
        # answered=False: это отказ по правилам, а не ответ по материалам — в аналитике у него свой исход.
        return _Draft(tidy(body), ctx, refusal="grading_request", answered=False,
                      lms_used=bool(reviews) or bool(status))

    def _schedule(self, question: str, module: int | None, student: Student | None) -> _Draft:
        p = self.prompts
        ctx = Context()
        now = self.settings.now()
        lms_source = self._lms_source(fmt_dt(now))
        about_events = bool(_EVENT_WORDS.search(question)) and not _DEADLINE_WORDS.search(
            _EVENT_PHRASES.sub(" ", question))
        # Просьба сдвинуть событие — правило «куратор расписание не меняет» (FAQ о расписании, в нём же — про запись);
        # если такого вопроса в FAQ нет — правило о записях пропущенного вебинара.
        faq = (self._faq("schedule", prefer=("расписани",)) or self._faq("recordings", prefer=("пропустил",))) \
            if about_events else self._faq("extensions", prefer=("можно ли перенести",))
        rule = ""
        if faq:
            source, section = faq
            rule = p.block("refusal_schedule", "rule", ref=ctx.ref(source, section.text),
                           excerpt=quote(_excerpt(section.text, EXCERPT_CHARS)))
        facts: list[str] = []
        lms_used = False
        try:
            if about_events:
                events = self.lms.upcoming(now, module, 3)
                if events:
                    lines = [f"- {event_when(e)} МСК — {e.title}" for e in events]
                    facts.append(p.block("refusal_schedule", "events", ref=ctx.ref(lms_source, "\n".join(lines)),
                                         events="\n".join(lines)))
                    lms_used = True
                steps = p.block("refusal_schedule", "steps_events")
            else:
                assignments = self.lms.assignments(module) if module is not None else []
                if not assignments:  # модуль не назван — ближайшее задание, срок которого ещё не прошёл
                    upcoming = [a for a in self.lms.assignments()
                                if (student.deadline_for(a) if student else a.deadline) > now]
                    assignments = upcoming[:1] if module is None else []
                revision = bool(assignments and student
                                and student.assignments.get(assignments[0].id) == "revision")
                if assignments and revision:
                    # На доработке срок — 5 дней от решения преподавателя, а не дедлайн задания и не окно переноса.
                    facts.append(self._status_line("refusal_schedule", ctx, assignments[0], student))
                    lms_used = True
                    steps = p.block("refusal_schedule", "steps_revision")
                elif assignments:
                    st = deadline_status(assignments[0], now, student)
                    a = st.assignment
                    personal = " (ваш срок после одобренного переноса)" if st.personal else ""
                    values = {"assignment": f"{a.id} «{a.title}»{personal}", "deadline_at": fmt_dt(st.deadline),
                              "left": st.phrase}
                    fact_text = f"{a.id}: дедлайн {fmt_dt(st.deadline)} МСК{personal}, {st.phrase}"
                    ref = ctx.ref(lms_source, fact_text)
                    used_already = bool(student and student.extension_used)
                    if st.is_past or used_already:  # окно заявки не важно: срок прошёл или перенос уже был
                        facts.append(p.block("refusal_schedule", "deadline_plain", ref=ref, **values))
                    elif st.can_apply:
                        facts.append(p.block("refusal_schedule", "deadline_open", ref=ref,
                                             apply_until=fmt_dt(st.apply_until), **values))
                    else:
                        facts.append(p.block("refusal_schedule", "deadline_closed", ref=ref, **values))
                    facts.append(self._status_line("refusal_schedule", ctx, a, student))
                    lms_used = True
                    closed = st.is_past or not st.can_apply or used_already
                    steps = p.block("refusal_schedule", "steps_closed" if closed else "steps_open")
                else:
                    steps = p.block("refusal_schedule", "steps_unknown")
                if student:
                    used = "использован" if student.extension_used else "не использован"
                    ref = ctx.ref(lms_source, f"Перенос у студента {student.id}: {used}")
                    facts.append(p.block("refusal_schedule", "student_used" if student.extension_used
                                         else "student_free", ref=ref))
                    lms_used = True
        except LMSError as exc:
            log.warning("LMS недоступна: %s", exc)
            steps = p.block("refusal_schedule", "steps_unknown")
        main = "main_events" if about_events else "main"
        body = p.block("refusal_schedule", main, rule=rule, facts="\n\n".join(f for f in facts if f), steps=steps)
        return _Draft(tidy(body), ctx, refusal="schedule_change", answered=False, lms_used=lms_used)

    def _search_error(self) -> _Draft:
        p = self.prompts
        return _Draft(tidy(p.block("search_error", "main")), Context(), refusal="search_error", answered=False,
                      empty_sources=p.block("search_error", "sources"))

    def _no_answer(self) -> _Draft:
        ctx = Context()
        now = self.settings.now()
        nxt = ""
        try:
            webinars = self.lms.upcoming(now, None, 1, types=("webinar",))
        except LMSError:
            webinars = []
        if webinars:
            event = webinars[0]
            when = f"{fmt_dt(event.starts_at)} МСК"
            ref = ctx.ref(self._lms_source(fmt_dt(now)), f"{when} — {event.title}")
            nxt = self.prompts.block("no_answer", "next", when=when, ref=ref)
        body = self.prompts.block("no_answer", "main", next=nxt)
        return _Draft(tidy(body), ctx, refusal="no_context", answered=False,
                      empty_sources=self.prompts.block("no_answer", "sources"), lms_used=bool(webinars))

    # --- ответ модели по материалам

    def _revision_context(self, q: _Query, lms_used: bool) -> bool:
        """У студента задание на доработке, и о сроках речь может зайти: модель видит его данные LMS или вопрос
        сам о доработке («вернули на исправление…»)."""
        return _has_revision(q.student) and (lms_used or bool(_REVISION_WORDS.search(q.question.replace("ё", "е"))))

    def _materials(self, ctx: Context, q: _Query, queries: list[str]) -> tuple[RetrievalResult, bool]:
        now = self.settings.now()
        lms_used = False
        overview = self._is_module_overview(q)
        if q.intent in LMS_INTENTS or q.uses_student_data or overview:
            try:
                ctx.add(self._lms_source(fmt_dt(now)), self.lms.facts(now, q.module, q.student))
                lms_used = True
            except LMSError as exc:
                log.warning("LMS недоступна: %s", exc)
        if self._revision_context(q, lms_used):
            self._add_revision_rule(ctx)
        if q.intent in LATE_RULE_INTENTS and _LATE_WORDS.search(q.question.replace("ё", "е")):
            self._add_late_rule(ctx)
        if overview and q.module is not None:
            found = self.kb.module_overview(q.module)
            if found:
                doc, section = found
                ctx.add(SourceRef("overview", doc.doc_id, doc.title, section.title), section.text)
        if self._is_about_assignment(q) and q.module is not None:
            for doc, section in self.kb.guide_sections(q.module, ASSIGNMENT_SECTION_KINDS):
                ctx.add(SourceRef("guide", doc.doc_id, doc.title, section.title, module=doc.module),
                        _guide_excerpt(section.text))
        try:
            result = self.retriever.search(queries, q.hard, q.prefer)
        except Exception as exc:  # эмбеддинги OpenAI (сеть, 429, 5xx, таймаут) или Chroma
            log.warning("поиск по базе знаний не ответил: %s", type(exc).__name__)
            raise _SearchUnavailable(type(exc).__name__) from exc
        hits = result.hits[:HITS_BY_INTENT[q.intent]] if q.intent in HITS_BY_INTENT else result.hits
        add_hits(ctx, hits)
        rec_module, rec_topic = self._recommendation_scope(q.module, q.topic, result)
        k_res = RESOURCES_BY_INTENT.get(q.intent, 0)
        # Без модуля и темы каталог отдал бы общие ресурсы (установка среды) к любому вопросу — например, к просьбе
        # посоветовать сериал, и модель сослалась бы на них во фразе «в материалах этого нет».
        if k_res and (rec_module is not None or rec_topic is not None):
            for res in self.recommender.resources_for(q.level, rec_module, rec_topic, k_res):
                ctx.add(SourceRef("resource", res.id, res.title, url=res.url, module=res.module or None),
                        res.summary())
        k_sec = SECTIONS_BY_INTENT.get(q.intent, 0)
        if k_sec and result.hits:
            for ref in self.recommender.sections_for(q.level, rec_module, rec_topic, k_sec):
                ctx.add(SourceRef(ref.doc.resource_type, ref.doc.doc_id, ref.doc.title, ref.section.title,
                                  module=ref.doc.module), ref.section.text)
        return result, lms_used

    def _add_revision_rule(self, ctx: Context) -> None:
        """Разделы FAQ о доработке (тема grading, «доработк» в вопросе): правило 5 дней от решения преподавателя."""
        for doc, section in self.kb.faq("grading"):
            if "доработк" in section.title.lower():
                ctx.add(SourceRef("faq", doc.doc_id, doc.title, section.title), section.text)

    def _add_late_rule(self, ctx: Context) -> None:
        """Разделы FAQ о несданной в срок работе (тема deadlines, в заголовке — «не успею», «опоздал»): что делать,
        если не успеваешь или дедлайн прошёл. Доработка из них не следует — она бывает только у сданной работы."""
        for doc, section in self.kb.faq("deadlines"):
            if _LATE_WORDS.search(section.title.replace("ё", "е")):
                ctx.add(SourceRef("faq", doc.doc_id, doc.title, section.title), section.text)

    @staticmethod
    def _is_module_overview(q: _Query) -> bool:
        """Вопрос о модуле целиком: модуль известен и анализ так его пометил — или тема не определена
        (вопрос не о конкретной теме) в сценарии «материал» или «организационный»."""
        if q.module is None:
            return False
        return q.module_overview or (q.topic is None and q.intent in OVERVIEW_INTENTS)

    def _is_about_assignment(self, q: _Query) -> bool:
        """Вопрос о задании модуля (модуль известен, у него есть методичка):
        - material/difficulty: тема вопроса — тема методички, основной тип поиска — методичка (первый в списке
          анализа или выбран в интерфейсе) или в вопросе слова «ДЗ», «домашка», «hwN», «итоговый проект», «задание»;
        - organizational/deadline: спрашивают, что в задании сделать («что нужно сделать в hw2», «какие шаги
          в итоговом проекте»), и речь о задании — по теме или по словам вопроса;
        - в любом из этих сценариев — короткое уточнение «а что там нужно сделать?», если прошлая реплика
          диалога была о задании."""
        if q.module is None or q.intent not in (*ASSIGNMENT_INTENTS, *ASSIGNMENT_TASK_INTENTS):
            return False
        guide = self.kb.guide(q.module)
        if guide is None:
            return False
        task = bool(_ASSIGNMENT_TASK.search(q.question))
        on_topic = q.topic is not None and q.topic == guide.topic
        if q.intent in ASSIGNMENT_INTENTS:
            if on_topic or (q.asked_types and q.asked_types[0] == "guide") or _ASSIGNMENT_WORDS.search(q.question):
                return True
        elif task and (on_topic or _mentions_assignment(q.question)):
            return True
        if not (task and q.history and len(q.question.split()) <= FOLLOWUP_WORDS):
            return False
        previous = q.history[-1]
        return _mentions_assignment(f"{previous.question}\n{previous.answer}")

    @staticmethod
    def _recommendation_scope(module: int | None, topic: str | None,
                              result: RetrievalResult) -> tuple[int | None, str | None]:
        """Модуль и тема для каталога ресурсов. Если анализ их не определил (например, в вопросе слова двух модулей
        или работают запасные правила), — из лучшего найденного фрагмента лекции или методички: иначе на вопрос
        про seaborn каталог отдал бы общие ресурсы модуля 0 (установка среды). Если лучший фрагмент выдачи —
        лекция или методичка другого модуля (обогнала фрагменты модуля анализа, несмотря на прибавку), анализ ошибся
        с модулем — модуль и тема берутся из этого фрагмента."""
        top = result.hits[0] if result.hits else None
        top_module = top.metadata.get("module") if top is not None else None
        if (top is not None and module is not None and isinstance(top_module, int) and top_module >= 1
                and top_module != module and top.metadata.get("resource_type") in ("lecture", "guide")):
            lecture = top.metadata.get("resource_type") == "lecture"
            return top_module, (str(top.metadata.get("topic") or "") or None) if lecture else None
        if module is not None and topic is not None:
            return module, topic
        best = next((h for h in result.hits if h.metadata.get("resource_type") in ("lecture", "guide")
                     and isinstance(h.metadata.get("module"), int) and int(h.metadata["module"]) >= 1), None)
        if best is None:
            return module, topic
        if module is None:
            module = int(best.metadata["module"])
        if topic is None and best.metadata.get("resource_type") == "lecture":
            topic = str(best.metadata.get("topic") or "") or None
        return module, topic

    def _messages(self, *, intent: str, level: str, question: str, history: Sequence[Turn],
                  ctx: Context, tags: frozenset[str] = frozenset()) -> list[BaseMessage]:
        p = self.prompts
        system = [p.system(today_text(self.settings)), p.level(level), p.scenario(intent)]
        examples = self.fewshot.select(intent, level, tags=tags)
        if examples:
            shots = [p.block("context", "fewshot_header")]
            shots += [p.block("context", "fewshot_item", level=e.level, question=e.question,
                              bad_answer=e.bad_answer, why_bad=e.why_bad, good_answer=e.good_answer,
                              sources=e.sources_text())
                      for e in examples]
            system.append("\n\n".join(shots))
        materials = [p.block("context", "header")]
        materials += [p.block("context", "item", n=item.n, label=item.source.label(), text=item.text)
                      for item in ctx.items]
        messages: list[BaseMessage] = [SystemMessage("\n\n".join(system)), SystemMessage("\n\n".join(materials))]
        for turn in history:
            messages.append(HumanMessage(f"<вопрос>{turn.question}</вопрос>"))
            messages.append(AIMessage(turn.answer[:HISTORY_ANSWER_CHARS]))
        messages.append(HumanMessage(f"<вопрос>{question}</вопрос>"))
        return messages

    def _revision_fix(self, text: str, ctx: Context, student: Student) -> tuple[str, bool]:
        """Вырезать предложения, где срок доработки назван датой или «осталось N», и добавить правило доработки."""
        revision_ids = {a for a, status in student.assignments.items() if status == "revision"}
        bad = revision_deadline_violations(text, revision_ids)
        if not bad:
            return text, False
        item = next((i for i in ctx.items if i.source.kind == "faq" and "доработк" in i.source.section.lower()), None)
        item = item or next((i for i in ctx.items if i.source.kind == "lms"), None)
        ref = f" [{item.n}]" if item else ""
        fix = self.prompts.block("revision", "fix", assignment=", ".join(sorted(revision_ids)), ref=ref,
                                 rule=revision_rule())
        kept = remove_sentences(text, bad)
        return (f"{kept}\n\n{fix}" if kept else fix), True

    def _generate(self, q: _Query) -> _Draft:
        ctx = Context()
        queries = [q.question]
        if q.history and len(q.question.split()) <= FOLLOWUP_WORDS:
            # Уточнение «а подробнее?» само по себе в базе ничего не найдёт: ищем и вместе с прошлым вопросом,
            # у фрагмента — лучшая оценка из двух запросов.
            queries.append(f"{q.history[-1].question}\n{q.question}")
        try:
            result, lms_used = self._materials(ctx, q, queries)
        except _SearchUnavailable:
            return self._search_error()
        has_resources = any(i.source.kind == "resource" for i in ctx.items)
        if not result.hits and not lms_used and not (q.intent == "recommendation" and has_resources):
            if off_topic_marker(q.question) or leisure_request(q.question):
                # Поиск ничего не нашёл, и в вопросе признак темы не о курсе: это не «пробел в материалах».
                draft = self._off_topic(q.question)
                draft.intent = "off_topic"
            else:
                draft = self._no_answer()
            draft.retrieval = result
            return draft
        revision = self._revision_context(q, lms_used)
        messages = self._messages(intent=q.intent, level=q.level, question=q.question, history=q.history, ctx=ctx,
                                  tags=frozenset({"revision"}) if revision else frozenset())
        try:
            reply = self.llm.invoke(messages)
        except Exception as exc:  # модель не ответила — студент получает хотя бы список материалов
            log.warning("модель не ответила: %s", type(exc).__name__)
            items = "\n".join(f"- [{i.n}] {i.source.label()}" for i in ctx.items)
            body = self.prompts.block("llm_error", "main", items=items)
            return _Draft(tidy(body), ctx, refusal="llm_error", answered=False, retrieval=result, lms_used=lms_used)
        prompt_tokens, completion_tokens = _usage(reply)
        text = strip_model_sources(_content(reply).strip())
        violation = output_violation(text, q.student if lms_used else None)
        if violation:
            log.warning("ответ модели нарушил ограничение %s — заменён фиксированным сценарием", violation)
            draft = self._grading(q.module, q.student if lms_used else None) if violation == "grading_request" \
                else self._schedule(q.question, q.module, q.student)
            draft.refusal = "output_guard"
            draft.guard = f"output:{violation}"
            draft.prompt_tokens, draft.completion_tokens = prompt_tokens, completion_tokens
            draft.retrieval = result
            return draft
        guards: list[str] = []
        if revision and q.student is not None:
            text, fixed = self._revision_fix(text, ctx, q.student)
            if fixed:
                log.warning("ответ модели назвал срок доработки датой или остатком — заменено правилом")
                guards.append("output:revision_deadline")
        if q.intent in DURATION_CHECK_INTENTS:
            text, cut = strip_unsupported_durations(text, ctx)
            if cut:
                log.warning("ответ модели назвал длительность, которой нет в материалах, — вырезана")
                guards.append("output:duration")
        guard = ",".join(guards)
        body, _ = process_citations(text, len(ctx))
        body = body.strip()
        if not body:  # пустой ответ модели — как сбой: показываем материалы
            items = "\n".join(f"- [{i.n}] {i.source.label()}" for i in ctx.items)
            return _Draft(tidy(self.prompts.block("llm_error", "main", items=items)), ctx, refusal="llm_error",
                          answered=False, retrieval=result, lms_used=lms_used, prompt_tokens=prompt_tokens,
                          completion_tokens=completion_tokens)
        return _Draft(body, ctx, retrieval=result, lms_used=lms_used, prompt_tokens=prompt_tokens,
                      completion_tokens=completion_tokens, from_model=True, guard=guard)

    # --- уровень

    @staticmethod
    def choose_level(mode: str, hint: str | None, student: Student | None) -> tuple[str, str]:
        """Выбор в интерфейсе → подсказка из вопроса → уровень демо-студента → начинающий."""
        if mode in LEVELS:
            return mode, "manual"
        if hint in LEVELS:
            return str(hint), "hint"
        if student is not None:
            return student.level, "student"
        return "beginner", "default"

    # --- главный вход

    def ask(self, question: str, *, session_id: str, level_mode: str = "auto", module: int | None = None,
            resource_type: str | None = None, topic: str | None = None, student_id: str | None = None,
            source: str | None = None, client_id: str | None = None) -> CuratorAnswer:
        """client_id — ключ почасового лимита (посетитель, а не диалог): «Новый диалог» и перезагрузка страницы
        дают новую сессию, но не новый лимит. Без него лимит считается по session_id."""
        started = time.perf_counter()
        source = source or self.settings.default_source
        raw = clean_question(question)
        session = user_hash(self.settings.user_hash_salt, session_id)
        if self.limits is not None and source != "eval":
            decision = self.limits.check(client_id or session_id, raw)
            if not decision.ok:
                return self._limited(raw, decision.reason, dict(decision.values), session, source, started)
        text = redact(raw, keep=self.settings.public_contacts)
        history = self.memory.history(session_id)[-self.settings.history_turns:] if self.settings.history_turns \
            else []
        student = self._student(student_id)
        analysis_result: AnalysisResult = self.analyzer.analyze(text, history)
        a = analysis_result.analysis
        intent = a.intent
        eff_module = self.choose_module(module, a.module, explicit_module(text))
        eff_topic = topic or a.topic
        level, level_source = self.choose_level(level_mode, a.level_hint, student)
        # Выбор в интерфейсе — жёсткий фильтр; модуль из интерфейса, который перебил модуль, названный в вопросе,
        # в жёсткий фильтр не идёт. Анализ — предпочтение (типы сценария по умолчанию — если анализ их не назвал).
        hard = SearchFilter(module if module is not None and eff_module == module else None,
                            (resource_type,) if resource_type else (), topic)
        asked_types = (resource_type,) if resource_type else tuple(a.resource_types)
        prefer = SearchFilter(eff_module, tuple(a.resource_types) or INTENT_RESOURCE_TYPES[intent], a.topic)

        if intent == "off_topic":
            draft = self._off_topic(text)
        elif intent == "grading_request":
            draft = self._grading(eff_module, student)
        elif intent == "schedule_change":
            draft = self._schedule(text, eff_module, student)
        else:
            draft = self._generate(_Query(
                intent=intent, question=text, history=history, module=eff_module, topic=eff_topic, hard=hard,
                prefer=prefer, asked_types=asked_types, level=level, student=student,
                uses_student_data=a.uses_student_data, module_overview=a.module_overview))
        intent = draft.intent or intent

        cited = self._cited(draft)
        body, shown = renumber(draft.body, used_items(draft.context, cited))
        block = sources_block(shown, draft.empty_sources)
        model_cited = draft.from_model and bool(cited)
        prompt_tokens = analysis_result.prompt_tokens + draft.prompt_tokens
        completion_tokens = analysis_result.completion_tokens + draft.completion_tokens
        answer = CuratorAnswer(
            text=f"{body}\n\n{block}", body=body, sources_block=block, sources=shown, intent=intent,
            module=eff_module, topic=eff_topic, level=level, level_source=level_source, refusal=draft.refusal,
            answered=draft.answered, cited=model_cited, analysis_method=analysis_result.method,
            guard=",".join(g for g in (analysis_result.guard, draft.guard) if g),
            latency_ms=int((time.perf_counter() - started) * 1000), prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, cost_usd=self.settings.cost_usd(prompt_tokens, completion_tokens),
            retrieval=draft.retrieval, lms_used=draft.lms_used, student_id=student.id if student else None,
        )
        if analysis_result.error:
            answer.notes.append(f"анализ правилами: модель не ответила ({analysis_result.error})")
        self.memory.add(session_id, Turn(text, body, intent, eff_module))
        retrieval = draft.retrieval
        answer.journal_id = self._log(Record(
            session=session, source=source, question=text, intent=intent, module=eff_module,
            topic=eff_topic or "", level=level, level_source=level_source, analysis_method=analysis_result.method,
            guard=answer.guard, sources=[i.source.journal_entry() for i in shown], cited=model_cited,
            answered=draft.answered, refusal=draft.refusal,
            top_score=retrieval.top_score if retrieval else None,
            filter=retrieval.filter_used.describe() if retrieval else "",
            relaxed=retrieval.relaxed if retrieval else False, latency_ms=answer.latency_ms,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost_usd=answer.cost_usd,
            model=self.settings.chat_model, kb_version=self.kb_version,
            student_used=draft.lms_used and student is not None,
        ))
        return answer

    @staticmethod
    def choose_module(ui_module: int | None, analysis_module: int | None, explicit: int | None) -> int | None:
        """Модуль для поиска и сценария. Модуль не выбран в интерфейсе — берётся модуль из анализа запроса.
        Выбран — берётся он, если только в вопросе модуль не назван прямо («hw3», «модуль 2», «L2.2»): тогда
        названный. Модуль, который анализ угадал по словам темы, выбор в интерфейсе не перебивает."""
        if ui_module is None:
            return analysis_module
        return explicit if explicit is not None else ui_module

    @staticmethod
    def _cited(draft: _Draft) -> list[int]:
        """Номера, на которые ссылается тело ответа (у шаблонов — все материалы, у модели — что она выбрала)."""
        return citation_numbers(draft.body, len(draft.context))

    def _limited(self, raw: str, reason: str, values: dict[str, object], session: str, source: str,
                 started: float) -> CuratorAnswer:
        body = self.prompts.block("refusal_limits", reason, **values)
        block = sources_block([], "правила демо-версии куратора (лимиты вопросов).")
        refusal = f"limit_{reason}"
        latency = int((time.perf_counter() - started) * 1000)
        # Сначала обезличить, потом обрезать: почта на границе обрезки иначе ушла бы в журнал наполовину.
        question = redact(raw[:LIMITED_SCAN_CHARS], keep=self.settings.public_contacts)[:LIMITED_LOG_CHARS]
        answer = CuratorAnswer(text=f"{body}\n\n{block}", body=body, sources_block=block, sources=[],
                               intent="", module=None, topic=None, level="", level_source="", refusal=refusal,
                               answered=False, latency_ms=latency)
        answer.journal_id = self._log(Record(session=session, source=source, question=question, refusal=refusal,
                                             latency_ms=latency, model=self.settings.chat_model,
                                             kb_version=self.kb_version))
        return answer

    def rate(self, journal_id: int | None, value: int | None) -> None:
        if self.journal is not None and journal_id:
            self.journal.rate(journal_id, value)

    def reset(self, session_id: str) -> None:
        self.memory.reset(session_id)

    def _log(self, record: Record) -> int | None:
        if self.journal is None:
            return None
        try:
            return self.journal.write(record)
        except Exception:  # сбой журнала не должен оставлять студента без ответа
            log.exception("не удалось записать в журнал")
            return None
