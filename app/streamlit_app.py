"""Чат с ИИ-куратором курса «Python для анализа данных» и меню страниц (Аналитика, LMS, О проекте).

    streamlit run app/streamlit_app.py

Слева — уровень ответа (авто / начинающий / продвинутый), модуль, тип материалов и тема для поиска, демо-студент
из заглушки LMS. Под ответом — источники (блок развёрнут) и оценка 👍/👎 (пишется в журнал).

Вопрос сначала записывается в session_state как задание, ответ — туда же сразу после ask(), до любого обращения
к st.*: если во время генерации посетитель что-то нажал и Streamlit перезапустил скрипт, ответ не теряется,
а незаконченное задание доделывается в следующем прогоне. Сбой куратора посетитель видит как короткое сообщение,
а не как traceback.
"""
from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Any  # noqa: E402

import streamlit as st  # noqa: E402

from app import common  # noqa: E402
from curator.analytics import INTENT_LABELS, REFUSAL_LABELS  # noqa: E402
from curator.config import MODULE_TITLES  # noqa: E402
from curator.curator import CuratorAnswer  # noqa: E402
from curator.lms import LEVEL_LABELS, STATUS_LABELS, LMSError  # noqa: E402

EXAMPLES = (
    "Когда дедлайн второго ДЗ и сколько осталось?",
    "Не понимаю, чем loc отличается от iloc",
    "Что почитать по визуализации новичку?",
    "Поставьте мне оценку за hw1, я всё сделал",
    "Можно перенести дедлайн hw1 на неделю?",
    "Как получить сертификат?",
)
FIXED = ("grading_request", "schedule_change", "off_topic")
# Отказы, у которых тело ответа собирает код из шаблона, хотя сценарий — ответ модели по материалам.
TEMPLATE_REFUSALS = ("no_context", "output_guard", "search_error", "llm_error")
FAILURE_TEXT = ("Не получилось ответить: сервис куратора сейчас не отвечает. Повторите вопрос через минуту — "
                "или напишите преподавателю в чат «Помощь» в LMS.")

log = logging.getLogger("curator.app")


def _meta(answer: CuratorAnswer) -> str:
    if not answer.intent:
        return f"Ограничение демо: {REFUSAL_LABELS.get(answer.refusal, answer.refusal)}"
    parts = [f"Сценарий: {INTENT_LABELS.get(answer.intent, answer.intent)}"]
    if answer.level:
        parts.append(f"уровень: {common.LEVEL_LABELS.get(answer.level, answer.level)} "
                     f"({common.LEVEL_SOURCE_LABELS.get(answer.level_source, answer.level_source)})")
    if answer.module is not None:
        parts.append(f"модуль {answer.module}")
    if answer.intent in FIXED or answer.refusal in TEMPLATE_REFUSALS:
        parts.append("ответ собран кодом по шаблону, модель его не писала")
    if answer.analysis_method == "rules":
        parts.append("разбор вопроса — правилами (модель не ответила)")
    return " · ".join(parts)


def _sources_list(answer: CuratorAnswer) -> str:
    """Тело блока «Источники» без заголовка — заголовок у раскрывающегося блока."""
    return answer.sources_block.split("\n", 2)[-1].strip()


def _on_feedback(index: int) -> None:
    message = st.session_state["messages"][index]
    value = st.session_state.get(f"fb-{index}")
    rating = None if value is None else (1 if value == 1 else -1)
    services = common.services()
    if services is not None:
        services.curator.rate(message.get("journal_id"), rating)
    message["rating"] = rating


def _render(index: int, message: dict[str, Any]) -> None:
    with st.chat_message(message["role"], avatar="🎓" if message["role"] == "assistant" else None):
        st.markdown(message["content"])
        if message["role"] != "assistant":
            return
        if message.get("sources"):
            # Развёрнут: куратор всегда ссылается на источник, и расшифровка [1], [2] видна без клика.
            with st.expander("Источники", expanded=True):
                st.markdown(message["sources"])
        st.caption(message["meta"])
        if message.get("journal_id"):
            st.feedback("thumbs", key=f"fb-{index}", on_change=_on_feedback, args=(index,))


def _answer_message(services: Any, job: dict[str, Any]) -> dict[str, Any]:
    """Ответ куратора на задание чата — сообщение для session_state; сбой — короткий текст вместо traceback."""
    try:
        answer = services.curator.ask(
            job["question"], session_id=job["session_id"], level_mode=job["level_mode"], module=job["module"],
            resource_type=job["resource_type"], topic=job["topic"], student_id=job["student_id"],
            source=job["source"], client_id=job["client_id"])
    except Exception:  # куратор сам обрабатывает сбои сервисов; это — на непредвиденный случай
        log.exception("куратор не ответил")
        return {"role": "assistant", "content": FAILURE_TEXT, "sources": "", "meta": "Ошибка сервиса",
                "journal_id": None, "intent": "", "job": job["id"]}
    return {"role": "assistant", "content": answer.body, "sources": _sources_list(answer), "meta": _meta(answer),
            "journal_id": answer.journal_id, "intent": answer.intent, "job": job["id"]}


def _process_jobs(services: Any) -> None:
    """Задания по одному; ответ встаёт сразу за своим вопросом. Rerun, пришедший во время генерации (посетитель
    что-то нажал), Streamlit выполняет в ближайшей «точке проверки» — это любой вызов st.*, в том числе чтение
    st.session_state[...]. Поэтому списки берутся из session_state до ask(), а после ask() меняются на месте без
    обращений к st.*: ответ сохранится раньше, чем скрипт прервётся."""
    jobs: list[dict[str, Any]] = st.session_state["jobs"]
    messages: list[dict[str, Any]] = st.session_state["messages"]
    with st.spinner("Куратор ищет в материалах курса…"):
        while jobs:
            job = jobs[0]
            reply = _answer_message(services, job)
            at = next((i for i, m in enumerate(messages) if m.get("job") == job["id"] and m["role"] == "user"),
                      len(messages) - 1)
            messages.insert(at + 1, reply)
            jobs.pop(0)


def chat() -> None:
    st.session_state.setdefault("session_id", uuid.uuid4().hex)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("jobs", [])
    st.title("ИИ-куратор курса «Python для анализа данных»")
    st.caption("Учебный проект: вымышленная онлайн-школа «Академия Данных», поток «Ноябрь-2026». "
               "Все данные — вымышленные. Оценки и сроки куратор не меняет — это делает преподаватель.")

    services = common.services()
    if services is None:
        st.info("Страницы «Аналитика», «LMS» и «О проекте» работают и без ключа OpenAI — они в меню слева.")
        st.stop()
        return

    if not services.index.ok:
        st.warning(services.index.message)
    for problem in services.problems[:5]:
        st.warning(f"Данные: {problem}")

    with st.sidebar:
        st.header("Настройки ответа")
        level_label = st.radio("Уровень ответа", list(common.LEVEL_OPTIONS), horizontal=True,
                               help="«Авто» — по вопросу, затем по уровню демо-студента, иначе — начинающий.")
        module = st.selectbox("Модуль", [None, *sorted(MODULE_TITLES)], format_func=common.module_label)
        type_label = st.selectbox("Тип материалов", list(common.TYPE_OPTIONS))
        topics = {slug: info for slug, info in services.kb.topics().items()
                  if module is None or info.module in (module, 0)}
        ordered: list[str] = sorted(topics, key=lambda slug: (topics[slug].module == 0, topics[slug].label))
        topic = st.selectbox("Тема", [None, *ordered],  # темы лекций и методичек, затем — темы FAQ
                             format_func=lambda s: "Любая" if s is None else topics[s].label)
        try:
            students = services.lms.students()
        except LMSError as exc:
            students = []
            st.warning(f"LMS: {exc}")
        by_id = {s.id: s for s in students}
        student_id = st.selectbox(
            "Демо-студент", [None, *by_id],
            format_func=lambda sid: "Без студента" if sid is None else
            f"{by_id[sid].name} — {LEVEL_LABELS.get(by_id[sid].level, by_id[sid].level)}",
            help="Вымышленные студенты из заглушки LMS: их статусы ДЗ и перенос учитываются в ответах о сроках.")
        if student_id:
            student = by_id[student_id]
            statuses = ", ".join(f"{k}: {STATUS_LABELS.get(v, v)}" for k, v in student.assignments.items())
            st.caption(f"Статусы ДЗ — {statuses}. Перенос: {'использован' if student.extension_used else 'нет'}.")
        if st.button("Новый диалог", width="stretch"):
            # Лимит вопросов в час остаётся прежним: он считается по посетителю (limit_key), а не по диалогу.
            services.curator.reset(st.session_state["session_id"])
            st.session_state["session_id"] = uuid.uuid4().hex
            st.session_state["messages"] = []
            st.session_state["jobs"] = []
            st.rerun()
        s = services.settings
        st.caption(f"Демо: вопрос до {s.max_question_chars} знаков, до {s.rate_per_hour} вопросов в час. "
                   f"Сейчас по Москве: {s.now():%d.%m.%Y %H:%M}.")

    for i, msg in enumerate(st.session_state["messages"]):
        _render(i, msg)

    pending: tuple[str, str] | None = None
    if not st.session_state["messages"]:
        st.markdown("**Попробуйте спросить:**")
        columns = st.columns(2)
        for n, example in enumerate(EXAMPLES):
            if columns[n % 2].button(example, key=f"ex-{n}", width="stretch"):
                pending = (example, "demo")

    # max_chars — тот же лимит длины, что у куратора: браузер не даст ввести больше, а Streamlit обрежет и ввод,
    # присланный в обход браузера, ещё до session_state — огромная вставка не осядет в памяти сеанса.
    typed = st.chat_input("Спросите о лекциях, заданиях, сроках или правилах курса…",
                          max_chars=services.settings.max_question_chars)
    if typed:
        pending = (typed, services.settings.default_source)

    if pending:
        question, source = pending
        job = {"id": uuid.uuid4().hex, "question": question, "source": source,
               "session_id": st.session_state["session_id"], "level_mode": common.LEVEL_OPTIONS[level_label],
               "module": module, "resource_type": common.TYPE_OPTIONS[type_label], "topic": topic,
               "student_id": student_id, "client_id": common.limit_key(services.settings.user_hash_salt)}
        # Вопрос и задание записываются вместе, без обращений к st.* между ними (даже чтение session_state —
        # точка, где Streamlit может прервать скрипт ради rerun), и только потом рисуются: rerun посреди отрисовки
        # не потеряет вопрос, задание доделается в следующем прогоне.
        messages: list[dict[str, Any]] = st.session_state["messages"]
        jobs: list[dict[str, Any]] = st.session_state["jobs"]
        messages.append({"role": "user", "content": question, "job": job["id"]})
        jobs.append(job)
        _render(len(messages) - 1, messages[-1])

    if st.session_state["jobs"]:
        _process_jobs(services)
        st.rerun()  # перерисовать диалог целиком: без кнопок-примеров и с оценкой у нового ответа


st.set_page_config(page_title="ИИ-куратор курса", page_icon="🎓", layout="wide")
navigation = st.navigation([
    st.Page(chat, title="Чат с куратором", icon="🎓", default=True),
    st.Page("views/1_Аналитика.py", title="Аналитика", icon="📊"),
    st.Page("views/2_LMS.py", title="LMS", icon="🗓️"),
    st.Page("views/3_О_проекте.py", title="О проекте", icon="ℹ️"),
])
navigation.run()
