"""Streamlit-интерфейс через streamlit.testing.v1.AppTest: настоящая сборка сервисов (curator.factory),
но модель, анализ запроса и эмбеддинги — фейки; данные — фикстуры, журнал — во временной папке."""
from __future__ import annotations

import importlib
import re
import tomllib
from collections.abc import Iterator
from datetime import timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import streamlit as st
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from streamlit.runtime.scriptrunner_utils.script_requests import RerunData
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from streamlit.testing.v1 import AppTest

from app import common
from curator import analytics, factory, lms, retriever
from curator.config import load_settings
from curator.curator import Curator
from curator.journal import Journal, Record, utc_iso
from tests.conftest import EMBED_MODEL, FIXTURES, MIN_SIMILARITY, NOW, ROOT
from tests.fakes import FakeChat, FakeRouterLLM, TrigramEmbeddings

APP = ROOT / "app" / "streamlit_app.py"
PAGES = ROOT / "app" / "views"
TIMEOUT = 90


@pytest.fixture
def app_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
            fixture_index: tuple[Path, dict[str, Any]]) -> Iterator[dict[str, Any]]:
    chat = FakeChat(reply="Дедлайн hw1 — вс 15.11.2026 23:59 МСК, осталось 3 дня [1].")
    journal_path = tmp_path / "journal.db"
    for key, value in {"CURATOR_DATA_DIR": str(FIXTURES), "CURATOR_INDEX_DIR": str(fixture_index[0]),
                       "CURATOR_JOURNAL": str(journal_path), "CURATOR_NOW": NOW.isoformat(),
                       "EMBED_MODEL": EMBED_MODEL, "MIN_SIMILARITY": str(MIN_SIMILARITY)}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(factory, "make_chat", lambda settings: chat)
    monkeypatch.setattr(factory, "make_router_llm", lambda settings: FakeRouterLLM(fail=True))  # → правила
    monkeypatch.setattr(factory, "make_embeddings", lambda settings: TrigramEmbeddings())
    monkeypatch.setattr(analytics, "wall_now", lambda: NOW)  # период дашборда — от «настоящих» часов
    st.cache_resource.clear()
    yield {"chat": chat, "journal": journal_path}
    st.cache_resource.clear()


def selectbox(at: AppTest, label: str) -> Any:
    return next(s for s in at.selectbox if s.label == label)


def test_chat_answers_with_sources_and_feedback(app_env: dict[str, Any]):
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert at.title[0].value.startswith("ИИ-куратор курса")
    assert len([b for b in at.button if b.key and str(b.key).startswith("ex-")]) == 6

    selectbox(at, "Демо-студент").set_value("s2")
    at.chat_input[0].set_value("Когда дедлайн hw1 и сколько мне осталось?").run()
    assert not at.exception
    assert len(at.chat_message) == 2
    assert "осталось 3 дня" in at.chat_message[1].markdown[0].value
    assert [e.label for e in at.expander] == ["Источники"]
    assert at.expander[0].proto.expanded  # блок источников раскрыт сразу
    assert "LMS: задания, сроки и расписание" in at.expander[0].markdown[0].value
    assert any("Сценарий: Дедлайны" in c.value for c in at.caption)
    prompt = app_env["chat"].prompt_text()
    assert "Уровень студента: продвинутый." in prompt and "Студент s2" in prompt

    at.feedback[0].set_value(1).run()
    assert not at.exception
    rows = Journal(app_env["journal"]).fetch()
    assert rows[-1]["rating"] == 1 and rows[-1]["source"] == "live" and rows[-1]["intent"] == "deadline"


def test_navigation_menu_reaches_every_page(app_env: dict[str, Any]):
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    at.switch_page("views/2_LMS.py").run()
    assert not at.exception and at.title[0].value == "Данные LMS (заглушка)"
    at.switch_page("views/3_О_проекте.py").run()
    assert not at.exception and at.title[0].value == "О проекте"


def test_example_button_goes_to_fixed_scenario_as_demo(app_env: dict[str, Any]):
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    at.button(key="ex-3").click().run()  # «Поставьте мне оценку за hw1…»
    assert not at.exception
    assert "Поставить или предсказать оценку я не могу" in at.chat_message[1].markdown[0].value
    assert app_env["chat"].calls == []
    rows = Journal(app_env["journal"]).fetch()
    assert rows[-1]["source"] == "demo" and rows[-1]["refusal"] == "grading_request"


def test_new_dialog_clears_history(app_env: dict[str, Any]):
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    at.chat_input[0].set_value("Как получить сертификат?").run()
    assert len(at.chat_message) == 2
    next(b for b in at.button if b.label == "Новый диалог").click().run()
    assert not at.exception and len(at.chat_message) == 0


def test_without_openai_key_chat_explains(app_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert "OPENAI_API_KEY" in at.error[0].value


def seed_journal(path: Path) -> None:
    journal = Journal(path)
    base = NOW.astimezone(timezone.utc)
    journal.write(Record(session="a", source="live", question="секретный живой вопрос", intent="material",
                         module=3, topic="matplotlib-basics", refusal="no_context", latency_ms=800,
                         ts=utc_iso(base - timedelta(hours=1))))
    row = journal.write(Record(session="b", source="eval", question="eval: что такое маска", intent="difficulty",
                               module=2, topic="pandas-filtering", answered=True, latency_ms=2100,
                               sources=[{"kind": "lecture", "id": "L2.2", "section": "Булевы маски"}],
                               prompt_tokens=1200, completion_tokens=300, cost_usd=0.001,
                               ts=utc_iso(base - timedelta(days=1))))
    journal.rate(row, 1)
    journal.write(Record(session="c", source="demo", question="Поставьте оценку", intent="grading_request",
                         module=1, refusal="grading_request", answered=False, latency_ms=40,
                         ts=utc_iso(base - timedelta(days=2))))


def test_analytics_page_shows_aggregates_but_not_live_texts(app_env: dict[str, Any]):
    seed_journal(app_env["journal"])
    at = AppTest.from_file(str(PAGES / "1_Аналитика.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Вопросов"] == "3" and metrics["Нет в материалах"] == "1"
    assert metrics["Ответ по материалам"] == "33 %"  # отказ «оценку ставит преподаватель» — не ответ
    assert len(at.get("plotly_chart")) >= 5
    tables = "\n".join(df.value.to_string() for df in at.dataframe)
    assert "eval: что такое маска" in tables
    assert "секретный живой вопрос" not in tables
    assert "тексты живых посетителей скрыты" in tables
    questions = at.dataframe[-1].value  # у записи demo нет темы и оценки
    assert not questions.isna().to_numpy().any()
    assert "None" not in questions.astype(str).to_numpy()
    assert len(at.get("download_button")) == 1


def test_analytics_page_empty_journal(app_env: dict[str, Any]):
    at = AppTest.from_file(str(PAGES / "1_Аналитика.py"), default_timeout=TIMEOUT).run()
    assert not at.exception and "записей нет" in at.info[0].value


def test_analytics_page_with_only_limit_refusals(app_env: dict[str, Any]):
    """За период в журнале только отказы по лимитам демо (один слишком длинный вопрос на свежем демо): графикам
    строить нечего — страница показывает отказы, а не «This app has encountered an error»."""
    journal = Journal(app_env["journal"])
    base = NOW.astimezone(timezone.utc)
    for hours, refusal in ((1, "limit_too_long"), (2, "limit_per_hour")):
        journal.write(Record(session="a", source="live", question="x" * 300, refusal=refusal, latency_ms=3,
                             ts=utc_iso(base - timedelta(hours=hours))))
    at = AppTest.from_file(str(PAGES / "1_Аналитика.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert "только отказы по лимитам демо" in at.info[0].value
    table = at.dataframe[0].value.to_string()
    assert "Лимит: длинный вопрос" in table and "Лимит: вопросов в час" in table
    assert not at.get("plotly_chart")


def test_lms_page_tables_and_downloads(app_env: dict[str, Any]):
    at = AppTest.from_file(str(PAGES / "2_LMS.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert len(at.dataframe) == 3
    assignments = at.dataframe[0].value
    assert list(assignments["осталось"]) == ["осталось 3 дня 9 часов", "осталось 17 дней 9 часов"]
    assert len(at.get("download_button")) == 6


def test_about_page(app_env: dict[str, Any]):
    at = AppTest.from_file(str(PAGES / "3_О_проекте.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert [s.value for s in at.subheader] == ["Как устроен ответ", "Состояние", "Обновление базы знаний", "Ссылки"]
    assert "Индекс актуален" in at.success[0].value
    text = "\n".join(m.value for m in at.markdown)
    assert GOOGLE_DOC in text and "python -m curator.ingest" in text
    for path in DOC_PATHS:
        assert f"/blob/main/{path})" in text, path


def test_about_page_describes_search_as_in_retriever(app_env: dict[str, Any]):
    """Описание поиска совпадает с curator/retriever.py: выбор слева — жёсткий фильтр с ослаблением, анализ
    запроса — приоритет (прибавки из кода), эмбеддинг вопроса — один раз; числа берутся из кода и настроек."""
    at = AppTest.from_file(str(PAGES / "3_О_проекте.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    search = next(line for line in text.splitlines() if "**Поиск (RAG).**" in line)
    settings = load_settings()

    def bonus(value: float) -> str:
        return f"+{value:.2f}".replace(".", ",")

    assert "настроек слева — жёсткий фильтр" in search
    assert f"меньше {settings.min_filtered_hits} фрагментов, он ослабляется" in search
    assert "из анализа запроса — не фильтр, а приоритет" in search
    assert bonus(retriever.PREFER_BONUS) in search and bonus(retriever.MODULE_BONUS) in search
    assert "Эмбеддинг вопроса считается один раз" in search
    assert "Фильтр по метаданным из анализа" not in text  # прежнее описание: анализ как стена-фильтр
    assert f"SQLite, {settings.retention_days} дней" in text


def test_lms_page_commands_match_stub_and_deploy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Команды на вкладке «API заглушки»: модуль uvicorn существует, адрес и порт — как у службы заглушки и LMS_URL,
    команда выгрузки принимает свои аргументы."""
    source = (PAGES / "2_LMS.py").read_text(encoding="utf-8")
    run = re.search(r"uvicorn ([\w.]+):(\w+) --host (\S+) --port (\d+)", source)
    assert run is not None
    module, attr, host, port = run.groups()
    assert hasattr(importlib.import_module(module), attr)
    unit = (ROOT / "deploy" / "ai-curator-lms.service").read_text(encoding="utf-8")
    assert f"uvicorn {module}:{attr} --host {host} --port {port} " in unit
    assert f"LMS_URL=http://{host}:{port} streamlit run app/streamlit_app.py" in source
    assert (ROOT / "app" / "streamlit_app.py").is_file()
    export = re.search(r"python -m curator\.lms (export --format \w+)", source)
    assert export is not None
    monkeypatch.setenv("CURATOR_DATA_DIR", str(FIXTURES))
    monkeypatch.delenv("LMS_URL", raising=False)
    assert lms.main([*export.group(1).split(), "--out", str(tmp_path)]) == 0
    assert {p.suffix for p in tmp_path.iterdir()} == {".json", ".csv"}


def test_pages_live_outside_streamlit_autodiscovered_dir():
    """Страницы — в app/views, а не в app/pages: каталог pages/ рядом с главным скриптом Streamlit подхватывает сам,
    и по прямой ссылке (/Аналитика, /LMS) запускает страницу в обход st.navigation — в меню тогда «streamlit app»
    вместо «Чат с куратором» и без значков. Проверено на демо 12.11.2026."""
    assert not (ROOT / "app" / "pages").exists()
    source = APP.read_text(encoding="utf-8")
    pages = re.findall(r'st\.Page\("([^"]+)"', source)
    assert pages and all(p.startswith("views/") and (ROOT / "app" / p).is_file() for p in pages)


GOOGLE_DOC = "https://docs.google.com/document/d/1W8aHKrNGa0gk_KIkr4i4ijbSL2v7D042FHnHj1NocvU/edit"
DOC_PATHS = ("docs/architecture.md", "docs/kb-update.md", "docs/analytics.md", "docs/lms-stub.md", "docs/diagram.svg")


def test_about_page_doc_links_point_to_existing_files():
    """Каждый путь docs/… со страницы «О проекте» существует в репозитории (на GitHub не будет ссылок в 404).
    Документацию пишут отдельно: пока папки docs нет, проверяется только список путей."""
    source = (PAGES / "3_О_проекте.py").read_text(encoding="utf-8")
    paths = re.findall(r'"(docs/[^"]+)"', source)
    assert set(DOC_PATHS) <= set(paths)
    docs = ROOT / "docs"
    if docs.is_dir():
        missing = [p for p in paths if not (ROOT / p).is_file()]
        assert not missing, f"на странице «О проекте» ссылки на несуществующие файлы: {missing}"


class ChatWithInterruption(FakeChat):
    """Пока модель «думает», посетитель меняет настройку слева или шлёт второй вопрос: Streamlit ставит rerun."""

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        ctx = get_script_run_ctx()
        if ctx is not None and ctx.script_requests is not None and not self.calls:
            ctx.script_requests.request_rerun(RerunData(page_script_hash=ctx.page_script_hash))
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_answer_survives_rerun_during_generation(app_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    chat = ChatWithInterruption(reply="Дедлайн hw1 — вс 15.11.2026 23:59 МСК [1].")
    monkeypatch.setattr(factory, "make_chat", lambda settings: chat)
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    at.chat_input[0].set_value("Когда дедлайн hw1?").run()
    assert not at.exception
    assert [m["role"] for m in at.session_state["messages"]] == ["user", "assistant"]
    assert at.session_state["jobs"] == [] and len(chat.calls) == 1
    assert len(Journal(app_env["journal"]).fetch()) == 1


def test_hourly_limit_survives_new_dialog(app_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RATE_PER_HOUR", "2")
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    for question in ("Что такое DataFrame?", "Как читать CSV в pandas?"):
        at.chat_input[0].set_value(question).run()
    next(b for b in at.button if b.label == "Новый диалог").click().run()
    at.chat_input[0].set_value("Что такое groupby?").run()
    assert not at.exception
    assert "В демо можно задать до 2 вопросов в час" in at.chat_message[-1].markdown[0].value
    assert Journal(app_env["journal"]).fetch()[-1]["refusal"] == "limit_per_hour"


def test_chat_input_limited_to_question_length(app_env: dict[str, Any]):
    """Поле ввода ограничено лимитом длины вопроса: огромная вставка не оседает в session_state целиком
    (Streamlit обрезает значение по max_chars и на сервере, до кода страницы)."""
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    limit = load_settings().max_question_chars
    assert at.chat_input[0].proto.max_chars == limit
    at.chat_input[0].set_value("Что такое groupby? " + "x" * (limit * 5)).run()
    assert not at.exception
    question = at.session_state["messages"][0]["content"]
    assert question.startswith("Что такое groupby?") and len(question) <= limit


def test_llm_error_answer_is_marked_as_template(app_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    """Модель не ответила — список материалов собрал код: под ответом пометка «собран кодом по шаблону»."""
    monkeypatch.setattr(factory, "make_chat", lambda settings: FakeChat(fail=True))
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    at.chat_input[0].set_value("Чем loc отличается от iloc?").run()
    assert not at.exception
    assert "языковая модель сейчас не отвечает" in at.chat_message[-1].markdown[0].value
    assert any("ответ собран кодом по шаблону" in c.value for c in at.caption)


def test_curator_failure_shows_friendly_message(app_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch):
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("секретная деталь сервера")

    monkeypatch.setattr(Curator, "ask", broken)
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    at.chat_input[0].set_value("Когда дедлайн hw1?").run()
    assert not at.exception
    reply = at.chat_message[-1].markdown[0].value
    assert reply.startswith("Не получилось ответить") and "секретная" not in reply


def test_topic_select_shows_russian_names(app_env: dict[str, Any]):
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()
    options = list(selectbox(at, "Тема").options)
    assert "FAQ: Перенос срока" in options and "FAQ: Дедлайны" in options
    assert not any("Частые вопросы (FAQ)" in o or "(extensions)" in o for o in options)


@pytest.fixture
def default_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    """CURATOR_TRUSTED_PROXIES не задан (в .env разработчика его тоже может не быть) — доверены 127.0.0.1 и ::1."""
    monkeypatch.delenv(common.TRUSTED_PROXIES_ENV, raising=False)


@pytest.mark.usefixtures("default_proxies")
@pytest.mark.parametrize("headers, ip, expected", [
    # None — соединение с localhost (так Streamlit отдаёт 127.0.0.1 и ::1): заголовкам верим.
    ({"Cf-Connecting-Ip": "203.0.113.7", "X-Forwarded-For": "198.51.100.1"}, None, "203.0.113.7"),
    ({"cf-connecting-ip": "203.0.113.7"}, "127.0.0.1", "203.0.113.7"),
    ({"Cf-Connecting-Ip": "203.0.113.7"}, "::1", "203.0.113.7"),
    # Начало цепочки может подставить сам посетитель — берётся адрес, который дописал ближайший прокси.
    ({"X-Forwarded-For": "1.1.1.1, 198.51.100.1"}, None, "198.51.100.1"),
    ({"X-Forwarded-For": " , "}, None, None),
    ({}, "192.0.2.10", "192.0.2.10"),
    ({}, None, None),
    ({}, object(), None),  # не строка (заглушка в AppTest) — адреса нет, будет запасной ключ вкладки
    # Источник соединения неизвестен — и заголовкам не верим.
    ({"Cf-Connecting-Ip": "203.0.113.7"}, object(), None),
    ({"Cf-Connecting-Ip": "203.0.113.7"}, "   ", None),
])
def test_client_address_priority(headers: dict[str, str], ip: object, expected: str | None):
    assert common.client_address(headers, ip) == expected


@pytest.mark.usefixtures("default_proxies")
@pytest.mark.parametrize("headers", [
    {"Cf-Connecting-Ip": "203.0.113.7"},
    {"X-Forwarded-For": "203.0.113.7"},
    {"cf-connecting-ip": "203.0.113.7", "x-forwarded-for": "198.51.100.1, 203.0.113.8"},
])
def test_forged_headers_from_untrusted_lan_address_are_ignored(headers: dict[str, str]):
    """Streamlit слушает LAN: клиент из LAN, подключившись напрямую, сам пишет заголовки — ключом лимита остаётся
    адрес его соединения."""
    assert common.client_address(headers, "192.168.95.50") == "192.168.95.50"


def test_trusted_proxy_headers_are_used(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(common.TRUSTED_PROXIES_ENV, "10.20.30.40")
    assert common.client_address({"Cf-Connecting-Ip": "203.0.113.7"}, "10.20.30.40") == "203.0.113.7"
    assert common.client_address({"X-Forwarded-For": "1.1.1.1, 198.51.100.1"}, "10.20.30.40") == "198.51.100.1"
    # Тот же адрес в записи IPv4 внутри IPv6 — тот же прокси.
    assert common.client_address({"Cf-Connecting-Ip": "203.0.113.7"}, "::ffff:10.20.30.40") == "203.0.113.7"
    # Прокси без заголовков — ключом служит его адрес; соседний адрес LAN прокси не считается.
    assert common.client_address({}, "10.20.30.40") == "10.20.30.40"
    assert common.client_address({"Cf-Connecting-Ip": "203.0.113.7"}, "10.20.30.41") == "10.20.30.41"
    # Локальное соединение (None) доверено всегда, даже если список задан без 127.0.0.1.
    assert common.client_address({"Cf-Connecting-Ip": "203.0.113.7"}, None) == "203.0.113.7"
    # Явный список в вызове важнее окружения.
    assert common.client_address({"Cf-Connecting-Ip": "203.0.113.7"}, "10.20.30.40", trusted=()) == "10.20.30.40"
    assert common.client_address({"Cf-Connecting-Ip": "203.0.113.7"}, "10.0.0.9", trusted=["10.0.0.9"]) \
        == "203.0.113.7"


@pytest.mark.parametrize("raw, expected", [
    (None, {"127.0.0.1", "::1"}),
    ("", {"127.0.0.1", "::1"}),
    ("  ", {"127.0.0.1", "::1"}),
    ("10.20.30.40", {"10.20.30.40"}),  # список заменяет значение по умолчанию
    (" 10.20.30.40 , 10.20.30.41,,", {"10.20.30.40", "10.20.30.41"}),
    ("::ffff:10.20.30.40, 0:0:0:0:0:0:0:1", {"10.20.30.40", "::1"}),
])
def test_trusted_proxies_from_environment(monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: set[str]):
    if raw is None:
        monkeypatch.delenv(common.TRUSTED_PROXIES_ENV, raising=False)
    else:
        monkeypatch.setenv(common.TRUSTED_PROXIES_ENV, raw)
    assert common.trusted_proxies() == expected


def test_trusted_proxies_skip_non_addresses_with_warning(monkeypatch: pytest.MonkeyPatch,
                                                          caplog: pytest.LogCaptureFixture):
    """Опечатка или сеть вместо адреса не превращаются в «доверять всем»: запись пропускается, в журнале службы —
    предупреждение, остальные адреса работают."""
    monkeypatch.setenv(common.TRUSTED_PROXIES_ENV, "10.20.30.40, 10.20.30.0/24, cloudflared, 10.20.30.256")
    with caplog.at_level("WARNING", logger="curator.app"):
        assert common.trusted_proxies() == {"10.20.30.40"}
    assert sum(common.TRUSTED_PROXIES_ENV in r.getMessage() for r in caplog.records) == 3
    assert common.client_address({"Cf-Connecting-Ip": "203.0.113.7"}, "10.20.30.7") == "10.20.30.7"


def test_limit_key_does_not_follow_forged_header(monkeypatch: pytest.MonkeyPatch):
    """Ключ лимита: два вопроса с одного адреса LAN с разными подделанными Cf-Connecting-Ip — один ключ;
    от доверенного прокси разные Cf-Connecting-Ip — разные посетители."""
    monkeypatch.setenv(common.TRUSTED_PROXIES_ENV, "10.20.30.40")

    def key(ip: str, visitor: str) -> str:
        monkeypatch.setattr(st, "context", SimpleNamespace(headers={"Cf-Connecting-Ip": visitor}, ip_address=ip))
        return common.limit_key("salt")

    assert key("192.168.95.50", "203.0.113.7") == key("192.168.95.50", "203.0.113.8")
    assert key("192.168.95.50", "203.0.113.7").startswith("ip:")
    assert key("10.20.30.40", "203.0.113.7") != key("10.20.30.40", "203.0.113.8")


def test_streamlit_hides_error_details():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["client"]["showErrorDetails"] == "none"
