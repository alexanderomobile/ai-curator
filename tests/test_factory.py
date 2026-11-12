"""Сборка сервисов: таймаут эмбеддингов, лимиты после перезапуска (счётчики из журнала без отказов по лимитам)."""
from __future__ import annotations

import socket
import threading
import time
from dataclasses import replace

import pytest

from curator import factory
from curator.config import Settings, load_settings
from curator.journal import Journal, Record
from tests.fakes import FakeChat, FakeRouterLLM, TrigramEmbeddings


def test_embeddings_have_timeout(monkeypatch: pytest.MonkeyPatch):
    embeddings = factory.make_embeddings(Settings(openai_api_key="sk-x"))
    assert getattr(embeddings, "request_timeout") == 15.0  # без таймаута OpenAIEmbeddings ждёт бесконечно
    assert getattr(embeddings, "max_retries") == 2
    monkeypatch.setenv("EMBED_TIMEOUT_S", "7.5")
    assert load_settings().embed_timeout_s == 7.5


def test_embeddings_do_not_hang_on_silent_server(monkeypatch: pytest.MonkeyPatch):
    """Сервер принимает соединение и молчит: запрос эмбеддинга обрывается по таймауту, а не висит."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(10)
    held: list[socket.socket] = []

    def accept() -> None:
        while True:
            try:
                held.append(server.accept()[0])
            except OSError:
                return

    threading.Thread(target=accept, daemon=True).start()
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.getsockname()[1]}/v1")
    embeddings = factory.make_embeddings(Settings(openai_api_key="sk-x", embed_timeout_s=0.3))
    started = time.monotonic()
    try:
        with pytest.raises(Exception) as err:
            embeddings.embed_query("loc и iloc")
        elapsed = time.monotonic() - started
    finally:
        server.close()
        for conn in held:
            conn.close()
    assert "Timeout" in type(err.value).__name__ or "timed out" in str(err.value).lower()
    assert elapsed < 20  # 3 попытки по 0,3 с и паузы между ними, а не бесконечное ожидание


def test_limit_settings_from_environment(monkeypatch: pytest.MonkeyPatch):
    assert (load_settings().rate_per_hour, load_settings().hourly_cap, load_settings().daily_cap) == (20, 60, 300)
    monkeypatch.setenv("HOURLY_CAP", "5")
    assert load_settings().hourly_cap == 5


def test_restart_restores_limits_without_limit_refusals(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(factory, "make_chat", lambda s: FakeChat())
    monkeypatch.setattr(factory, "make_router_llm", lambda s: FakeRouterLLM(fail=True))
    monkeypatch.setattr(factory, "make_embeddings", lambda s: TrigramEmbeddings())
    journal = Journal(settings.journal_path)
    for _ in range(50):  # кто-то спамил после почасового отказа — это не вопросы, дневной лимит они не тратят
        journal.write(Record(session="spam", source="live", refusal="limit_per_hour"))
    journal.write(Record(session="a", source="live", intent="material", answered=True))
    services = factory.build_services(replace(settings, daily_cap=40, hourly_cap=2, rate_per_hour=10))
    assert services.limits.daily_used == 1
    assert services.limits.check("ip:b", "вопрос").ok  # второй вопрос часа
    assert services.limits.check("ip:c", "вопрос").reason == "hourly"  # потолок часа помнит вопрос до перезапуска
