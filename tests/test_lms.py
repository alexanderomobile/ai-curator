"""LMS: дни до дедлайна по МСК (считает код), окно переноса, выгрузка CSV, HTTP-заглушка и выбор бэкенда."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from curator.config import MSK
from curator.lms import (RECENT_DAYS, RECORDING_DELAY, REVISION_DAYS, Assignment, FileBackend, HttpBackend, LMSClient,
                         LMSError, deadline_status, duration_text, event_when, recording_due, revision_rule,
                         export, left_phrase, main, make_backend, plural)
from curator.lms_stub_api import create_app
from tests.conftest import FIXTURES, NOW

LMS_DIR = FIXTURES / "lms"
HW1_DEADLINE = datetime(2026, 11, 15, 23, 59, tzinfo=MSK)


@pytest.fixture
def client() -> LMSClient:
    return LMSClient(FileBackend(LMS_DIR))


@pytest.mark.parametrize("n, word", [(1, "день"), (2, "дня"), (4, "дня"), (5, "дней"), (11, "дней"), (12, "дней"),
                                     (21, "день"), (22, "дня"), (25, "дней"), (111, "дней")])
def test_plural(n: int, word: str):
    assert plural(n, ("день", "дня", "дней")) == word


@pytest.mark.parametrize("now, phrase", [
    # Остаток — длительность (дни и часы), а не разница календарных дат: в 09:59 и в 23:58 одного дня он разный.
    (NOW, "осталось 3 дня 9 часов"),
    (datetime(2026, 11, 12, 9, 59, tzinfo=MSK), "осталось 3 дня 14 часов"),
    (datetime(2026, 11, 12, 23, 58, tzinfo=MSK), "осталось 3 дня"),  # 3 дня 0 ч 1 мин — без «0 часов»
    (datetime(2026, 11, 13, 0, 1, tzinfo=MSK), "осталось 2 дня 23 часа"),  # было «осталось 2 дня»
    (datetime(2026, 11, 14, 9, 0, tzinfo=MSK), "дедлайн завтра в 23:59, осталось 38 часов"),
    (datetime(2026, 11, 15, 12, 0, tzinfo=MSK), "дедлайн сегодня в 23:59, осталось 11 часов"),
    # 20:30 UTC — это 23:30 по Москве: дедлайн ещё сегодня, хотя в UTC до полуночи далеко.
    (datetime(2026, 11, 15, 20, 30, tzinfo=timezone.utc), "дедлайн сегодня в 23:59, осталось меньше часа"),
    # 21:30 UTC 15.11 — уже 00:30 16.11 по Москве: дедлайн прошёл вчера, хотя по UTC ещё 15-е.
    (datetime(2026, 11, 15, 21, 30, tzinfo=timezone.utc), "дедлайн прошёл вчера в 23:59"),
    # Через 30 секунд после дедлайна — «вчера в 23:59», а не «1 день назад».
    (datetime(2026, 11, 16, 0, 0, 30, tzinfo=MSK), "дедлайн прошёл вчера в 23:59"),
    (datetime(2026, 11, 15, 23, 59, 30, tzinfo=MSK), "дедлайн прошёл сегодня в 23:59"),
    (datetime(2026, 11, 20, 10, 0, tzinfo=MSK), "дедлайн прошёл 4 дня 10 часов назад"),
])
def test_left_phrase_in_moscow_time(now: datetime, phrase: str):
    assert left_phrase(HW1_DEADLINE, now) == phrase


def test_left_phrase_tomorrow_less_than_hour():
    deadline = datetime(2026, 11, 16, 0, 30, tzinfo=MSK)
    assert left_phrase(deadline, datetime(2026, 11, 15, 23, 50, tzinfo=MSK)) == \
        "дедлайн завтра в 00:30, осталось меньше часа"  # было «осталось 0 часов»


@pytest.mark.parametrize("delta, text", [
    (timedelta(minutes=59), "меньше часа"), (timedelta(hours=1), "1 час"), (timedelta(hours=47, minutes=59), "47 часов"),
    (timedelta(days=2), "2 дня"), (timedelta(days=2, hours=1), "2 дня 1 час"), (timedelta(days=5, hours=22), "5 дней 22 часа"),
    (timedelta(days=21, hours=2, minutes=59), "21 день 2 часа"),
])
def test_duration_text_never_rounds_up(delta: timedelta, text: str):
    assert duration_text(delta) == text


def test_deadline_status_and_extension_window(client: LMSClient):
    hw1 = client.assignments(1)[0]
    status = deadline_status(hw1, NOW)
    assert (status.days_left, status.is_past, status.can_apply) == (3, False, True)
    assert status.apply_until == datetime(2026, 11, 14, 23, 59, tzinfo=MSK)
    late = deadline_status(hw1, datetime(2026, 11, 15, 1, 0, tzinfo=MSK))
    assert not late.can_apply and not late.is_past


def test_personal_deadline_after_extension(client: LMSClient):
    hw1 = client.assignments(1)[0]
    s5 = client.student("s5")
    assert s5 is not None
    status = deadline_status(hw1, NOW, s5)
    assert status.personal and status.deadline == datetime(2026, 11, 18, 23, 59, tzinfo=MSK)
    assert status.phrase == "осталось 6 дней 9 часов"


def test_facts_text_is_ready_for_prompt(client: LMSClient):
    facts = client.facts(NOW, module=1, student=client.student("s2"))
    assert "Данные LMS на чт 12.11.2026 14:00 МСК" in facts
    assert "hw1 «Анализ списка покупок» (модуль 1)" in facts and "— осталось 3 дня 9 часов." in facts
    assert "а не «ровно N дней»" in facts
    assert "Статус у студента: принято." in facts
    assert "пт 13.11.2026 18:00 МСК — Разбор ДЗ" in facts
    assert "Студент s2, уровень продвинутый; перенос дедлайна в курсе: не использован." in facts
    assert "Дмитрий" not in facts  # имя студента модели не нужно


def test_upcoming_events_scoped_to_module_with_fallback(client: LMSClient):
    assert [e.id for e in client.upcoming(NOW, module=2)] == ["webinar-03", "review-03"]
    assert [e.id for e in client.upcoming(NOW, module=4)] == ["review-02", "webinar-03", "review-03"]
    assert [e.id for e in client.upcoming(NOW, types=("review",), limit=1)] == ["review-02"]


def test_csv_export_round_trip(client: LMSClient, tmp_path: Path):
    written = export(client, tmp_path, "both")
    assert {p.name for p in written} == {f"{k}.{ext}" for k in ("schedule", "assignments", "students")
                                         for ext in ("csv", "json")}
    raw = (tmp_path / "students.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM: Excel открывает кириллицу
    header = raw.decode("utf-8-sig").splitlines()[0]
    assert header == ("id,name,level,extension_used,extension_assignment,extension_days,extension_new_deadline,"
                      "hw1,hw2")
    from_csv = LMSClient(FileBackend(tmp_path, fmt="csv"))
    assert from_csv.schedule() == client.schedule()
    assert from_csv.assignments() == client.assignments()
    assert from_csv.students() == client.students()


def test_export_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CURATOR_DATA_DIR", str(FIXTURES))
    assert main(["export", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "assignments.csv").read_text(encoding="utf-8-sig").startswith("id,module,title,opens_at")


@pytest.fixture
def stub() -> TestClient:
    return TestClient(create_app(lambda: FileBackend(LMS_DIR)))


def test_stub_api_json_and_csv(stub: TestClient):
    assert stub.get("/health").json()["status"] == "ok"
    schedule = stub.get("/api/schedule", params={"module": 2})
    assert schedule.status_code == 200 and [e["id"] for e in schedule.json()] == ["webinar-03", "review-03"]
    csv = stub.get("/api/assignments", params={"format": "csv"})
    assert csv.headers["content-type"].startswith("text/csv")
    assert "attachment" in csv.headers["content-disposition"]
    lines = csv.text.lstrip("﻿").splitlines()
    assert lines[0].startswith("id,module,title") and len(lines) == 3
    assert stub.get("/api/students/s5").json()["extension"]["assignment_id"] == "hw1"
    assert stub.get("/api/students/nobody").status_code == 404
    assert stub.get("/api/schedule", params={"format": "xml"}).status_code == 422


def test_http_backend_matches_files(stub: TestClient, client: LMSClient):
    http = LMSClient(HttpBackend("http://testserver", client=stub))
    assert http.schedule() == client.schedule()
    assert http.assignments() == client.assignments()
    assert http.students() == client.students()
    assert http.student("nobody") is None
    assert http.facts(NOW, 1) == client.facts(NOW, 1)


def test_backend_chosen_by_lms_url(tmp_path: Path):
    assert isinstance(make_backend("", tmp_path), FileBackend)
    backend = make_backend("http://127.0.0.1:8502", tmp_path)
    assert isinstance(backend, HttpBackend) and backend.base_url == "http://127.0.0.1:8502"


def test_assignment_model_accepts_lms_json():
    a = Assignment.model_validate({"id": "x", "module": 1, "title": "t", "opens_at": "2026-11-02T00:00:00+03:00",
                                   "deadline": "2026-11-15T23:59:00+03:00", "max_score": 10})
    assert a.deadline.utcoffset() is not None


def test_assignment_model_rejects_naive_datetime():
    with pytest.raises(ValidationError):
        Assignment.model_validate({"id": "x", "module": 1, "title": "t", "opens_at": "2026-11-02T00:00:00",
                                   "deadline": "2026-11-15T23:59:00", "max_score": 10})


class _Response:
    def __init__(self, status: int, body: str):
        self.status_code = status
        self.body = body

    def json(self) -> Any:
        return json.loads(self.body)


class _Client:
    def __init__(self, status: int, body: str):
        self.response = _Response(status, body)

    def get(self, url: str) -> _Response:
        return self.response


@pytest.mark.parametrize("status, body", [
    (200, "<html>Bad gateway</html>"),  # прокси или туннель ответил страницей с кодом 200
    (200, '[{"id": "x"}]'),  # не та схема
    (200, '[{"id": "w", "type": "webinar", "title": "t", "starts_at": "2026-11-18T19:00:00"}]'),  # дата без зоны
    (502, "bad gateway"),
])
def test_http_backend_errors_become_lms_error(status: int, body: str):
    client = LMSClient(HttpBackend("http://lms.example", client=_Client(status, body)))
    for call in (client.schedule, client.assignments, client.students, lambda: client.student("s1"),
                 lambda: client.facts(NOW, 1)):
        with pytest.raises(LMSError):
            call()


def test_file_backend_errors_become_lms_error(tmp_path: Path):
    shutil.copytree(LMS_DIR, tmp_path / "lms")
    schedule = tmp_path / "lms" / "schedule.json"
    schedule.write_text(schedule.read_text(encoding="utf-8").replace("+03:00", ""), encoding="utf-8")
    (tmp_path / "lms" / "students.json").write_text("{не json", encoding="utf-8")
    client = LMSClient(FileBackend(tmp_path / "lms"))
    with pytest.raises(LMSError, match="schedule.json"):
        client.upcoming(NOW)
    with pytest.raises(LMSError, match="students.json"):
        client.student("s1")
    assert [a.id for a in client.assignments()] == ["hw1", "hw2"]  # исправный файл читается как раньше
    (tmp_path / "lms" / "assignments.csv").write_text("id,title\nhw1,t\n", encoding="utf-8")
    with pytest.raises(LMSError):
        LMSClient(FileBackend(tmp_path / "lms", fmt="csv")).assignments()


# --- Доработка: правило вместо «осталось N до дедлайна»; недавние события; время событий.

def test_facts_for_revision_give_rule_not_time_left(client: LMSClient):
    facts = client.facts(NOW, student=client.student("s3"))
    hw1 = next(line for line in facts.splitlines() if line.startswith("- hw1 "))
    hw2 = next(line for line in facts.splitlines() if line.startswith("- hw2 "))
    assert "осталось" not in hw1
    assert "исходный дедлайн вс 15.11.2026 23:59 МСК (к доработке не относится)" in hw1
    assert f"на доработку — на исправление {REVISION_DAYS} дней с момента решения преподавателя" in hw1
    assert "даты решения в данных LMS нет" in hw1 and "доработка не тратит перенос дедлайна" in hw1
    assert "Формат сдачи: Ссылка на Google Colab или файл .ipynb в LMS." in hw1
    assert "осталось 17 дней 9 часов" in hw2  # у задания не на доработке — как раньше


def test_status_labels_match_course_terms(client: LMSClient):
    assert "Статус у студента: на проверке." in client.facts(NOW, 1, client.student("s4"))
    assert revision_rule().startswith(f"на исправление {REVISION_DAYS} дней")


@pytest.mark.parametrize("now, recent", [
    (NOW, ["webinar-02"]),  # ср 11.11 19:00 — вчера; разбор 13.11 ещё впереди
    (datetime(2026, 11, 14, 12, 0, tzinfo=MSK), ["webinar-02", "review-02"]),
    (datetime(2026, 11, 15, 12, 0, tzinfo=MSK), ["review-02"]),  # вебинар 11.11 — больше трёх дней назад
    (datetime(2026, 11, 11, 18, 0, tzinfo=MSK), []),
])
def test_recent_events_window(client: LMSClient, now: datetime, recent: list[str]):
    assert RECENT_DAYS == 3
    assert [e.id for e in client.recent(now)] == recent


def test_facts_list_recent_events(client: LMSClient):
    facts = client.facts(NOW, 2)
    assert "Прошедшие события за последние 3 дня:\n- ср 11.11.2026 19:00 МСК — Вебинар 2. Функции и модули (90 мин)" \
        in facts


def test_event_when_has_end_time(client: LMSClient):
    webinar = next(e for e in client.schedule() if e.id == "webinar-03")
    assert event_when(webinar) == "ср 18.11.2026 19:00-20:30"
    assert event_when(webinar.model_copy(update={"duration_min": None})) == "ср 18.11.2026 19:00"


# --- R3-6: срок появления записи прошедшего вебинара считает код (правило FAQ: 24 часа после эфира).

def _recording_line(facts: str) -> str:
    lines = facts.splitlines()
    head = next(n for n, line in enumerate(lines) if line.startswith("Записи прошедших вебинаров"))
    assert "запись появляется в LMS не позже чем через 24 часа после окончания эфира" in lines[head]
    return lines[head + 1]


def test_recording_due_is_end_of_webinar_plus_rule(client: LMSClient):
    webinar = next(e for e in client.schedule() if e.id == "webinar-02")  # ср 11.11 19:00, 90 мин
    assert RECORDING_DELAY == timedelta(hours=24)
    assert recording_due(webinar) == datetime(2026, 11, 12, 20, 30, tzinfo=MSK)
    assert recording_due(webinar.model_copy(update={"duration_min": None})) is None  # конец эфира неизвестен
    review = next(e for e in client.schedule() if e.type == "review")
    assert recording_due(review) is None  # запись — только у вебинаров


def test_facts_say_recording_not_due_yet_for_yesterday_webinar(client: LMSClient):
    line = _recording_line(client.facts(NOW, 2))  # чт 12.11 14:00 — сутки после эфира ещё не прошли
    assert line.startswith("- Вебинар 2. Функции и модули (ср 11.11.2026 19:00-20:30 МСК): ")
    assert line.endswith("запись появится в LMS не позже чт 12.11.2026 20:30 МСК; этот срок ещё не наступил "
                         "(до него 6 часов)")
    late = _recording_line(client.facts(datetime(2026, 11, 12, 21, 0, tzinfo=MSK), 2))
    assert late.endswith("запись должна была появиться в LMS не позже чт 12.11.2026 20:30 МСК; этот срок уже прошёл")


def test_recordings_block_only_for_recent_webinars(client: LMSClient):
    facts = client.facts(datetime(2026, 11, 14, 12, 0, tzinfo=MSK), 2)  # прошли вебинар 11.11 и разбор 13.11
    block = facts.split("Записи прошедших вебинаров", 1)[1].splitlines()[1:]
    assert [line for line in block if line.startswith("- ")] == [block[0]] and "Вебинар 2." in block[0]
    assert "Записи прошедших вебинаров" not in client.facts(datetime(2026, 11, 16, 12, 0, tzinfo=MSK), 2)
