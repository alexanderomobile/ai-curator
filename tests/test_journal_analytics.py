"""Журнал (запись, оценка, срок хранения), аналитика (KPI, темы, пробелы, сложные лекции) и отчёт по темам."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from curator import analytics as an
from curator.config import MSK
from curator.journal import Journal, JournalMissing, Record, utc_iso
from curator.report import SECTIONS, build_report, kb_titles, main
from tests.conftest import FIXTURES, NOW


def lecture(doc_id: str) -> dict[str, str]:
    return {"kind": "lecture", "id": doc_id, "section": "Частые ошибки"}


def seed(journal: Journal) -> None:
    base = NOW.astimezone(timezone.utc)
    records = [
        Record(session="a", source="live", question="живой вопрос про groupby", intent="difficulty", module=2,
               topic="pandas-groupby", answered=True, sources=[lecture("L2.3")], latency_ms=2000,
               prompt_tokens=1000, completion_tokens=200, cost_usd=0.0007, ts=utc_iso(base - timedelta(hours=1))),
        Record(session="a", source="live", question="живой вопрос без ответа", intent="material", module=3,
               topic="matplotlib-basics", refusal="no_context", latency_ms=500, ts=utc_iso(base - timedelta(hours=2))),
        Record(session="b", source="eval", question="eval: как сделать pivot", intent="material", module=2,
               topic="pandas-groupby", refusal="no_context", latency_ms=400, ts=utc_iso(base - timedelta(days=1))),
        Record(session="b", source="eval", question="eval: не понимаю маски", intent="difficulty", module=2,
               topic="pandas-filtering", answered=True, sources=[lecture("L2.2"), lecture("L2.3")],
               latency_ms=3000, ts=utc_iso(base - timedelta(days=1))),
        Record(session="c", source="demo", question="Поставьте оценку", intent="grading_request", module=1,
               refusal="grading_request", answered=True, latency_ms=50, ts=utc_iso(base - timedelta(days=2))),
        Record(session="c", source="live", question="", refusal="limit_per_hour", ts=utc_iso(base - timedelta(days=2))),
        Record(session="d", source="live", question="очень старый", intent="material", ts=utc_iso(base - timedelta(days=95))),
    ]
    ids = [journal.write(r) for r in records]
    journal.rate(ids[0], -1)
    journal.rate(ids[3], 1)


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    j = Journal(tmp_path / "journal.db", retention_days=90)
    j._last_cleanup = float("inf")  # pyright: ignore[reportPrivateUsage]  # чистку проверяем отдельно
    seed(j)
    return j


def test_journal_write_rate_fetch(journal: Journal):
    rows = journal.fetch()
    assert len(rows) == 7
    assert rows[-1]["rating"] == -1 and json.loads(rows[-1]["sources"])[0]["id"] == "L2.3"
    assert rows[-1]["doc_ids"] == "L2.3" and rows[-1]["n_sources"] == 1
    assert {r["source"] for r in journal.fetch(sources=("eval",))} == {"eval"}
    with pytest.raises(ValueError):
        journal.rate(1, 5)


def test_cleanup_removes_records_older_than_retention(journal: Journal):
    assert journal.cleanup(now=NOW) == 1
    assert all(r["question"] != "очень старый" for r in journal.fetch())


def test_auto_cleanup_at_most_hourly(tmp_path: Path):
    clock = [0.0]
    j = Journal(tmp_path / "j.db", retention_days=90, clock=lambda: clock[0])
    old = utc_iso(datetime.now(timezone.utc) - timedelta(days=100))
    j.write(Record(session="x", ts=old))  # первая запись сразу запускает чистку и сама под неё попадает
    j.write(Record(session="x", ts=old))  # вторая — нет: чаще раза в час журнал не чистится
    assert len(j.fetch()) == 1
    clock[0] = 3601
    j.write(Record(session="x"))
    assert len(j.fetch()) == 1


def test_kpis_and_breakdowns(journal: Journal):
    rows = an.load(journal, 7, NOW)
    k = an.kpis(rows)
    assert k["total"] == 5 and k["limited"] == 1 and k["no_context"] == 2 and k["refusals"] == 1
    # Отказ «оценку ставит преподаватель» в старой записи с answered=1 ответом по материалам не считается.
    assert k["answered_share"] == 0.4
    assert {i["intent"]: i["answered"] for i in an.by_intent(rows)}["grading_request"] == 0.0
    by_day = {d["day"]: d for d in an.by_day(rows)}
    assert all(d["answered"] + d["no_context"] + d["refused"] <= d["questions"] for d in by_day.values())
    assert k["rated"] == 2 and k["positive_share"] == 0.5
    assert k["median_latency_ms"] == 500.0
    topics = {t["topic"]: t for t in an.by_topic(rows, {"pandas-groupby": "Группировка и агрегация"})}
    assert topics["pandas-groupby"]["questions"] == 2 and topics["pandas-groupby"]["no_context"] == 1
    assert topics["pandas-groupby"]["title"] == "Группировка и агрегация"
    assert {i["intent"]: i["questions"] for i in an.by_intent(rows)} == {"difficulty": 2, "material": 2,
                                                                        "grading_request": 1}
    refusals = {r["refusal"]: r["count"] for r in an.refusals(rows)}
    assert refusals == {"no_context": 2, "grading_request": 1, "limit_per_hour": 1}


def test_content_gaps_hide_live_texts(journal: Journal):
    rows = an.load(journal, 7, NOW)
    gaps = {g["topic"]: g for g in an.content_gaps(rows)}
    assert gaps["pandas-groupby"]["examples"] == ["eval: как сделать pivot"]
    assert gaps["matplotlib-basics"]["examples"] == ["(ещё 1 — тексты живых посетителей скрыты)"]
    shown = an.content_gaps(rows, show_live=True)
    assert any("живой вопрос без ответа" in g["examples"] for g in shown)
    assert all(not r["вопрос"].startswith("живой") for r in an.questions_table(rows, show_live=False))


def test_hard_lectures_from_difficulty_answers(journal: Journal):
    rows = an.load(journal, 7, NOW)
    hard = an.hard_lectures(rows, {"L2.3": "Группировка и агрегация"})
    assert hard[0] == {"lecture": "L2.3", "title": "Группировка и агрегация", "questions": 2, "negative": 1}
    assert hard[1]["lecture"] == "L2.2"


def test_by_day_outcome_is_complete_grid(journal: Journal):
    grid = an.by_day_outcome(an.load(journal, 7, NOW))
    days = {g["day"] for g in grid}
    assert len(grid) == len(days) * len(an.OUTCOMES)
    assert sum(g["questions"] for g in grid) == 5


def test_quantile():
    assert an.quantile([], 0.5) is None
    assert an.quantile([1, 2, 3, 4], 0.5) == 2.5
    assert an.quantile([10], 0.95) == 10


def test_report_has_all_sections(journal: Journal):
    text = build_report(an.load(journal, 7, NOW), days=7, now=NOW, titles={"L2.3": "Группировка и агрегация"})
    for n, title in enumerate(SECTIONS, 1):
        assert f"## {n}. {title}" in text
    assert "| pandas-groupby |" in text
    assert "живой вопрос" not in text and "eval: как сделать pivot" in text
    assert "Лекция L2.3 «Группировка и агрегация»: 2 вопр." in text
    assert "live 3, eval 2, demo 1" in text


def test_report_empty_period():
    text = build_report([], days=7, now=NOW)
    assert "_нет данных_" in text and "Явных пробелов не видно" in text


def test_report_cli(tmp_path: Path, journal: Journal, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("CURATOR_JOURNAL", str(journal.path))
    monkeypatch.setattr(an, "wall_now", lambda: NOW)  # «сейчас» для периода отчёта — настоящие часы
    out = tmp_path / "report.md"
    assert main(["--days", "7", "--source", "eval,demo", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "в отчёте: eval, demo" in text and "| Вопросов | 3 |" in text
    assert main(["--source", "nope"]) == 2
    capsys.readouterr()


def test_count_since_skips_limit_refusals(tmp_path: Path):
    j = Journal(tmp_path / "j.db")
    base = NOW.astimezone(timezone.utc)
    for _ in range(50):
        j.write(Record(session="spam", refusal="limit_per_hour", ts=utc_iso(base - timedelta(minutes=5))))
    j.write(Record(session="x", source="live", intent="material", answered=True, ts=utc_iso(base - timedelta(minutes=30))))
    j.write(Record(session="y", source="demo", refusal="no_context", ts=utc_iso(base - timedelta(minutes=20))))
    j.write(Record(session="z", source="eval", intent="material", ts=utc_iso(base - timedelta(minutes=10))))
    assert j.count_since(NOW - timedelta(hours=3)) == 2  # live и demo, прошедшие лимиты; eval и limit_* — нет
    stamps = j.stamps_since(NOW - timedelta(hours=1))
    assert [s.astimezone(MSK).strftime("%H:%M") for s in stamps] == ["13:30", "13:40"]


def test_fetch_until_bounds_period(journal: Journal):
    rows = journal.fetch(NOW - timedelta(days=3), until=NOW - timedelta(minutes=90))
    assert {r["question"] for r in rows} >= {"живой вопрос без ответа", "eval: как сделать pivot"}
    assert "живой вопрос про groupby" not in {r["question"] for r in rows}  # час назад — уже после until


def test_report_period_uses_wall_clock_not_curator_now(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "j.db"
    j = Journal(path)
    for day, question in ((datetime(2026, 11, 10, 12, tzinfo=MSK), "демо-дата"),
                          (datetime(2026, 12, 20, 12, tzinfo=MSK), "декабрь"),
                          (datetime(2027, 1, 15, 11, tzinfo=MSK), "сегодня")):
        j.write(Record(session="s", source="eval", question=question, intent="material", answered=True,
                       ts=utc_iso(day)))
    monkeypatch.setenv("CURATOR_JOURNAL", str(path))
    monkeypatch.setenv("CURATOR_NOW", "2026-11-12T14:00")  # застывшая демо-дата на период не влияет
    monkeypatch.setattr(an, "wall_now", lambda: datetime(2027, 1, 15, 12, 0, tzinfo=MSK))
    out = tmp_path / "r.md"
    assert main(["--days", "7", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "Период: 08.01.2027 12:00 — 15.01.2027 12:00 (МСК)" in text
    assert "| Вопросов | 1 |" in text and "live 0, eval 1, demo 0" in text  # ни демо-дата, ни декабрь не вошли
    assert [r.question for r in an.load(Journal.open_readonly(path), 7)] == ["сегодня"]


def test_report_and_stats_need_existing_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]):
    from curator.journal import main as journal_main

    missing = tmp_path / "nope" / "journal.db"
    monkeypatch.setenv("CURATOR_JOURNAL", str(missing))
    assert main(["--days", "7"]) == 2
    assert journal_main(["stats"]) == 2
    assert "журнала нет" in capsys.readouterr().err
    assert not missing.exists() and not missing.parent.exists()  # пустая база не создаётся
    with pytest.raises(JournalMissing):
        Journal.open_readonly(missing)


def test_readonly_journal_reads_and_refuses_writes(tmp_path: Path):
    path = tmp_path / "журнал куратора" / "journal.db"  # путь с кириллицей и пробелом — как у проекта
    Journal(path).write(Record(session="s", question="вопрос"))
    ro = Journal.open_readonly(path)
    assert [r["question"] for r in ro.fetch()] == ["вопрос"]
    with pytest.raises(sqlite3.OperationalError):
        ro.write(Record(session="s"))


def test_report_uses_russian_faq_topic_names(journal: Journal):
    titles = kb_titles(FIXTURES / "kb")
    assert titles["extensions"] == "FAQ: Перенос срока" and titles["grading"] == "FAQ: Оценки и проверка ДЗ"
    rows = an.load(journal, 7, NOW) + an.to_rows([{"id": 99, "ts": utc_iso(NOW - timedelta(hours=3)),
                                                     "topic": "extensions", "intent": "organizational",
                                                     "source": "eval"}])
    text = build_report(rows, days=7, now=NOW, titles=titles)
    assert "| extensions | FAQ: Перенос срока |" in text
