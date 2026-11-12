"""Журнал запросов в SQLite — основа аналитики и отчёта по темам.

Обезличен: вместо сессии — HMAC идентификатора с солью, вопрос — после вычистки контактов (privacy.redact),
ответ не хранится вовсе. source отделяет живых посетителей (live) от прогонов оценки (eval) и кнопок-примеров
демо (demo): на публичной странице аналитики видны тексты только eval/demo. Записи старше retention_days удаляются:
таймер ai-curator-cleanup раз в сутки и само приложение не чаще раза в час, когда приходят вопросы.

Отчёт и `stats` открывают журнал только для чтения: если файла нет (не тот путь, не загружены пути службы),
команда падает с понятной ошибкой, а не создаёт пустую базу и не печатает отчёт из нулей.

    python -m curator.journal cleanup [--days 90]
    python -m curator.journal stats [--days 7]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,                       -- UTC, ISO 8601
    session TEXT NOT NULL,                  -- HMAC идентификатора сессии
    source TEXT NOT NULL DEFAULT 'live',    -- live | eval | demo
    question TEXT NOT NULL DEFAULT '',      -- после обезличивания
    intent TEXT NOT NULL DEFAULT '',
    module INTEGER,
    topic TEXT NOT NULL DEFAULT '',
    level TEXT NOT NULL DEFAULT '',         -- beginner | advanced
    level_source TEXT NOT NULL DEFAULT '',  -- manual | hint | student | default
    analysis_method TEXT NOT NULL DEFAULT '',  -- llm | rules
    guard TEXT NOT NULL DEFAULT '',         -- правило, переопределившее сценарий модели
    sources TEXT NOT NULL DEFAULT '[]',     -- JSON: [{"kind", "id", "section"}]
    doc_ids TEXT NOT NULL DEFAULT '',       -- через запятую, для быстрых выборок
    n_sources INTEGER NOT NULL DEFAULT 0,
    cited INTEGER NOT NULL DEFAULT 0,       -- модель сама сослалась хотя бы на один источник
    answered INTEGER NOT NULL DEFAULT 0,    -- ответ опирается на материалы курса
    refusal TEXT NOT NULL DEFAULT '',       -- grading_request | schedule_change | off_topic | no_context | limit_* | ...
    top_score REAL,
    filter TEXT NOT NULL DEFAULT '',        -- самый узкий фильтр цепочки ослабления, в который укладывается выдача
    relaxed INTEGER NOT NULL DEFAULT 0,     -- в выдаче есть фрагменты вне фильтра интерфейса и анализа запроса
    latency_ms INTEGER,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    rating INTEGER,                         -- 1 👍, -1 👎, NULL — без оценки
    model TEXT NOT NULL DEFAULT '',
    kb_version TEXT NOT NULL DEFAULT '',
    student_used INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_requests_ts ON requests(ts);
"""
COLUMNS = ("id", "ts", "session", "source", "question", "intent", "module", "topic", "level", "level_source",
           "analysis_method", "guard", "sources", "doc_ids", "n_sources", "cited", "answered", "refusal",
           "top_score", "filter", "relaxed", "latency_ms", "prompt_tokens", "completion_tokens", "cost_usd",
           "rating", "model", "kb_version", "student_used")
CLEANUP_EVERY_S = 3600
# Запись прошла лимиты демо, то есть дошла до куратора: отказы limit_* в дневной счётчик не входят.
_PASSED_LIMITS = "substr(refusal, 1, 6) != 'limit_'"


class JournalMissing(FileNotFoundError):
    """Журнала нет по указанному пути — для команд только чтения."""


def utc_iso(moment: datetime | None = None) -> str:
    moment = (moment or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass
class Record:
    session: str
    source: str = "live"
    question: str = ""
    intent: str = ""
    module: int | None = None
    topic: str = ""
    level: str = ""
    level_source: str = ""
    analysis_method: str = ""
    guard: str = ""
    sources: list[dict[str, str]] = field(default_factory=list)
    cited: bool = False
    answered: bool = False
    refusal: str = ""
    top_score: float | None = None
    filter: str = ""
    relaxed: bool = False
    latency_ms: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    kb_version: str = ""
    student_used: bool = False
    ts: str = field(default_factory=utc_iso)

    def row(self) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = json.dumps(self.sources, ensure_ascii=False)
        data["doc_ids"] = ",".join(dict.fromkeys(s.get("id", "") for s in self.sources if s.get("id")))
        data["n_sources"] = len(self.sources)
        for key in ("cited", "answered", "relaxed", "student_used"):
            data[key] = int(bool(data[key]))
        return data


class Journal:
    def __init__(self, path: Path, retention_days: int = 90, clock: Any = time.monotonic, readonly: bool = False):
        """readonly=True — для отчёта и статистики: файл не создаётся, запись невозможна, нет файла — JournalMissing."""
        self.path = path
        self.retention_days = retention_days
        self.readonly = readonly
        self._clock = clock
        self._last_cleanup = float("-inf")
        self._lock = threading.Lock()
        if readonly:
            if not path.is_file():
                raise JournalMissing(f"журнала нет: {path}. Проверьте CURATOR_JOURNAL (на сервере пути службы — "
                                     "в /etc/ai-curator.paths.env)")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as con:
            con.executescript(SCHEMA)

    @classmethod
    def open_readonly(cls, path: Path) -> Journal:
        return cls(path, readonly=True)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """Соединение на одну операцию: commit по выходу и обязательно close — на Windows незакрытое соединение
        держит файл журнала."""
        with self._lock:
            if self.readonly:
                con = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
            else:
                con = sqlite3.connect(self.path, timeout=10)
            con.row_factory = sqlite3.Row
            try:
                with con:
                    yield con
            finally:
                con.close()

    def write(self, record: Record) -> int:
        row = record.row()
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        with self._db() as con:
            cursor = con.execute(f"INSERT INTO requests ({cols}) VALUES ({marks})", list(row.values()))
            row_id = int(cursor.lastrowid or 0)
        now = self._clock()
        if now - self._last_cleanup >= CLEANUP_EVERY_S:
            self._last_cleanup = now
            self.cleanup()
        return row_id

    def rate(self, row_id: int, value: int | None) -> None:
        """Оценка ответа: 1 — 👍, -1 — 👎, None — снять оценку."""
        if value not in (1, -1, None):
            raise ValueError("оценка — 1, -1 или None")
        with self._db() as con:
            con.execute("UPDATE requests SET rating = ? WHERE id = ?", (value, row_id))

    def cleanup(self, days: int | None = None, now: datetime | None = None) -> int:
        keep = self.retention_days if days is None else days
        cutoff = utc_iso((now or datetime.now(timezone.utc)) - timedelta(days=keep))
        with self._db() as con:
            return con.execute("DELETE FROM requests WHERE ts < ?", (cutoff,)).rowcount

    def fetch(self, since: datetime | None = None, sources: tuple[str, ...] = (),
              until: datetime | None = None) -> list[dict[str, Any]]:
        """Записи за [since, until): обе границы — настоящее время (журнал пишет время UTC)."""
        query = f"SELECT {', '.join(COLUMNS)} FROM requests"
        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("ts >= ?")
            params.append(utc_iso(since))
        if until is not None:
            clauses.append("ts < ?")
            params.append(utc_iso(until))
        if sources:
            clauses.append(f"source IN ({', '.join('?' for _ in sources)})")
            params.extend(sources)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._db() as con:
            return [dict(r) for r in con.execute(query + " ORDER BY ts", params)]

    def count_since(self, since: datetime, sources: tuple[str, ...] = ("live", "demo")) -> int:
        """Сколько вопросов прошло лимиты демо с момента since — для дневного счётчика после перезапуска.
        Отказы limit_* не считаются: DemoLimits их тоже не засчитывает."""
        marks = ", ".join("?" for _ in sources)
        with self._db() as con:
            row = con.execute(f"SELECT COUNT(*) FROM requests WHERE ts >= ? AND source IN ({marks}) "
                              f"AND {_PASSED_LIMITS}", [utc_iso(since), *sources]).fetchone()
        return int(row[0])

    def stamps_since(self, since: datetime, sources: tuple[str, ...] = ("live", "demo")) -> list[datetime]:
        """Когда были вопросы, прошедшие лимиты, с момента since — окно общего лимита в час после перезапуска."""
        marks = ", ".join("?" for _ in sources)
        with self._db() as con:
            rows = con.execute(f"SELECT ts FROM requests WHERE ts >= ? AND source IN ({marks}) AND {_PASSED_LIMITS} "
                               "ORDER BY ts", [utc_iso(since), *sources]).fetchall()
        return [datetime.fromisoformat(str(r[0]).replace("Z", "+00:00")) for r in rows]


def main(argv: list[str] | None = None) -> int:
    from .config import load_settings

    settings = load_settings()
    parser = argparse.ArgumentParser(prog="python -m curator.journal", description="Журнал запросов куратора")
    sub = parser.add_subparsers(dest="command", required=True)
    clean = sub.add_parser("cleanup", help="удалить записи старше срока хранения")
    clean.add_argument("--days", type=int, default=settings.retention_days)
    stats = sub.add_parser("stats", help="сколько записей по источникам")
    stats.add_argument("--days", type=int, default=7)
    args = parser.parse_args(argv)
    if args.command == "cleanup":
        journal = Journal(settings.journal_path, settings.retention_days)
        print(f"удалено записей: {journal.cleanup(args.days)} (старше {args.days} дн.)")
        return 0
    try:
        journal = Journal.open_readonly(settings.journal_path)
    except JournalMissing as exc:
        print(exc, file=sys.stderr)
        return 2
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    counts: dict[str, int] = {}
    for row in journal.fetch(since):
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    print(json.dumps({"days": args.days, "by_source": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
