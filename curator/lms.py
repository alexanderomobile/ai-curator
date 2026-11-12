"""Данные LMS: расписание, задания, студенты — и всё, что из них считает код (дни до дедлайна, окно переноса).

Два бэкенда с одним интерфейсом:
- файлы data/lms/*.json (и обратно — выгрузка в CSV: `python -m curator.lms export`);
- HTTP-заглушка LMS (curator/lms_stub_api.py), если задана переменная LMS_URL.

Модель сама даты не считает: «осталось 3 дня 9 часов», «заявку можно подать до …» и «запись вчерашнего вебинара —
не позже …, срок ещё не наступил» готовит этот модуль в московском времени, а в промпт попадает уже готовый текст.
«Осталось» — длительность до дедлайна (дни и часы, минуты отбрасываются), а не разница календарных дат: в 23:58
до дедлайна через три дня осталось 3 дня, а не «3 дня» и в 09:59, и в 23:58.

Любой сбой источника данных — сеть, HTML вместо JSON, неверная схема, дата без часового пояса — превращается
в LMSError: куратор отвечает без данных LMS, а не падает.

    python -m curator.lms export [--out DIR] [--format csv|json|both]
    python -m curator.lms show [--student s3]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

import httpx
from pydantic import AwareDatetime, BaseModel, TypeAdapter

from .config import MSK

T = TypeVar("T")

Status = Literal["not_submitted", "submitted", "accepted", "revision"]
# Названия статусов — как в LMS и в правилах курса (обзор курса, «Оценивание»).
STATUS_LABELS = {"not_submitted": "не сдано", "submitted": "на проверке", "accepted": "принято",
                 "revision": "на доработку"}
EVENT_LABELS = {"webinar": "Вебинар", "review": "Разбор ДЗ", "defense": "Защита"}
LEVEL_LABELS = {"beginner": "начинающий", "advanced": "продвинутый"}
WEEKDAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")
EXTENSION_NOTICE = timedelta(hours=24)  # заявка на перенос — не позже чем за 24 часа до дедлайна (FAQ курса)
# Доработка (FAQ «Сколько времени даётся на доработку?»): 5 дней с решения преподавателя, а не от исходного
# дедлайна; перенос не тратит. Даты решения в данных LMS нет, поэтому срок доработки код не считает.
REVISION_DAYS = 5
# Прошедшие события в фактах LMS: «где запись вчерашнего вебинара?» — запись появляется в течение суток, а вебинар
# идёт раз в неделю; трёх дней хватает на вчерашний и позавчерашний эфир, но не тянет в промпт всю историю.
RECENT_DAYS = 3
# Запись вебинара (FAQ «Будут ли записи вебинаров?»): появляется в LMS в течение 24 часов после эфира. Срок для
# прошедшего вебинара считает код — «не позже <дата время>» и наступил ли он, — как «осталось» до дедлайна: иначе
# модель выводит «запись уже должна быть» про вчерашний эфир, хотя сутки ещё не прошли.
RECORDING_DELAY = timedelta(hours=24)
RECORDED_TYPES = ("webinar",)
KINDS = ("schedule", "assignments", "students")


def revision_rule() -> str:
    """Правило доработки для промпта и шаблонов: без даты — её в данных LMS нет."""
    return (f"на исправление {REVISION_DAYS} {plural(REVISION_DAYS, DAYS)} с момента решения преподавателя; "
            "даты решения в данных LMS нет, поэтому точный срок доработки здесь не посчитать; замечания преподавателя — "
            "в комментариях к работе в LMS. Исходный дедлайн "
            "задания к доработке не относится, доработка не тратит перенос дедлайна")


# Даты — только с часовым поясом (AwareDatetime): дата без зоны из «настоящей LMS» даёт понятную ошибку при загрузке,
# а не TypeError при сравнении с текущим временем посреди ответа.
class Event(BaseModel):
    id: str
    type: str
    title: str
    module: int | None = None
    starts_at: AwareDatetime
    duration_min: int | None = None
    link: str = ""


class Assignment(BaseModel):
    id: str
    module: int
    title: str
    opens_at: AwareDatetime
    deadline: AwareDatetime
    max_score: int
    submit_format: str = ""
    guide_id: str = ""


class Extension(BaseModel):
    assignment_id: str
    days: int
    new_deadline: AwareDatetime


class Student(BaseModel):
    id: str
    name: str
    level: Literal["beginner", "advanced"]
    extension_used: bool = False
    extension: Extension | None = None
    assignments: dict[str, Status] = {}

    def deadline_for(self, assignment: Assignment) -> datetime:
        """Личный дедлайн: после одобренного переноса — новый."""
        if self.extension and self.extension.assignment_id == assignment.id:
            return self.extension.new_deadline
        return assignment.deadline


_EVENTS = TypeAdapter(list[Event])
_ASSIGNMENTS = TypeAdapter(list[Assignment])
_STUDENTS = TypeAdapter(list[Student])


class LMSError(RuntimeError):
    pass


def _guarded(what: str, fn: Callable[[], T]) -> T:
    """Ошибки разбора данных LMS (JSONDecodeError, ValidationError — подклассы ValueError; KeyError в CSV без
    столбца; TypeError от неожиданных типов) → LMSError с понятным текстом."""
    try:
        return fn()
    except LMSError:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise LMSError(f"LMS: {what} — данные не по формату ({type(exc).__name__})") from exc


class LMSBackend(Protocol):
    name: str

    def schedule(self) -> list[Event]: ...

    def assignments(self) -> list[Assignment]: ...

    def students(self) -> list[Student]: ...

    def student(self, student_id: str) -> Student | None: ...


# --- CSV: одна плоская таблица на вид данных. У студента статусы заданий — отдельные столбцы.

def _iso(value: datetime | None) -> str:
    return value.isoformat() if value else ""


def rows(kind: str, schedule: Sequence[Event] = (), assignments: Sequence[Assignment] = (),
         students: Sequence[Student] = ()) -> tuple[list[str], list[dict[str, Any]]]:
    if kind == "schedule":
        header = ["id", "type", "title", "module", "starts_at", "duration_min", "link"]
        return header, [{**e.model_dump(), "starts_at": _iso(e.starts_at)} for e in schedule]
    if kind == "assignments":
        header = ["id", "module", "title", "opens_at", "deadline", "max_score", "submit_format", "guide_id"]
        return header, [{**a.model_dump(), "opens_at": _iso(a.opens_at), "deadline": _iso(a.deadline)}
                        for a in assignments]
    if kind == "students":
        ids = sorted({k for s in students for k in s.assignments}, key=lambda k: (k == "final", k))
        header = ["id", "name", "level", "extension_used", "extension_assignment", "extension_days",
                  "extension_new_deadline", *ids]
        out = []
        for s in students:
            ext = s.extension
            out.append({"id": s.id, "name": s.name, "level": s.level, "extension_used": s.extension_used,
                        "extension_assignment": ext.assignment_id if ext else "",
                        "extension_days": ext.days if ext else "",
                        "extension_new_deadline": _iso(ext.new_deadline) if ext else "",
                        **{k: s.assignments.get(k, "") for k in ids}})
        return header, out
    raise LMSError(f"неизвестный вид данных {kind!r}: {', '.join(KINDS)}")


def to_csv(header: list[str], data: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    for row in data:
        writer.writerow({k: ("true" if v is True else "false" if v is False else v) for k, v in row.items()})
    return buffer.getvalue()


def _read_csv(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))


def _blank_none(value: str) -> str | None:
    return value if value != "" else None


def parse_csv(kind: str, text: str) -> list[Any]:
    """Обратный разбор выгрузки — та же модель данных, что из JSON."""
    table = _read_csv(text)
    if kind == "schedule":
        return _EVENTS.validate_python([{**r, "module": _blank_none(r["module"]),
                                         "duration_min": _blank_none(r["duration_min"])} for r in table])
    if kind == "assignments":
        return _ASSIGNMENTS.validate_python(table)
    if kind == "students":
        fixed = {"id", "name", "level", "extension_used", "extension_assignment", "extension_days",
                 "extension_new_deadline"}
        out = []
        for r in table:
            ext = None
            if r.get("extension_assignment"):
                ext = {"assignment_id": r["extension_assignment"], "days": r["extension_days"],
                       "new_deadline": r["extension_new_deadline"]}
            out.append({"id": r["id"], "name": r["name"], "level": r["level"],
                        "extension_used": r["extension_used"].strip().lower() == "true", "extension": ext,
                        "assignments": {k: v for k, v in r.items() if k not in fixed and v}})
        return _STUDENTS.validate_python(out)
    raise LMSError(f"неизвестный вид данных {kind!r}")


class FileBackend:
    """data/lms/{schedule,assignments,students}.json; fmt="csv" — читать выгрузку *.csv из той же папки."""

    def __init__(self, directory: Path, fmt: Literal["json", "csv"] = "json"):
        self.directory = directory
        self.fmt = fmt
        self.name = f"files:{fmt}"

    def _load(self, kind: str) -> list[Any]:
        path = self.directory / f"{kind}.{self.fmt}"
        if not path.is_file():
            raise LMSError(f"нет файла LMS {path}")
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            raise LMSError(f"не прочитать файл LMS {path}: {type(exc).__name__}") from exc
        if self.fmt == "csv":
            return _guarded(path.name, lambda: parse_csv(kind, text))
        adapter = {"schedule": _EVENTS, "assignments": _ASSIGNMENTS, "students": _STUDENTS}[kind]
        return _guarded(path.name, lambda: adapter.validate_json(text))

    def schedule(self) -> list[Event]:
        return sorted(self._load("schedule"), key=lambda e: e.starts_at)

    def assignments(self) -> list[Assignment]:
        return sorted(self._load("assignments"), key=lambda a: a.deadline)

    def students(self) -> list[Student]:
        return self._load("students")

    def student(self, student_id: str) -> Student | None:
        return next((s for s in self.students() if s.id == student_id), None)


class HttpResponse(Protocol):
    @property
    def status_code(self) -> int: ...

    def json(self) -> Any: ...


class HttpClient(Protocol):
    """Что нужно от HTTP-клиента: httpx.Client в работе, TestClient заглушки в тестах (он на httpx2)."""

    def get(self, url: str) -> HttpResponse: ...


class HttpBackend:
    """HTTP-заглушка LMS (или настоящая LMS с тем же API)."""

    def __init__(self, base_url: str, timeout: float = 5.0, client: HttpClient | None = None):
        self.base_url = base_url.rstrip("/")
        self.client: HttpClient = client or httpx.Client(timeout=timeout)
        self.name = f"http:{self.base_url}"

    def _get(self, path: str) -> Any:
        try:
            response = self.client.get(f"{self.base_url}{path}")
        except httpx.HTTPError as exc:
            raise LMSError(f"LMS {self.base_url} недоступна: {type(exc).__name__}") from exc
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise LMSError(f"LMS {self.base_url}{path}: HTTP {response.status_code}")
        # Прокси или туннель может ответить HTML-страницей с кодом 200 — это сбой LMS, а не данные.
        return _guarded(f"{self.base_url}{path}", response.json)

    def schedule(self) -> list[Event]:
        data = self._get("/api/schedule") or []
        return sorted(_guarded("/api/schedule", lambda: _EVENTS.validate_python(data)), key=lambda e: e.starts_at)

    def assignments(self) -> list[Assignment]:
        data = self._get("/api/assignments") or []
        return sorted(_guarded("/api/assignments", lambda: _ASSIGNMENTS.validate_python(data)),
                      key=lambda a: a.deadline)

    def students(self) -> list[Student]:
        data = self._get("/api/students") or []
        return _guarded("/api/students", lambda: _STUDENTS.validate_python(data))

    def student(self, student_id: str) -> Student | None:
        data = self._get(f"/api/students/{student_id}")
        return _guarded("/api/students/{id}", lambda: Student.model_validate(data)) if data else None


# --- Расчёты в московском времени.

def plural(n: int, forms: tuple[str, str, str]) -> str:
    """plural(3, ("день", "дня", "дней")) → «дня»."""
    n = abs(n)
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


DAYS = ("день", "дня", "дней")
HOURS = ("час", "часа", "часов")


def fmt_dt(moment: datetime, weekday: bool = True) -> str:
    local = moment.astimezone(MSK)
    prefix = f"{WEEKDAYS[local.weekday()]} " if weekday else ""
    return f"{prefix}{local:%d.%m.%Y %H:%M}"


def calendar_days(target: datetime, now: datetime) -> int:
    """Сколько календарных дней по Москве от now до target (сегодня → 0, завтра → 1)."""
    return (target.astimezone(MSK).date() - now.astimezone(MSK).date()).days


def duration_text(delta: timedelta) -> str:
    """Длительность для студента: от двух суток — «3 дня 14 часов» (минуты отбрасываются, ровное число дней —
    без часов), до двух суток — «38 часов», меньше часа — «меньше часа». Округления вверх нет."""
    total_hours = int(delta.total_seconds() // 3600)
    if total_hours < 1:
        return "меньше часа"
    days, hours = divmod(total_hours, 24)
    if days < 2:
        return f"{total_hours} {plural(total_hours, HOURS)}"
    tail = f" {hours} {plural(hours, HOURS)}" if hours else ""
    return f"{days} {plural(days, DAYS)}{tail}"


def left_phrase(deadline: datetime, now: datetime) -> str:
    """Сколько осталось до дедлайна (или сколько прошло после) — длительность, а не разница календарных дат."""
    delta = deadline - now
    days = calendar_days(deadline, now)
    at = f"{deadline.astimezone(MSK):%H:%M}"
    if delta.total_seconds() <= 0:
        passed = -delta
        if passed < timedelta(hours=24):  # меньше суток — по календарю это сегодня или вчера
            return f"дедлайн прошёл {'сегодня' if days == 0 else 'вчера'} в {at}"
        return f"дедлайн прошёл {duration_text(passed)} назад"
    if days == 0:
        return f"дедлайн сегодня в {at}, осталось {duration_text(delta)}"
    if days == 1:
        return f"дедлайн завтра в {at}, осталось {duration_text(delta)}"
    return f"осталось {duration_text(delta)}"


@dataclass(frozen=True)
class DeadlineStatus:
    assignment: Assignment
    deadline: datetime  # личный, если у студента одобрен перенос этого задания
    personal: bool
    days_left: int  # календарные дни по Москве; отрицательно — дедлайн прошёл
    hours_left: float
    is_open: bool  # задание уже открыто
    is_past: bool
    apply_until: datetime  # последний момент подачи заявки на перенос
    can_apply: bool  # окно подачи ещё открыто (без учёта того, использован ли перенос)
    phrase: str
    status: str = ""  # статус студента по заданию, если студент известен


def deadline_status(assignment: Assignment, now: datetime, student: Student | None = None) -> DeadlineStatus:
    deadline = student.deadline_for(assignment) if student else assignment.deadline
    seconds = (deadline - now).total_seconds()
    apply_until = assignment.deadline - EXTENSION_NOTICE
    return DeadlineStatus(
        assignment=assignment, deadline=deadline, personal=deadline != assignment.deadline,
        days_left=calendar_days(deadline, now), hours_left=round(seconds / 3600, 1),
        is_open=now >= assignment.opens_at, is_past=seconds <= 0,
        apply_until=apply_until, can_apply=now < apply_until, phrase=left_phrase(deadline, now),
        status=student.assignments.get(assignment.id, "") if student else "",
    )


class LMSClient:
    def __init__(self, backend: LMSBackend):
        self.backend = backend

    @property
    def name(self) -> str:
        return self.backend.name

    # Бэкенды сами переводят сбои в LMSError; обёртка _guarded — на случай другого бэкенда с тем же протоколом.
    def schedule(self, module: int | None = None) -> list[Event]:
        events = _guarded("расписание", self.backend.schedule)
        return [e for e in events if module is None or e.module == module]

    def assignments(self, module: int | None = None) -> list[Assignment]:
        items = _guarded("задания", self.backend.assignments)
        return [a for a in items if module is None or a.module == module]

    def students(self) -> list[Student]:
        return _guarded("студенты", self.backend.students)

    def student(self, student_id: str | None) -> Student | None:
        return _guarded("студент", lambda: self.backend.student(student_id)) if student_id else None

    def upcoming(self, now: datetime, module: int | None = None, limit: int = 3,
                 types: Sequence[str] = ()) -> list[Event]:
        events = [e for e in self.schedule() if e.starts_at >= now and (not types or e.type in types)]
        scoped = [e for e in events if module is None or e.module == module]
        return (scoped or events)[:limit]

    def recent(self, now: datetime, days: int = RECENT_DAYS) -> list[Event]:
        """События, которые уже начались за последние days дней (по времени начала), от ранних к поздним."""
        since = now - timedelta(days=days)
        return [e for e in self.schedule() if since <= e.starts_at < now]

    def deadlines(self, now: datetime, module: int | None = None, student: Student | None = None) -> list[DeadlineStatus]:
        items = self.assignments(module) or self.assignments()
        return [deadline_status(a, now, student) for a in items]

    def facts(self, now: datetime, module: int | None = None, student: Student | None = None,
              events_limit: int = 3) -> str:
        """Текст для промпта: задания со сроками, ближайшие и недавние события, статусы студента — всё уже
        посчитано. У задания на доработке «осталось N» до исходного дедлайна не пишется: срок доработки считается
        от решения преподавателя, и модель приняла бы остаток до дедлайна за время на исправление."""
        lines = [f"Данные LMS на {fmt_dt(now)} МСК. Сроки посчитаны системой — пересчитывать их не нужно. "
                 "«Осталось» — точное время до дедлайна в днях и часах (минуты отброшены), а не «ровно N дней»."]
        lines.append("Задания:")
        for st in self.deadlines(now, module, student):
            a = st.assignment
            state = (f"открыто с {fmt_dt(a.opens_at, weekday=False)}" if st.is_open
                     else f"откроется {fmt_dt(a.opens_at)} МСК")
            tail = f"Максимум {a.max_score} баллов. Формат сдачи: {a.submit_format}."
            if st.status == "revision":
                lines.append(f"- {a.id} «{a.title}» (модуль {a.module}): {state}; исходный дедлайн "
                             f"{fmt_dt(st.deadline)} МСК (к доработке не относится). {tail} Статус у студента: "
                             f"{STATUS_LABELS['revision']} — {revision_rule()}.")
                continue
            personal = " (личный дедлайн после одобренного переноса)" if st.personal else ""
            status = f" Статус у студента: {STATUS_LABELS.get(st.status, st.status)}." if st.status else ""
            lines.append(f"- {a.id} «{a.title}» (модуль {a.module}): {state}; дедлайн {fmt_dt(st.deadline)} МСК"
                         f"{personal} — {st.phrase}. {tail}{status}")
        events = self.upcoming(now, module, events_limit)
        if events:
            lines.append("Ближайшие события:")
            lines += [_event_line(e) for e in events]
        past = self.recent(now)
        if past:
            lines.append(f"Прошедшие события за последние {RECENT_DAYS} {plural(RECENT_DAYS, DAYS)}:")
            lines += [_event_line(e) for e in past]
            recorded = [e for e in past if e.type in RECORDED_TYPES]
            if recorded:
                hours = int(RECORDING_DELAY.total_seconds() // 3600)
                lines.append("Записи прошедших вебинаров — срок посчитан системой по правилу курса (запись появляется "
                             f"в LMS не позже чем через {hours} {plural(hours, HOURS)} после окончания эфира), "
                             "называй его как есть:")
                lines += [recording_line(e, now) for e in recorded]
        if student:
            used = "использован" if student.extension_used else "не использован"
            lines.append(f"Студент {student.id}, уровень {LEVEL_LABELS.get(student.level, student.level)}; "
                         f"перенос дедлайна в курсе: {used}.")
        return "\n".join(lines)


def event_when(event: Event) -> str:
    """«ср 18.11.2026 19:00-20:30» — с концом, если известна длительность, иначе только начало."""
    start = fmt_dt(event.starts_at)
    if not event.duration_min:
        return start
    end = (event.starts_at + timedelta(minutes=event.duration_min)).astimezone(MSK)
    return f"{start}-{end:%H:%M}"


def _event_line(event: Event) -> str:
    duration = f" ({event.duration_min} мин)" if event.duration_min else ""
    return f"- {fmt_dt(event.starts_at)} МСК — {event.title}{duration}"


def recording_due(event: Event) -> datetime | None:
    """Последний момент появления записи по правилу курса: конец эфира + RECORDING_DELAY. None — у события записи
    нет (не вебинар) или длительность эфира неизвестна (конец не посчитать)."""
    if event.type not in RECORDED_TYPES or not event.duration_min:
        return None
    return event.starts_at + timedelta(minutes=event.duration_min) + RECORDING_DELAY


def recording_line(event: Event, now: datetime) -> str:
    """Строка фактов LMS о записи прошедшего вебинара: до какого момента она появится и наступил ли этот срок."""
    head = f"- {event.title} ({event_when(event)} МСК)"
    due = recording_due(event)
    if due is None:
        return f"{head}: длительность эфира в LMS не указана — точный срок появления записи не посчитать"
    if now < due:
        return (f"{head}: запись появится в LMS не позже {fmt_dt(due)} МСК; этот срок ещё не наступил "
                f"(до него {duration_text(due - now)})")
    return f"{head}: запись должна была появиться в LMS не позже {fmt_dt(due)} МСК; этот срок уже прошёл"


def make_backend(lms_url: str, lms_dir: Path, timeout: float = 5.0) -> LMSBackend:
    """LMS_URL задан — HTTP-заглушка, иначе файлы."""
    return HttpBackend(lms_url, timeout) if lms_url else FileBackend(lms_dir)


def export(client: LMSClient, out_dir: Path, fmt: str = "csv") -> list[Path]:
    """Выгрузка расписания, заданий и студентов. CSV — в UTF-8 с BOM, чтобы Excel открыл кириллицу."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"schedule": client.schedule(), "assignments": client.assignments(),
                            "students": client.students()}
    written: list[Path] = []
    for kind in KINDS:
        if fmt in ("csv", "both"):
            header, table = rows(kind, data["schedule"], data["assignments"], data["students"])
            path = out_dir / f"{kind}.csv"
            path.write_text(to_csv(header, table), encoding="utf-8-sig")
            written.append(path)
        if fmt in ("json", "both"):
            path = out_dir / f"{kind}.json"
            payload = [item.model_dump(mode="json") for item in data[kind]]
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    from .config import load_settings

    settings = load_settings()
    parser = argparse.ArgumentParser(prog="python -m curator.lms", description="Данные заглушки LMS")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="выгрузить расписание, задания и студентов")
    exp.add_argument("--out", type=Path, default=settings.exports_dir)
    exp.add_argument("--format", choices=("csv", "json", "both"), default="csv")
    show = sub.add_parser("show", help="показать дедлайны и ближайшие события на сейчас")
    show.add_argument("--student", default=None)
    show.add_argument("--module", type=int, default=None)
    args = parser.parse_args(argv)
    client = LMSClient(make_backend(settings.lms_url, settings.lms_dir, settings.lms_timeout_s))
    try:
        if args.command == "export":
            for path in export(client, args.out, args.format):
                print(path)
            return 0
        print(client.facts(settings.now(), args.module, client.student(args.student)))
        return 0
    except LMSError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
