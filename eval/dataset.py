"""Набор вопросов оценки и общие помощники eval/: чтение и проверка набора, JSON и JSONL, папки прогонов.

Набор — JSON-массив объектов (поля подробно — eval/README.md):

    {"id": "d01", "question": "...", "intent": "deadline", "level": "auto", "student": "s1" | null,
     "module": 1..4 | null, "history": [{"q": "...", "a": "..."}], "answerable": true,
     "expected_points": ["..."], "forbidden": ["..."], "expected_sources": ["L2.3", "faq", "lms"], "notes": "..."}

Ошибка в наборе останавливает прогон до первого платного вызова: текст ошибки называет вопрос и поле.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent
RUNS_DIR = EVAL_DIR / "runs"

# Сценарии набора: восемь сценариев анализа запроса куратора и no_answer — ответа нет в материалах курса.
INTENTS = ("deadline", "difficulty", "material", "recommendation", "organizational", "grading_request",
           "schedule_change", "off_topic", "no_answer")
LEVEL_MODES = ("auto", "beginner", "advanced")
SYSTEMS = ("curator", "baseline")


class DatasetError(ValueError):
    """Набор не читается или поле не того вида; текст — какой вопрос и что поправить."""


@dataclass(frozen=True)
class HistoryTurn:
    q: str
    a: str


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str
    intent: str
    level: str = "auto"
    student: str | None = None
    module: int | None = None
    history: tuple[HistoryTurn, ...] = ()
    answerable: bool = True
    expected_points: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    expected_sources: tuple[str, ...] = ()
    notes: str = ""


def _str_list(data: dict[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = data.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in cast(list[Any], value)):
        raise DatasetError(f"{where}: поле {key} — список строк")
    return tuple(v.strip() for v in cast(list[str], value) if v.strip())


def parse_item(raw: Any, where: str) -> EvalItem:
    if not isinstance(raw, dict):
        raise DatasetError(f"{where}: ожидается объект с полями id, question, intent …")
    data = cast(dict[str, Any], raw)
    item_id = data.get("id")
    if not isinstance(item_id, str) or not item_id.strip():
        raise DatasetError(f"{where}: нет id (непустая строка)")
    item_id = item_id.strip()
    where = f"{where} ({item_id})"
    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        raise DatasetError(f"{where}: нет question (непустая строка)")
    intent = data.get("intent")
    if intent not in INTENTS:
        raise DatasetError(f"{where}: intent {intent!r} — один из: {', '.join(INTENTS)}")
    level = data.get("level") or "auto"
    if level not in LEVEL_MODES:
        raise DatasetError(f"{where}: level {level!r} — один из: {', '.join(LEVEL_MODES)}")
    student = data.get("student")
    if student is not None and (not isinstance(student, str) or not student.strip()):
        raise DatasetError(f"{where}: student — строка вроде \"s1\" или null")
    module = data.get("module")
    if module is not None and (isinstance(module, bool) or not isinstance(module, int) or not 1 <= module <= 4):
        raise DatasetError(f"{where}: module — число 1-4 или null")
    history_raw = data.get("history") or []
    if not isinstance(history_raw, list):
        raise DatasetError(f"{where}: history — список объектов {{\"q\": ..., \"a\": ...}}")
    history: list[HistoryTurn] = []
    for n, turn in enumerate(cast(list[Any], history_raw), 1):
        t = cast(dict[str, Any], turn) if isinstance(turn, dict) else {}
        q, a = t.get("q"), t.get("a")
        if not isinstance(q, str) or not isinstance(a, str) or not q.strip():
            raise DatasetError(f"{where}: history[{n}] — объект {{\"q\": \"вопрос\", \"a\": \"ответ\"}}")
        history.append(HistoryTurn(q.strip(), a.strip()))
    answerable = data.get("answerable", True)
    if not isinstance(answerable, bool):
        raise DatasetError(f"{where}: answerable — true или false")
    notes = data.get("notes") or ""
    if not isinstance(notes, str):
        raise DatasetError(f"{where}: notes — строка")
    return EvalItem(
        id=item_id, question=question.strip(), intent=str(intent), level=str(level),
        student=student.strip() if isinstance(student, str) else None, module=module, history=tuple(history),
        answerable=answerable, expected_points=_str_list(data, "expected_points", where),
        forbidden=_str_list(data, "forbidden", where), expected_sources=_str_list(data, "expected_sources", where),
        notes=notes.strip(),
    )


def parse_items(raw: Any, where: str = "набор") -> list[EvalItem]:
    if not isinstance(raw, list):
        raise DatasetError(f"{where}: ожидается JSON-массив вопросов")
    items = [parse_item(entry, f"{where}, вопрос №{n}") for n, entry in enumerate(cast(list[Any], raw), 1)]
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise DatasetError(f"{where}: id {item.id!r} повторяется")
        seen.add(item.id)
    if not items:
        raise DatasetError(f"{where}: в наборе нет вопросов")
    return items


def load_items(path: Path) -> list[EvalItem]:
    try:
        raw = read_json(path)
    except (OSError, ValueError) as exc:
        raise DatasetError(f"{path.name}: не читается как JSON ({exc})") from exc
    return parse_items(raw, path.name)


# --- файлы

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}, строка {n}: ожидается объект")
        rows.append(cast(dict[str, Any], value))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    """Путь для meta.json относительно корня проекта: в публичный репозиторий не попадает путь чужого диска."""
    try:
        return Path(os.path.relpath(path.resolve(), ROOT_DIR)).as_posix()
    except ValueError:  # другой диск на Windows
        return path.name


def resolve_run_dir(value: str) -> Path:
    """Папка прогона: путь к ней или имя прогона в eval/runs."""
    path = Path(value)
    if path.is_dir():
        return path
    return RUNS_DIR / value


def run_items(run_dir: Path, set_path: Path | None = None) -> list[EvalItem]:
    """Вопросы прогона: из --set или из набора, записанного в meta.json прогона."""
    if set_path is None:
        meta_path = run_dir / "meta.json"
        if not meta_path.is_file():
            raise DatasetError(f"в {run_dir} нет meta.json — укажите набор через --set")
        meta = cast(dict[str, Any], read_json(meta_path))
        dataset = cast(dict[str, Any], meta.get("dataset") or {})
        set_path = ROOT_DIR / str(dataset.get("path", ""))
        if not set_path.is_file():
            raise DatasetError(f"набор {dataset.get('path')!r} из meta.json не найден — укажите его через --set")
    return load_items(set_path)
