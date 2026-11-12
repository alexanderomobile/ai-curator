"""HTTP-заглушка LMS: тот же набор данных, что в data/lms, по REST, в JSON или CSV.

    uvicorn curator.lms_stub_api:app --host 127.0.0.1 --port 8502
    LMS_URL=http://127.0.0.1:8502 streamlit run app/streamlit_app.py   # куратор берёт данные по HTTP

GET /api/schedule?module=2&format=csv, /api/assignments, /api/students, /api/students/{id}, /health.
Файлы читаются на каждый запрос: правка data/lms видна без перезапуска.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from .lms import FileBackend, LMSBackend, LMSError, rows, to_csv

Format = Literal["json", "csv"]


def _respond(kind: str, items: list[Any], fmt: Format, filename: str) -> Response:
    if fmt == "csv":
        header, table = rows(kind, **{kind: items})
        return Response("﻿" + to_csv(header, table), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{filename}.csv"'})
    return JSONResponse([item.model_dump(mode="json") for item in items])


def create_app(backend_factory: Callable[[], LMSBackend] | None = None) -> FastAPI:
    def _backend() -> LMSBackend:
        if backend_factory is not None:
            return backend_factory()
        from .config import load_settings

        return FileBackend(load_settings().lms_dir)

    api = FastAPI(title="LMS stub — Академия Данных", version="1.0",
                  description="Заглушка LMS для ИИ-куратора: вымышленные расписание, задания и студенты.")

    def _call(fn: Callable[[LMSBackend], Any]) -> Any:
        try:
            return fn(_backend())
        except LMSError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "backend": _backend().name}

    @api.get("/api/schedule")
    def schedule(module: int | None = Query(None, ge=0, le=4), format: Format = "json") -> Response:  # noqa: A002
        events = _call(lambda b: b.schedule())
        events = [e for e in events if module is None or e.module == module]
        return _respond("schedule", events, format, "schedule")

    @api.get("/api/assignments")
    def assignments(module: int | None = Query(None, ge=1, le=4), format: Format = "json") -> Response:  # noqa: A002
        items = _call(lambda b: b.assignments())
        items = [a for a in items if module is None or a.module == module]
        return _respond("assignments", items, format, "assignments")

    @api.get("/api/students")
    def students(format: Format = "json") -> Response:  # noqa: A002
        return _respond("students", _call(lambda b: b.students()), format, "students")

    @api.get("/api/students/{student_id}")
    def student(student_id: str, format: Format = "json") -> Response:  # noqa: A002
        found = _call(lambda b: b.student(student_id))
        if found is None:
            raise HTTPException(status_code=404, detail="студент не найден")
        if format == "csv":
            return _respond("students", [found], format, f"student-{student_id}")
        return JSONResponse(found.model_dump(mode="json"))

    return api


app = create_app()
