"""Данные заглушки LMS: расписание, задания со сроками (дни считает код, по МСК), демо-студенты; выгрузка JSON/CSV."""
from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import common  # noqa: E402
from curator.lms import (EVENT_LABELS, LEVEL_LABELS, STATUS_LABELS, Assignment, LMSError,  # noqa: E402
                         deadline_status, fmt_dt, rows, to_csv)

st.set_page_config(page_title="LMS — заглушка", page_icon="🗓️", layout="wide")
st.title("Данные LMS (заглушка)")

settings = common.settings()
client = common.current_lms()
now = settings.now()
source = f"HTTP-заглушка {settings.lms_url}" if settings.lms_url else f"файлы {settings.lms_dir.name}/*.json"
st.caption(f"Источник: {source}. Сейчас по Москве: {fmt_dt(now)}"
           + (" (время зафиксировано CURATOR_NOW)." if settings.now_override else ".")
           + " Все студенты и данные — вымышленные.")

try:
    schedule = client.schedule()
    assignments = client.assignments()
    students = client.students()
except LMSError as exc:
    st.error(f"LMS недоступна: {exc}")
    st.stop()


def downloads(kind: str, items: Sequence[BaseModel], key: str) -> None:
    header, table = rows(kind, schedule, assignments, students)
    payload = json.dumps([i.model_dump(mode="json") for i in items], ensure_ascii=False, indent=2)
    a, b, _ = st.columns([1, 1, 4])
    a.download_button("JSON", payload, file_name=f"{kind}.json", mime="application/json", key=f"{key}-json")
    b.download_button("CSV", "﻿" + to_csv(header, table), file_name=f"{kind}.csv", mime="text/csv",
                      key=f"{key}-csv")


def assignment_row(a: Assignment) -> dict[str, object]:
    status = deadline_status(a, now)
    return {"id": a.id, "модуль": a.module, "задание": a.title, "открывается": fmt_dt(a.opens_at),
            "дедлайн (МСК)": fmt_dt(a.deadline), "осталось": status.phrase,
            "заявка на перенос до": fmt_dt(status.apply_until) if status.can_apply else "окно закрыто",
            "баллы": a.max_score, "формат сдачи": a.submit_format, "методичка": a.guide_id}


tab_a, tab_s, tab_p, tab_api = st.tabs(["Задания и сроки", "Расписание", "Студенты", "API заглушки"])
with tab_a:
    st.caption("«Осталось» считает код от текущего времени по Москве — модели в промпт идёт уже готовый текст.")
    st.dataframe(pd.DataFrame([assignment_row(a) for a in assignments]), hide_index=True)
    downloads("assignments", assignments, "a")
with tab_s:
    st.dataframe(pd.DataFrame([{
        "когда (МСК)": fmt_dt(e.starts_at), "тип": EVENT_LABELS.get(e.type, e.type), "модуль": e.module,
        "событие": e.title, "мин": e.duration_min, "уже прошло": e.starts_at < now,
    } for e in schedule]), hide_index=True)
    downloads("schedule", schedule, "s")
with tab_p:
    st.dataframe(pd.DataFrame([{
        "id": s.id, "имя": s.name, "уровень": LEVEL_LABELS.get(s.level, s.level),
        **{k: STATUS_LABELS.get(v, v) for k, v in s.assignments.items()},
        "перенос": (f"{s.extension.assignment_id} до {fmt_dt(s.extension.new_deadline, weekday=False)}"
                    if s.extension else ("использован" if s.extension_used else "нет")),
    } for s in students]), hide_index=True)
    downloads("students", students, "p")
with tab_api:
    st.markdown(
        "Заглушка LMS на FastAPI отдаёт те же данные по HTTP (JSON или CSV):\n\n"
        "```bash\n"
        "uvicorn curator.lms_stub_api:app --host 127.0.0.1 --port 8502\n"
        "LMS_URL=http://127.0.0.1:8502 streamlit run app/streamlit_app.py\n"
        "python -m curator.lms export --format both   # выгрузка в data/lms/exports\n"
        "```\n\n"
        "| Метод | Что отдаёт |\n|---|---|\n"
        "| `GET /api/schedule?module=2&format=csv` | расписание (вебинары, разборы, защиты) |\n"
        "| `GET /api/assignments?module=2` | задания со сроками |\n"
        "| `GET /api/students` | демо-студенты |\n"
        "| `GET /api/students/{id}` | один студент (404, если нет) |\n"
        "| `GET /health` | проверка работы |"
    )
