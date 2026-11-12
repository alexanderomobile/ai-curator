"""Дашборд аналитики запросов: KPI, вопросы по дням, сценариям, модулям и темам, пробелы в контенте,
сложные лекции, отказы, оценки, задержка, стоимость. На публичном демо тексты вопросов живых посетителей
не показываются — только агрегаты; тексты видны для прогонов оценки (eval) и кнопок-примеров (demo).
Период — последние N дней по настоящим часам: CURATOR_NOW сдвигает только «сегодня» курса, а журнал пишет
настоящее время."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from app import common  # noqa: E402
from curator import analytics as an  # noqa: E402
from curator.config import SOURCES  # noqa: E402
from curator.report import build_report  # noqa: E402

st.set_page_config(page_title="Аналитика запросов", page_icon="📊", layout="wide")
st.title("Аналитика запросов")

settings = common.settings()
c1, c2 = st.columns([1, 2])
days = c1.selectbox("Период", [7, 30, 90], format_func=lambda d: f"{d} дней")
sources = c2.multiselect("Источник записей", list(SOURCES), default=list(SOURCES),
                         help="live — посетители демо, eval — прогоны оценки, demo — кнопки-примеры в чате")
now = an.wall_now()
rows = an.load(common.current_journal(), days, now, tuple(sources))
kb = common.current_kb()
titles = {**{slug: t.label for slug, t in kb.topics().items()}, **kb.titles()} if kb else {}

if not rows:
    st.info("За выбранный период записей нет. Задайте несколько вопросов в чате или запустите оценку (eval/) — "
            "журнал наполнится, и здесь появятся графики.")
    st.stop()
if not an.questions(rows):
    # Только отказы по лимитам демо (длинный вопрос, вопросов в час…): до куратора не дошёл ни один вопрос, графикам
    # строить нечего — пустая таблица у plotly без столбцов падала бы.
    st.info("За выбранный период до куратора не дошёл ни один вопрос: в журнале только отказы по лимитам демо.")
    st.dataframe(pd.DataFrame(an.refusals(rows)), hide_index=True,
                 column_config={"refusal": "Код", "label": "Причина", "count": "Сколько"})
    st.stop()

k = an.kpis(rows)


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f} %"


m = [*st.columns(3), *st.columns(3)]  # два ряда: на узком экране подписи не обрезаются
m[0].metric("Вопросов", k["total"], help=f"Сессий: {k['sessions']}", border=True)
m[1].metric("Ответ по материалам", pct(k["answered_share"]), border=True)
m[2].metric("Нет в материалах", k["no_context"], help="Пробелы в контенте: поиск не нашёл материалов", border=True)
m[3].metric("Отказы по правилам", k["refusals"], help="Оценка, перенос срока, вопрос не по курсу", border=True)
m[4].metric("Доля 👍", pct(k["positive_share"]), help=f"Оценок: {k['rated']}", border=True)
latency = k["median_latency_ms"]
m[5].metric("Задержка, медиана", "—" if latency is None else f"{latency / 1000:.1f} с",
            help=f"Стоимость за период: ${k['cost_usd']:.4f}", border=True)

blue = [common.SERIES[0]]
left, right = st.columns(2)
daily = pd.DataFrame([{**d, "day": f"{d['day']:%d.%m}"} for d in an.by_day_outcome(rows)])
fig = px.bar(daily, x="day", y="questions", color="outcome", title="Вопросы по дням",
             category_orders={"outcome": list(an.OUTCOMES)}, color_discrete_sequence=list(common.SERIES[:4]),
             labels={"day": "День", "questions": "Вопросов", "outcome": "Исход"})
fig.update_xaxes(type="category")
left.plotly_chart(common.style(fig), key="by_day")

intents = pd.DataFrame(an.by_intent(rows))
fig = px.bar(intents, x="questions", y="label", orientation="h", title="Сценарии", color_discrete_sequence=blue,
             labels={"questions": "Вопросов", "label": "Сценарий"})
fig.update_yaxes(categoryorder="total ascending")
right.plotly_chart(common.style(fig, counts="x"), key="by_intent")

left, right = st.columns(2)
# На оси — понятное название темы («FAQ: Перенос срока»), slug — в подсказке.
topics = pd.DataFrame([{**t, "name": t["title"] or t["topic"]} for t in an.by_topic(rows, titles)[:15]])
fig = px.bar(topics, x="questions", y="name", orientation="h", title="Темы (топ-15)", hover_data=["topic"],
             color_discrete_sequence=blue, labels={"questions": "Вопросов", "name": "Тема", "topic": "Код темы"})
fig.update_yaxes(categoryorder="total ascending")
left.plotly_chart(common.style(fig, height=420, counts="x"), key="by_topic")

modules = pd.DataFrame([{**m, "module": "без модуля" if m["module"] == "—" else m["module"]}
                        for m in an.by_module(rows)])
fig = px.bar(modules, x="module", y="questions", title="Модули", color_discrete_sequence=blue,
             hover_data=["no_context"], labels={"module": "Модуль", "questions": "Вопросов",
                                                "no_context": "Без ответа"})
fig.update_xaxes(type="category")
right.plotly_chart(common.style(fig, height=420), key="by_module")

st.subheader("Пробелы в контенте")
st.caption("Вопросы, на которые в базе знаний не нашлось материалов: что дописать в FAQ или лекции.")
gaps = an.content_gaps(rows, settings.show_live_texts)
if gaps:
    st.dataframe(pd.DataFrame([{**g, "topic": titles.get(g["topic"]) or g["topic"], "examples": "; ".join(g["examples"])}
                               for g in gaps]), hide_index=True,
                 column_config={"topic": "Тема", "module": "Модуль", "questions": "Вопросов",
                                "examples": "Примеры вопросов"})
else:
    st.success("Вопросов без ответа за период нет.")

left, right = st.columns(2)
with left:
    st.subheader("Сложные лекции")
    st.caption("На какие лекции ссылались ответы на «мне трудно».")
    hard = an.hard_lectures(rows, titles)
    if hard:
        st.dataframe(pd.DataFrame(hard), hide_index=True,
                     column_config={"lecture": "Лекция", "title": "Название", "questions": "Вопросов",
                                    "negative": "👎"})
    else:
        st.info("Вопросов «трудно» за период нет.")
with right:
    refused = pd.DataFrame(an.refusals(rows))
    if not refused.empty:
        fig = px.bar(refused, x="count", y="label", orientation="h", title="Отказы и ограничения",
                     color_discrete_sequence=blue, labels={"count": "Сколько", "label": "Причина"})
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(common.style(fig, counts="x"), key="refusals")
    else:
        st.info("Отказов за период нет.")

left, right = st.columns(2)
rated = an.ratings(rows)
if rated:
    long = pd.DataFrame([{"label": r["label"], "оценка": mark, "ответов": r[key]}
                         for r in rated for mark, key in (("👍", "up"), ("👎", "down"))])
    fig = px.bar(long, x="label", y="ответов", color="оценка", title="Оценки ответов по сценариям",
                 category_orders={"оценка": ["👍", "👎"]}, color_discrete_sequence=list(common.SERIES[:2]),
                 labels={"label": "Сценарий"})
    left.plotly_chart(common.style(fig), key="ratings")
else:
    left.info("Оценок 👍/👎 за период нет.")
seconds = an.latencies(rows)
if seconds:
    fig = px.histogram(pd.DataFrame({"секунд": seconds}), x="секунд", nbins=20, title="Задержка ответа",
                       color_discrete_sequence=blue)
    right.plotly_chart(common.style(fig), key="latency")  # ось Y — число ответов

cost = pd.DataFrame([{**d, "day": f"{d['day']:%d.%m}"} for d in an.by_day(rows)])
fig = px.bar(cost, x="day", y="cost_usd", title="Стоимость по дням, $", color_discrete_sequence=blue,
             labels={"day": "День", "cost_usd": "$"})
fig.update_xaxes(type="category")
st.plotly_chart(common.style(fig, height=260, counts=None), key="cost")

st.subheader("Вопросы")
if settings.show_live_texts:
    st.caption("Показаны тексты всех вопросов после обезличивания (SHOW_LIVE_TEXTS включён).")
else:
    st.caption("Тексты вопросов живых посетителей на публичном демо не показываются — только eval и demo.")
table = an.questions_table(rows, settings.show_live_texts)
if table:
    frame = pd.DataFrame(table)
    # Пустые поля (нет модуля, темы, оценки) — пустая ячейка, а не «None» в таблице.
    st.dataframe(frame.astype(object).where(frame.notna(), ""), hide_index=True)
else:
    st.info("Текстов для показа нет: за период были только вопросы живых посетителей.")

report = build_report(rows, days=days, now=now, titles=titles, include_live_texts=settings.show_live_texts,
                      sources=tuple(sources))
st.download_button("Скачать отчёт (markdown)", report, file_name=f"report-{days}d.md", mime="text/markdown")
