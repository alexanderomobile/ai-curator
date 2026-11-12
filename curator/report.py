"""Отчёт по вопросам студентов за период — markdown для преподавателя и методиста.

    python -m curator.report --days 7                       # в консоль
    python -m curator.report --days 30 --out report.md --source live
    python -m curator.report --days 7 --include-live-texts  # примеры вопросов живых студентов (обезличенные)

Разделы: сводка, темы, сценарии, модули, пробелы в контенте, сложные лекции, отказы и ограничения, оценки ответов,
задержка и стоимость по дням, что сделать с контентом. Цифры — из журнала (curator.analytics).

Период — последние N дней по настоящим часам (CURATOR_NOW на отчёт не влияет: журнал пишет настоящее время).
Журнал открывается только для чтения; если файла нет — ошибка с путём, а не отчёт из нулей. На сервере пути службы
лежат в /etc/ai-curator.paths.env:  set -a; . /etc/ai-curator.paths.env; set +a; python -m curator.report --days 7
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import analytics as an
from .config import MSK, SOURCES

SECTIONS = ("Сводка", "Темы", "Сценарии", "Модули", "Пробелы в контенте", "Сложные лекции",
            "Отказы и ограничения", "Оценки ответов", "Задержка и стоимость по дням", "Что сделать с контентом")


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_нет данных_"
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f} %"


def _sec(ms: float | None) -> str:
    return "—" if ms is None else f"{ms / 1000:.1f} с"


def recommendations(rows: list[an.Row], titles: dict[str, str]) -> list[str]:
    """Что сделать с контентом — правилами по цифрам отчёта, без модели."""
    tips: list[str] = []

    def topic_name(slug: str) -> str:
        return f"{titles[slug]} ({slug})" if titles.get(slug) else slug

    for gap in an.content_gaps(rows)[:3]:
        tips.append(f"Тема «{topic_name(gap['topic'])}» (модуль {gap['module']}): {gap['questions']} вопр. без ответа "
                    "в материалах — дополнить FAQ или лекцию и пересобрать индекс (`python -m curator.ingest`).")
    for lec in an.hard_lectures(rows, titles)[:2]:
        name = f"{lec['lecture']} «{lec['title']}»" if lec["title"] else str(lec["lecture"])
        tips.append(f"Лекция {name}: {lec['questions']} вопр. «трудно» — проверить объяснение и раздел "
                    "«Если сложно».")
    for topic in [t for t in an.by_topic(rows, titles) if t["negative"] >= 2][:3]:
        tips.append(f"Тема «{topic_name(topic['topic'])}»: {topic['negative']} 👎 — просмотреть ответы куратора по ней.")
    k = an.kpis(rows)
    if k["rules_share"] is not None and k["rules_share"] > 0.1:
        tips.append(f"{_pct(k['rules_share'])} запросов разобраны правилами, а не моделью — проверить доступность "
                    "OpenAI и таймауты.")
    if k["limited"]:
        tips.append(f"Лимиты демо сработали {k['limited']} раз — если это живые студенты, пересмотреть лимиты.")
    if not tips:
        tips.append("Явных пробелов не видно: вопросов без ответа и жалоб за период нет.")
    return tips


def build_report(rows: list[an.Row], *, days: int, now: datetime, titles: dict[str, str] | None = None,
                 include_live_texts: bool = False, sources: Sequence[str] = ()) -> str:
    titles = titles or {}
    start = (now - timedelta(days=days)).astimezone(MSK)
    counts = Counter(r.source for r in rows)
    by_source = ", ".join(f"{s} {counts.get(s, 0)}" for s in SOURCES)
    k = an.kpis(rows)
    selected = f" (в отчёте: {', '.join(sources)})." if sources else "."
    texts = "включены." if include_live_texts else "не показываются — только eval/demo."
    out: list[str] = [
        f"# Отчёт ИИ-куратора: вопросы студентов за {days} дн.",
        "",
        f"Период: {start:%d.%m.%Y %H:%M} — {now.astimezone(MSK):%d.%m.%Y %H:%M} (МСК). "
        f"Записей по источникам: {by_source}{selected} Вопросы обезличены; тексты живых студентов {texts}",
        "",
        f"## 1. {SECTIONS[0]}",
        "",
        md_table(["Показатель", "Значение"], [
            ["Вопросов", k["total"]],
            ["Сессий", k["sessions"]],
            ["Ответ по материалам курса", _pct(k["answered_share"])],
            ["Без ответа в материалах (пробелы)", k["no_context"]],
            ["Отказы по ограничениям (оценка, перенос, не по курсу)", k["refusals"]],
            ["Срабатывания лимитов демо", k["limited"]],
            ["Оценок ответов / доля 👍", f"{k['rated']} / {_pct(k['positive_share'])}"],
            ["Задержка: медиана / p95", f"{_sec(k['median_latency_ms'])} / {_sec(k['p95_latency_ms'])}"],
            ["Токены / стоимость", f"{k['tokens']} / ${k['cost_usd']:.4f}"],
            ["Анализ запроса правилами (сбой модели)", _pct(k["rules_share"])],
        ]),
        "",
        f"## 2. {SECTIONS[1]}",
        "",
        md_table(["Тема", "Название", "Модуль", "Вопросов", "Без ответа", "«Трудно»", "👎"],
                 [[t["topic"], t["title"], t["module"], t["questions"], t["no_context"], t["difficulty"],
                   t["negative"]] for t in an.by_topic(rows, titles)]),
        "",
        f"## 3. {SECTIONS[2]}",
        "",
        md_table(["Сценарий", "Вопросов", "Ответ по материалам", "👎"],
                 [[i["label"], i["questions"], _pct(i["answered"]), i["negative"]] for i in an.by_intent(rows)]),
        "",
        f"## 4. {SECTIONS[3]}",
        "",
        md_table(["Модуль", "Вопросов", "Без ответа"],
                 [[m["module"], m["questions"], m["no_context"]] for m in an.by_module(rows)]),
        "",
        f"## 5. {SECTIONS[4]}",
        "",
        "Вопросы, по которым поиск не нашёл материалов выше порога близости.",
        "",
        md_table(["Тема", "Модуль", "Вопросов", "Примеры"],
                 [[g["topic"], g["module"], g["questions"], "; ".join(g["examples"]) or "—"]
                  for g in an.content_gaps(rows, include_live_texts)]),
        "",
        f"## 6. {SECTIONS[5]}",
        "",
        "Лекции, на которые ссылались ответы на вопросы «мне трудно».",
        "",
        md_table(["Лекция", "Название", "Вопросов", "👎"],
                 [[h["lecture"], h["title"], h["questions"], h["negative"]] for h in an.hard_lectures(rows, titles)]),
        "",
        f"## 7. {SECTIONS[6]}",
        "",
        md_table(["Причина", "Код", "Сколько"], [[r["label"], r["refusal"], r["count"]] for r in an.refusals(rows)]),
        "",
        f"## 8. {SECTIONS[7]}",
        "",
        md_table(["Сценарий", "👍", "👎", "Доля 👍"],
                 [[r["label"], r["up"], r["down"], _pct(r["positive_share"])] for r in an.ratings(rows)]),
        "",
        f"## 9. {SECTIONS[8]}",
        "",
        md_table(["День", "Вопросов", "С ответом", "Без ответа", "Отказы", "Стоимость, $"],
                 [[f"{d['day']:%d.%m.%Y}", d["questions"], d["answered"], d["no_context"], d["refused"],
                   f"{d['cost_usd']:.4f}"] for d in an.by_day(rows)]),
        "",
        f"## 10. {SECTIONS[9]}",
        "",
        *[f"- {tip}" for tip in recommendations(rows, titles)],
        "",
    ]
    return "\n".join(out)


def kb_titles(kb_dir: Path) -> dict[str, str]:
    """Названия тем и лекций для отчёта; без базы знаний отчёт строится, просто без названий."""
    from .kb import KBFormatError, load_kb

    try:
        kb = load_kb(kb_dir, strict=False)
    except KBFormatError:
        return {}
    return {**{slug: t.label for slug, t in kb.topics().items()}, **kb.titles()}


def main(argv: list[str] | None = None) -> int:
    from .config import load_settings
    from .journal import Journal, JournalMissing

    parser = argparse.ArgumentParser(prog="python -m curator.report", description="Отчёт по вопросам за период")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--source", default="", help="через запятую: live,eval,demo (по умолчанию все)")
    parser.add_argument("--out", type=Path, default=None, help="файл .md (по умолчанию — вывод в консоль)")
    parser.add_argument("--include-live-texts", action="store_true",
                        help="показать примеры вопросов живых студентов (обезличенные)")
    args = parser.parse_args(argv)
    settings = load_settings()
    sources = tuple(s.strip() for s in args.source.split(",") if s.strip())
    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        print(f"неизвестный источник: {', '.join(unknown)} (есть {', '.join(SOURCES)})", file=sys.stderr)
        return 2
    try:
        journal = Journal.open_readonly(settings.journal_path)
    except JournalMissing as exc:
        print(exc, file=sys.stderr)
        return 2
    now = an.wall_now()
    rows = an.load(journal, args.days, now, sources)
    text = build_report(rows, days=args.days, now=now, titles=kb_titles(settings.kb_dir),
                        include_live_texts=args.include_live_texts, sources=sources)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"отчёт: {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
