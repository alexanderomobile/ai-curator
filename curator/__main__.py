"""Вопрос куратору из терминала — для проверки без интерфейса.

    python -m curator ask "Когда дедлайн второго ДЗ?" --student s3
    python -m curator ask "Не понимаю groupby" --level beginner --module 2 --source demo --debug
"""
from __future__ import annotations

import argparse
import sys
import uuid

from .config import SOURCES, ConfigError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m curator", description="ИИ-куратор курса")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="задать вопрос")
    ask.add_argument("question")
    ask.add_argument("--level", choices=("auto", "beginner", "advanced"), default="auto")
    ask.add_argument("--module", type=int, choices=(1, 2, 3, 4), default=None)
    ask.add_argument("--type", dest="resource_type", choices=("lecture", "guide", "faq", "overview"), default=None)
    ask.add_argument("--student", default=None)
    ask.add_argument("--source", choices=SOURCES, default="demo")
    ask.add_argument("--debug", action="store_true", help="показать сценарий, фильтр, токены")
    args = parser.parse_args(argv)

    from .factory import build_services

    sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]
    try:
        services = build_services()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not services.index.ok:
        print(f"Внимание: {services.index.message}", file=sys.stderr)
    answer = services.curator.ask(args.question, session_id=f"cli-{uuid.uuid4()}", level_mode=args.level,
                                  module=args.module, resource_type=args.resource_type, student_id=args.student,
                                  source=args.source)
    print(answer.text)
    if args.debug:
        r = answer.retrieval
        print("\n---")
        print(f"сценарий: {answer.intent} ({answer.analysis_method}{', guard ' + answer.guard if answer.guard else ''})"
              f"; модуль: {answer.module}; тема: {answer.topic}; уровень: {answer.level} ({answer.level_source})")
        if r is not None:
            print(f"поиск: {r.filter_used.describe()}; ослаблен: {r.relaxed}; шаги: {' | '.join(r.steps)}; "
                  f"лучшая близость: {r.top_score}")
        print(f"отказ: {answer.refusal or '—'}; ссылки модели: {answer.cited}; токены: {answer.prompt_tokens}+"
              f"{answer.completion_tokens}; ${answer.cost_usd:.5f}; {answer.latency_ms} мс")
    return 0


if __name__ == "__main__":
    sys.exit(main())
