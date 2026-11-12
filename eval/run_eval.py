"""Прогон набора вопросов через куратора (полная система) и наивный RAG (eval/baseline.py).

    python -m eval.run_eval --set eval/sample.json --run sample-1
    python -m eval.run_eval --set eval/sets/dev.json --run dev-1 --systems curator
    python -m eval.run_eval --set eval/sample.json --dry-run     # проверить набор, ключ и индекс без вызовов API

Куратор собирается той же фабрикой, что и чат (curator.factory.build_services), с тремя отличиями:
«сейчас» зафиксировано (--now → Settings.now_override, тот же механизм, что CURATOR_NOW: дни до дедлайна,
окно заявки на перенос и «сегодня» в промпте считаются от него), журнал — свой файл прогона
(<папка прогона>/journal.db), записи помечены source=eval (лимиты демо на eval не действуют). Уровень, модуль
и студент — параметры ask(), как в интерфейсе; history из набора кладётся в память диалога новой сессии как
прошлые реплики. Наивный RAG ищет в том же хранилище Chroma теми же эмбеддингами.

Ответы дописываются в answers.jsonl по мере готовности: оборванный прогон продолжается тем же --run, готовые
пары «вопрос — система» не повторяются. В конце — checks.jsonl/checks.md (eval/checks.py) и summary.md/.json.
Стоимость — токены моделей ответа и анализа запроса по ценам из настроек; эмбеддинги не входят (у обеих систем
одни и те же, это доли цента).
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from curator.config import MSK, ConfigError, Settings, load_settings
from curator.curator import HITS_BY_INTENT, Curator, CuratorAnswer
from curator.ingest import index_status
from curator.memory import Turn
from eval import baseline as baseline_mod
from eval import checks
from eval.baseline import BaselineRAG, build_baseline, make_baseline_llm
from eval.dataset import (RUNS_DIR, SYSTEMS, DatasetError, EvalItem, append_jsonl, display_path, file_sha256,
                          load_items, read_json, read_jsonl, write_json, write_jsonl)

DEFAULT_NOW = "2026-11-12T12:00:00+03:00"
WORKERS = 4
RETRIES = 3  # повторов после первой попытки
BACKOFF_S = 2.0  # пауза перед повтором: 2, 4, 8 с
ERROR_CHARS = 300
_SECRET = re.compile(r"sk-[A-Za-z0-9_\-*]{4,}")

log = logging.getLogger("eval")

AnswerFn = Callable[[EvalItem, int], dict[str, Any]]


class Degraded(Exception):
    """Ответ есть, но собран в обход модели из-за сбоя (llm_error, search_error, анализ запроса правилами) — повторить."""

    def __init__(self, fields: dict[str, Any], reason: str):
        super().__init__(reason)
        self.fields = fields
        self.reason = reason


class FinishReasons(BaseCallbackHandler):
    """Почему модель ответа остановилась (stop — сама, length — упёрлась в потолок длины), по каждому вызову.
    Своя запись у каждого потока: куратор вызывает модель синхронно в потоке, который задал вопрос."""

    def __init__(self) -> None:
        super().__init__()
        self._local = threading.local()

    def start(self) -> None:
        self._local.reasons = []

    def take(self) -> list[str]:
        reasons = cast(list[str], getattr(self._local, "reasons", None) or [])
        self._local.reasons = None
        return reasons

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        reasons = cast("list[str] | None", getattr(self._local, "reasons", None))
        if reasons is None:
            return
        for generations in response.generations:
            for gen in generations:
                reason = (gen.generation_info or {}).get("finish_reason")
                message = getattr(gen, "message", None)
                if not reason and message is not None:
                    meta = getattr(message, "response_metadata", None) or {}
                    reason = cast(dict[str, Any], meta).get("finish_reason") if isinstance(meta, dict) else None
                reasons.append(str(reason or ""))


def tokens(prompt: int, completion: int) -> dict[str, int]:
    return {"prompt": prompt, "completion": completion, "total": prompt + completion}


def describe_error(exc: BaseException) -> str:
    """Тип и начало текста ошибки; всё похожее на ключ OpenAI вычищается."""
    text = _SECRET.sub("sk-***", " ".join(str(exc).split()))
    return f"{type(exc).__name__}: {text[:ERROR_CHARS]}" if text else type(exc).__name__


# --- системы

def curator_context_sources(answer: CuratorAnswer) -> list[dict[str, Any]]:
    """Что было в контексте модели: найденные фрагменты и данные LMS (показанные источники — отдельно).
    Фрагменты — столько, сколько куратор отдал модели в этом сценарии (HITS_BY_INTENT: у recommendation — 2)."""
    out: list[dict[str, Any]] = []
    if answer.retrieval is not None:
        limit = HITS_BY_INTENT.get(answer.intent)
        for hit in answer.retrieval.hits[:limit]:
            out.append({"kind": str(hit.metadata.get("resource_type", "")), "id": hit.doc_id, "section": hit.section,
                        "score": round(hit.score, 3)})
    if answer.lms_used:
        out.append({"kind": "lms", "id": "lms", "section": ""})
    return out


def curator_fields(answer: CuratorAnswer, finish: Sequence[str]) -> dict[str, Any]:
    r = answer.retrieval
    return {
        "answer": answer.text,
        "sources": [{"n": i.n, "kind": i.source.kind, "id": i.source.ref_id, "section": i.source.section,
                     "label": i.source.label()} for i in answer.sources],
        "intent_detected": answer.intent or None,
        "refusal": answer.refusal,
        "tokens": tokens(answer.prompt_tokens, answer.completion_tokens),
        "cost": round(answer.cost_usd, 6),
        "context_sources": curator_context_sources(answer),
        "finish_reasons": list(finish),
        "truncated": "length" in finish,
        "details": {
            "level": answer.level, "level_source": answer.level_source, "module": answer.module,
            "topic": answer.topic, "analysis_method": answer.analysis_method, "guard": answer.guard,
            "answered": answer.answered, "cited": answer.cited, "lms_used": answer.lms_used,
            "student_id": answer.student_id, "journal_id": answer.journal_id, "notes": list(answer.notes),
            "top_score": r.top_score if r else None, "filter": r.filter_used.describe() if r else "",
            "relaxed": r.relaxed if r else False, "outside_filter": r.outside if r else 0,
            "search_steps": list(r.steps) if r else [],
            "latency_internal_ms": answer.latency_ms,
        },
    }


def degraded_reason(answer: CuratorAnswer) -> str:
    if answer.refusal == "llm_error":
        return "llm_error: модель ответа не ответила, показан список материалов"
    if answer.refusal == "search_error":
        return "search_error: поиск (эмбеддинги) не ответил, показан шаблон «поиск недоступен»"
    note = next((n for n in answer.notes if n.startswith("анализ правилами")), "")
    return f"router: {note}" if note else ""


class CuratorSystem:
    """Вопрос набора → Curator.ask с параметрами интерфейса; своя сессия на каждую попытку."""

    def __init__(self, curator: Curator, run: str, finish: FinishReasons | None = None):
        self.curator = curator
        self.run = run
        self.finish = finish or FinishReasons()
        curator.llm.callbacks = [self.finish]

    def __call__(self, item: EvalItem, attempt: int) -> dict[str, Any]:
        # Новая сессия на попытку: иначе неудачная попытка осталась бы в памяти диалога как прошлая реплика.
        session = f"eval-{self.run}-{item.id}-{attempt}"
        for turn in item.history:
            self.curator.memory.add(session, Turn(turn.q, turn.a))
        self.finish.start()
        try:
            answer = self.curator.ask(item.question, session_id=session, level_mode=item.level, module=item.module,
                                      student_id=item.student, source="eval")
        finally:
            finish = self.finish.take()
            self.curator.memory.reset(session)
        fields = curator_fields(answer, finish)
        reason = degraded_reason(answer)
        if reason:
            raise Degraded(fields, reason)
        return fields


class BaselineSystem:
    """Вопрос набора → наивный RAG; history — прошлые реплики чата, уровень/модуль/студент ему неизвестны."""

    def __init__(self, rag: BaselineRAG):
        self.rag = rag

    def __call__(self, item: EvalItem, attempt: int) -> dict[str, Any]:
        answer = self.rag.answer(item.question, [(t.q, t.a) for t in item.history])
        if not answer.text:
            raise RuntimeError("пустой ответ модели")
        finish = [answer.finish_reason] if answer.finish_reason else []
        return {
            "answer": answer.text,
            "sources": [],  # блока «Источники» у наивного RAG нет
            "intent_detected": None,
            "refusal": None,
            "tokens": tokens(answer.prompt_tokens, answer.completion_tokens),
            "cost": round(answer.cost_usd, 6),
            "context_sources": answer.sources,
            "finish_reasons": finish,
            "truncated": answer.finish_reason == "length",
            "details": {},
        }


# --- прогон

def _row(item: EvalItem, system: str, fields: Mapping[str, Any], latency_ms: int, attempt: int) -> dict[str, Any]:
    """Строка answers.jsonl: сначала поля из ТЗ, потом подробности."""
    head = ("answer", "sources", "intent_detected", "refusal")
    row: dict[str, Any] = {"id": item.id, "system": system}
    row.update({k: fields.get(k) for k in head})
    row.update({"latency_ms": latency_ms, "tokens": fields.get("tokens") or tokens(0, 0),
                "cost": fields.get("cost") or 0.0})
    row.update({k: v for k, v in fields.items() if k not in row})
    row.update({"attempts": attempt, "error": None, "degraded": None, "errors": []})
    return row


def run_one(item: EvalItem, system: str, fn: AnswerFn, *, retries: int = RETRIES, backoff: float = BACKOFF_S,
            sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Один вопрос одной системой с повторами при сбое API. Не бросает исключений: итог — строка ответа,
    ответ «в обход модели» (degraded) после всех попыток или строка с error."""
    errors: list[str] = []
    fallback: dict[str, Any] | None = None
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            fields = fn(item, attempt)
        except Degraded as exc:
            errors.append(exc.reason)
            fallback = _row(item, system, exc.fields, int((time.perf_counter() - started) * 1000), attempt)
            fallback["degraded"] = exc.reason
        except Exception as exc:  # сеть, лимиты API, таймаут — повторяем
            errors.append(describe_error(exc))
        else:
            row = _row(item, system, fields, int((time.perf_counter() - started) * 1000), attempt)
            row["errors"] = errors
            return row
        if attempt < attempts:
            sleep(backoff * 2 ** (attempt - 1))
    if fallback is not None:
        fallback["attempts"], fallback["errors"] = attempts, errors
        return fallback
    return {"id": item.id, "system": system, "answer": "", "sources": [], "intent_detected": None, "refusal": None,
            "latency_ms": None, "tokens": tokens(0, 0), "cost": 0.0, "attempts": attempts,
            "error": errors[-1] if errors else "нет ответа", "degraded": None, "errors": errors}


def run_tasks(tasks: Sequence[tuple[EvalItem, str]], systems: Mapping[str, AnswerFn], *, workers: int = WORKERS,
              retries: int = RETRIES, backoff: float = BACKOFF_S, on_row: Callable[[dict[str, Any]], None] = lambda r: None,
              sleep: Callable[[float], None] = time.sleep) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="eval")
    try:
        futures: list[Future[dict[str, Any]]] = [
            pool.submit(run_one, item, system, systems[system], retries=retries, backoff=backoff, sleep=sleep)
            for item, system in tasks]
        for future in as_completed(futures):
            row = future.result()
            on_row(row)
            rows.append(row)
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)  # Ctrl+C: дождаться начатых, остальные не начинать
        raise
    pool.shutdown(wait=True)
    return rows


def ordered(rows: Iterable[dict[str, Any]], items: Sequence[EvalItem]) -> list[dict[str, Any]]:
    """Последняя строка на пару «вопрос — система», в порядке набора и систем."""
    order = {item.id: n for n, item in enumerate(items)}
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("id")) in order:
            latest[(str(row["id"]), str(row["system"]))] = row
    rank = {s: n for n, s in enumerate(SYSTEMS)}
    return sorted(latest.values(), key=lambda r: (order[str(r["id"])], rank.get(str(r["system"]), len(rank))))


def done_pairs(rows: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    """Готовые пары: ответ без ошибки (ответ «в обход модели» после всех попыток тоже готов)."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        latest[(str(row.get("id")), str(row.get("system")))] = row
    return {pair for pair, row in latest.items() if not row.get("error")}


# --- сводка

def percentile(values: Sequence[float], q: float) -> float | None:
    """Ближайший ранг: p50, p95 без интерполяции."""
    if not values:
        return None
    ranked = sorted(values)
    return ranked[max(0, math.ceil(q * len(ranked)) - 1)]


def run_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for system in sorted({str(r["system"]) for r in rows}, key=lambda s: SYSTEMS.index(s) if s in SYSTEMS else 9):
        mine = [r for r in rows if r["system"] == system]
        ok = [r for r in mine if not r.get("error")]
        latency = [float(r["latency_ms"]) for r in ok if r.get("latency_ms") is not None]
        cost = sum(float(r.get("cost") or 0.0) for r in mine)
        out[system] = {
            "answers": len(mine), "errors": len(mine) - len(ok),
            "degraded": sum(1 for r in mine if r.get("degraded")),
            "retried": sum(1 for r in mine if int(r.get("attempts") or 1) > 1),
            "truncated": sum(1 for r in ok if r.get("truncated")),
            "latency_ms": {"mean": round(sum(latency) / len(latency)) if latency else None,
                           "p50": percentile(latency, 0.5), "p95": percentile(latency, 0.95)},
            "tokens": {"prompt": sum(int((r.get("tokens") or {}).get("prompt", 0)) for r in mine),
                       "completion": sum(int((r.get("tokens") or {}).get("completion", 0)) for r in mine)},
            "cost_usd": round(cost, 6),
            "cost_per_answer_usd": round(cost / len(ok), 6) if ok else None,
            "intents_detected": dict(Counter(str(r.get("intent_detected")) for r in ok if r.get("intent_detected"))),
        }
    return out


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value / 1000:.1f} с"


def render_summary(meta: Mapping[str, Any], stats: Mapping[str, Any], checks_md: str) -> str:
    dataset = cast(Mapping[str, Any], meta.get("dataset") or {})
    lines = [f"# Прогон {meta.get('run')}", "",
             f"- Набор: `{dataset.get('path')}` ({dataset.get('items')} вопросов, sha256 {str(dataset.get('sha256'))[:12]})",
             f"- «Сейчас» для куратора: {meta.get('now')}",
             f"- Системы: {', '.join(cast(list[str], meta.get('systems') or []))}",
             f"- Модели: куратор {meta.get('curator', {}).get('chat_model')} (анализ запроса — "
             f"{meta.get('curator', {}).get('router_model')}), наивный RAG {meta.get('baseline', {}).get('model')}; "
             f"эмбеддинги {meta.get('embed_model')}",
             f"- Индекс: {meta.get('index', {}).get('collection')} (база {meta.get('index', {}).get('kb_version')}, "
             f"{meta.get('index', {}).get('state')})", "",
             "## Прогон", ""]
    systems = list(stats)
    rows = [
        ["Ответов", *(str(stats[s]["answers"]) for s in systems)],
        ["Ошибок (ответа нет)", *(str(stats[s]["errors"]) for s in systems)],
        ["Ответ в обход модели после всех попыток", *(str(stats[s]["degraded"]) for s in systems)],
        ["С повтором", *(str(stats[s]["retried"]) for s in systems)],
        ["Обрезано потолком длины", *(str(stats[s]["truncated"]) for s in systems)],
        ["Задержка, среднее", *(_ms(stats[s]["latency_ms"]["mean"]) for s in systems)],
        ["Задержка, p50 / p95", *(f"{_ms(stats[s]['latency_ms']['p50'])} / {_ms(stats[s]['latency_ms']['p95'])}"
                                  for s in systems)],
        ["Токены, вход / выход", *(f"{stats[s]['tokens']['prompt']} / {stats[s]['tokens']['completion']}"
                                   for s in systems)],
        ["Стоимость, $", *(f"{stats[s]['cost_usd']:.4f}" for s in systems)],
        ["Стоимость ответа, $", *("—" if stats[s]["cost_per_answer_usd"] is None
                                   else f"{stats[s]['cost_per_answer_usd']:.5f}" for s in systems)],
    ]
    lines += ["| Показатель | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n\n" + checks_md


def write_reports(run_dir: Path, items: Sequence[EvalItem], rows: Sequence[dict[str, Any]],
                  meta: Mapping[str, Any]) -> str:
    check_rows = checks.check_run(items, rows)
    checks_md = checks.write_checks(run_dir, check_rows)
    stats = run_summary(rows)
    write_json(run_dir / "summary.json", {"run": stats, "checks": checks.summary(check_rows)})
    text = render_summary(meta, stats, checks_md)
    (run_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


# --- настройки и сборка

def parse_now(value: str) -> datetime:
    moment = datetime.fromisoformat(value.strip())
    return moment if moment.tzinfo else moment.replace(tzinfo=MSK)


def eval_settings(base: Settings, now: datetime, run_dir: Path) -> Settings:
    """Настройки куратора для прогона: зафиксированное «сейчас», свой журнал, source=eval."""
    return replace(base, now_override=now, journal_path=run_dir / "journal.db", default_source="eval")


def build_systems(names: Sequence[str], settings: Settings, run: str) -> tuple[dict[str, AnswerFn], dict[str, Any]]:
    """Настоящие системы и сведения о них для meta.json. Наивный RAG берёт хранилище куратора, если тот собран:
    тот же индекс и те же эмбеддинги буквально."""
    from curator.factory import build_services

    systems: dict[str, AnswerFn] = {}
    info: dict[str, Any] = {}
    store = None
    if "curator" in names:
        services = build_services(settings)
        systems["curator"] = CuratorSystem(services.curator, run)
        store = services.curator.retriever.store
        info["problems"] = list(services.problems)
    if "baseline" in names:
        rag = BaselineRAG(store, make_baseline_llm(settings), price=settings.cost_usd) if store is not None \
            else build_baseline(settings)
        systems["baseline"] = BaselineSystem(rag)
    return systems, info


def make_meta(run: str, set_path: Path, items: Sequence[EvalItem], systems: Sequence[str], now: datetime,
              settings: Settings, index: Mapping[str, Any], workers: int, retries: int) -> dict[str, Any]:
    return {
        "run": run, "created_at": datetime.now(MSK).isoformat(timespec="seconds"),
        "now": now.isoformat(timespec="seconds"),
        "dataset": {"path": display_path(set_path), "sha256": file_sha256(set_path), "items": len(items)},
        "systems": list(systems), "workers": workers, "retries": retries,
        "curator": {"chat_model": settings.chat_model, "router_model": settings.router_model,
                    "temperature": settings.temperature, "max_tokens": settings.max_tokens, "top_k": settings.top_k,
                    "min_similarity": settings.min_similarity, "min_filtered_hits": settings.min_filtered_hits,
                    "history_turns": settings.history_turns},
        "baseline": {"model": baseline_mod.MODEL, "temperature": baseline_mod.TEMPERATURE,
                     "top_k": baseline_mod.TOP_K, "max_tokens": settings.max_tokens,
                     "system_prompt": baseline_mod.SYSTEM_PROMPT},
        "embed_model": settings.embed_model, "index": dict(index), "journal": "journal.db", "sessions": [],
    }


def select_items(items: Sequence[EvalItem], only: str, limit: int) -> list[EvalItem]:
    wanted = [x.strip() for x in only.split(",") if x.strip()]
    if wanted:
        unknown = sorted(set(wanted) - {i.id for i in items})
        if unknown:
            raise DatasetError(f"--only: в наборе нет id {', '.join(unknown)}")
        items = [i for i in items if i.id in wanted]
    return list(items[:limit]) if limit > 0 else list(items)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.run_eval", description="Прогон набора: куратор и наивный RAG")
    parser.add_argument("--set", type=Path, required=True, help="набор вопросов (JSON-массив, формат — eval/README.md)")
    parser.add_argument("--run", default="", help="имя прогона (по умолчанию <набор>-<дата>-<время>)")
    parser.add_argument("--out", type=Path, default=None, help="папка прогона (по умолчанию eval/runs/<run>)")
    parser.add_argument("--systems", default=",".join(SYSTEMS), help="curator,baseline или одна из них")
    parser.add_argument("--now", default=DEFAULT_NOW, help=f"«сейчас» для куратора, ISO (по умолчанию {DEFAULT_NOW})")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--retries", type=int, default=RETRIES, help="повторов при сбое API после первой попытки")
    parser.add_argument("--only", default="", help="только эти id через запятую")
    parser.add_argument("--limit", type=int, default=0, help="только первые N вопросов")
    parser.add_argument("--allow-stale-index", action="store_true",
                        help="прогонять на индексе, собранном по старой версии базы знаний")
    parser.add_argument("--dry-run", action="store_true", help="проверить набор, ключ и индекс; API не вызывать")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # pyright: ignore[reportAttributeAccessIssue]
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    if not names or any(s not in SYSTEMS for s in names):
        parser.error(f"--systems: из {', '.join(SYSTEMS)}")
    try:
        now = parse_now(args.now)
    except ValueError:
        parser.error(f"--now {args.now!r}: нужна дата-время ISO, например {DEFAULT_NOW}")
    try:
        all_items = load_items(args.set)
        items = select_items(all_items, args.only, args.limit)
    except DatasetError as exc:
        print(f"Набор: {exc}", file=sys.stderr)
        return 2
    try:
        base = load_settings()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not base.openai_api_key:
        print("Нет OPENAI_API_KEY: задайте его в .env в корне проекта.", file=sys.stderr)
        return 2
    status = index_status(base.kb_dir, base.index_dir, base.embed_model, base.chunk_size, base.chunk_overlap)
    manifest = status.manifest or {}
    index = {"collection": manifest.get("collection"), "kb_version": manifest.get("kb_version"),
             "built_at": manifest.get("built_at"), "state": status.state}
    if not status.ok and not (status.usable and args.allow_stale_index):
        print(f"Индекс: {status.message}", file=sys.stderr)
        return 2

    run = args.run or f"{args.set.stem}-{datetime.now(MSK):%Y%m%d-%H%M%S}"
    run_dir: Path = args.out or RUNS_DIR / run
    by_intent = Counter(i.intent for i in items)
    print(f"Прогон {run}: {len(items)} вопросов × {', '.join(names)}; «сейчас» {now.isoformat()}; "
          f"папка {display_path(run_dir)}")
    print("  по сценариям: " + ", ".join(f"{k} {v}" for k, v in sorted(by_intent.items())))
    if args.dry_run:
        print(f"Индекс: {status.message}. Ключ найден. Проверка без вызовов API закончена.")
        return 0

    answers_path = run_dir / "answers.jsonl"
    meta_path = run_dir / "meta.json"
    previous = read_jsonl(answers_path) if answers_path.is_file() else []
    settings = eval_settings(base, now, run_dir)
    if meta_path.is_file():
        meta = cast(dict[str, Any], read_json(meta_path))
        dataset = cast(dict[str, Any], meta.get("dataset") or {})
        old_index = cast(dict[str, Any], meta.get("index") or {}).get("collection")
        if dataset.get("sha256") != file_sha256(args.set) or meta.get("now") != now.isoformat(timespec="seconds") \
                or old_index != index["collection"]:
            print(f"В {display_path(run_dir)} прогон другого набора, другого --now или на другом индексе: "
                  "задайте новый --run.", file=sys.stderr)
            return 2
        meta["systems"] = sorted(set(cast(list[str], meta.get("systems") or [])) | set(names), key=SYSTEMS.index)
    else:
        meta = make_meta(run, args.set, all_items, names, now, settings, index, args.workers, args.retries)
    done = done_pairs(previous)
    tasks = [(item, system) for item in items for system in names if (item.id, system) not in done]
    if done:
        print(f"Продолжение: готово {len(done)}, осталось {len(tasks)}")

    try:
        systems, info = build_systems(names, settings, run) if tasks else ({}, {})
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    if info.get("problems"):
        print("Внимание, данные куратора: " + "; ".join(info["problems"]), file=sys.stderr)
    session = {"started_at": datetime.now(MSK).isoformat(timespec="seconds"), "tasks": len(tasks),
               "systems": names}
    write_json(meta_path, meta)

    lock = threading.Lock()
    counter = {"n": 0}

    def on_row(row: dict[str, Any]) -> None:
        with lock:
            append_jsonl(answers_path, row)
            counter["n"] += 1
            note = f" — ОШИБКА {row['error']}" if row.get("error") else ""
            note += f" — в обход модели: {row['degraded']}" if row.get("degraded") else ""
            note += f" (попыток: {row['attempts']})" if int(row.get("attempts") or 1) > 1 else ""
            print(f"[{counter['n']}/{len(tasks)}] {row['id']} {row['system']}: {_ms(row.get('latency_ms'))}, "
                  f"${float(row.get('cost') or 0):.4f}{note}", flush=True)

    rows = run_tasks(tasks, systems, workers=args.workers, retries=args.retries, on_row=on_row)
    final = ordered([*previous, *rows], all_items)
    write_jsonl(answers_path, final)
    if tasks:
        session["finished_at"] = datetime.now(MSK).isoformat(timespec="seconds")
        session["errors"] = sum(1 for r in rows if r.get("error"))
        cast(list[dict[str, Any]], meta.setdefault("sessions", [])).append(session)
        meta["finished_at"] = session["finished_at"]
        write_json(meta_path, meta)
    print()
    print(write_reports(run_dir, all_items, final, meta))
    errors = sum(1 for r in final if r.get("error"))
    if errors:
        print(f"Без ответа осталось {errors}: повторите тот же --run, готовые ответы не пересчитаются.",
              file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
