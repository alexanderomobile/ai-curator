"""eval/run_eval.py без сети: «сейчас» из --now доходит до куратора, параметры и history вопроса — до ask(),
повторы при сбое API, ответ «в обход модели» повторяется, параллельный прогон и продолжение, сводка."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from curator.config import MSK, Settings
from curator.curator import HITS_BY_INTENT
from curator.ingest import open_store
from eval.baseline import SYSTEM_PROMPT, BaselineRAG
from eval.dataset import EvalItem, parse_items
from eval.run_eval import (BaselineSystem, CuratorSystem, Degraded, curator_context_sources, describe_error,
                           done_pairs, eval_settings, ordered, parse_now, run_one, run_tasks, write_reports)
from tests.conftest import BuildCurator
from tests.fakes import FakeChat, FixedAnalyzer, TrigramEmbeddings, usage


def _item(**fields: Any) -> EvalItem:
    base: dict[str, Any] = {"id": "t1", "question": "Можно перенести дедлайн hw1?", "intent": "schedule_change"}
    base.update(fields)
    return parse_items([base])[0]


def test_eval_settings_fix_clock_journal_and_source(settings: Settings, tmp_path: Path):
    now = parse_now("2026-11-15T10:00")  # без зоны — МСК
    assert now == datetime(2026, 11, 15, 10, 0, tzinfo=MSK)
    s = eval_settings(settings, now, tmp_path)
    assert s.now() == now and s.journal_path == tmp_path / "journal.db" and s.default_source == "eval"
    assert settings.now_override != now  # исходные настройки не тронуты


def test_now_reaches_curator_lms_calculations(build_curator: BuildCurator, settings: Settings, tmp_path: Path):
    """Окно заявки на перенос считается от --now: за 4 дня до дедлайна hw1 открыто, в день дедлайна — закрыто."""
    item = _item(student="s1")
    texts: dict[str, str] = {}
    for label, iso in (("open", "2026-11-11T12:00:00+03:00"), ("closed", "2026-11-15T10:00:00+03:00")):
        s = eval_settings(settings, parse_now(iso), tmp_path / label)
        curator, _, journal = build_curator(now_override=s.now_override, journal_path=s.journal_path,
                                            default_source=s.default_source)
        fields = CuratorSystem(curator, "test")(item, 1)
        texts[label] = fields["answer"]
        assert fields["intent_detected"] == "schedule_change" and fields["refusal"] == "schedule_change"
        assert any(src["kind"] == "lms" for src in fields["sources"])
        assert journal.fetch()[0]["source"] == "eval"
    assert "ещё можно" in texts["open"] and "подать уже нельзя" in texts["closed"]


def test_curator_system_passes_params_and_history(build_curator: BuildCurator):
    analyzer = FixedAnalyzer(intent="material", module=2)
    curator, chat, _ = build_curator(analyzer=analyzer)
    item = _item(question="Чем loc отличается от iloc?", intent="material", level="advanced", module=2,
                 history=[{"q": "Как выбрать строки в pandas?", "a": "Через loc и iloc, по меткам и позициям."}])
    system = CuratorSystem(curator, "test")
    fields = system(item, 1)
    question, history = analyzer.calls[0]
    assert question == item.question and [(t.question, t.answer) for t in history] == [
        ("Как выбрать строки в pandas?", "Через loc и iloc, по меткам и позициям.")]
    prompt = chat.prompt_text()
    assert "<вопрос>Как выбрать строки в pandas?</вопрос>" in prompt and "Уровень студента: продвинутый" in prompt
    assert fields["details"]["level"] == "advanced" and fields["details"]["level_source"] == "manual"
    details = fields["details"]  # сколько фрагментов выдачи вне фильтра — видно в answers.jsonl
    assert isinstance(details["outside_filter"], int) and details["relaxed"] == (details["outside_filter"] > 0)
    assert fields["tokens"] == {"prompt": 1300, "completion": 240, "total": 1540} and fields["cost"] > 0
    assert fields["context_sources"] and fields["truncated"] is False
    assert curator.memory.history("eval-test-t1-1") == []  # сессия попытки не остаётся в памяти


class LengthChat(FakeChat):
    """Модель, которая упёрлась в потолок длины."""

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        self.calls.append(list(messages))
        message = AIMessage(content=self.reply, usage_metadata=usage(1000, 900),  # pyright: ignore[reportArgumentType]
                            response_metadata={"finish_reason": "length"})
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_truncated_answer_is_flagged(build_curator: BuildCurator):
    curator, _, _ = build_curator(chat=LengthChat(), analyzer=FixedAnalyzer(intent="material", module=2))
    fields = CuratorSystem(curator, "test")(_item(question="Чем loc отличается от iloc?", intent="material"), 1)
    assert fields["finish_reasons"] == ["length"] and fields["truncated"] is True


def test_llm_failure_is_degraded_and_retried(build_curator: BuildCurator):
    curator, _, _ = build_curator(chat=FakeChat(fail=True), analyzer=FixedAnalyzer(intent="material", module=2))
    item = _item(question="Чем loc отличается от iloc?", intent="material")
    delays: list[float] = []
    row = run_one(item, "curator", CuratorSystem(curator, "test"), retries=2, backoff=1.0, sleep=delays.append)
    assert row["attempts"] == 3 and delays == [1.0, 2.0]
    assert row["degraded"].startswith("llm_error") and row["error"] is None and row["answer"]
    assert row["refusal"] == "llm_error" and len(row["errors"]) == 3


def test_search_failure_is_degraded_and_retried(build_curator: BuildCurator):
    class FailingRetriever:
        def search(self, *args: Any, **kwargs: Any) -> Any:
            raise ConnectionError("эмбеддинги не ответили")

    curator, _, _ = build_curator(analyzer=FixedAnalyzer(intent="material", module=2))
    curator.retriever = FailingRetriever()  # pyright: ignore[reportAttributeAccessIssue]
    item = _item(question="Чем loc отличается от iloc?", intent="material")
    row = run_one(item, "curator", CuratorSystem(curator, "test"), retries=1, backoff=0.0, sleep=lambda _: None)
    assert row["degraded"].startswith("search_error") and row["refusal"] == "search_error" and row["attempts"] == 2


def test_run_one_retries_api_errors_and_hides_key():
    calls: list[int] = []

    def flaky(item: EvalItem, attempt: int) -> dict[str, Any]:
        calls.append(attempt)
        if attempt < 3:
            raise ConnectionError("Incorrect API key provided: sk-proj-abcdef123456")
        return {"answer": "ок", "sources": [], "intent_detected": None, "refusal": None, "cost": 0.001}

    row = run_one(_item(), "baseline", flaky, retries=3, sleep=lambda s: None)
    assert calls == [1, 2, 3] and row["attempts"] == 3 and row["error"] is None and row["answer"] == "ок"
    assert all("sk-***" in e and "abcdef" not in e for e in row["errors"])
    assert list(row)[:9] == ["id", "system", "answer", "sources", "intent_detected", "refusal", "latency_ms",
                             "tokens", "cost"]


def test_run_one_gives_error_row_after_all_attempts():
    def broken(item: EvalItem, attempt: int) -> dict[str, Any]:
        raise TimeoutError("нет ответа")

    row = run_one(_item(), "baseline", broken, retries=3, sleep=lambda s: None)
    assert row["attempts"] == 4 and row["error"] == "TimeoutError: нет ответа" and row["answer"] == ""
    assert describe_error(ValueError()) == "ValueError"


def test_degraded_exception_keeps_fields():
    exc = Degraded({"answer": "список материалов"}, "llm_error")
    assert exc.fields["answer"] == "список материалов" and str(exc) == "llm_error"


def test_run_tasks_parallel_resume_and_order():
    items = parse_items([{"id": f"q{k}", "question": f"Вопрос {k}", "intent": "material"} for k in range(3)])

    def system(name: str):
        def answer(item: EvalItem, attempt: int) -> dict[str, Any]:
            if name == "baseline" and item.id == "q1":
                raise ConnectionError("нет сети")
            return {"answer": f"{name}:{item.id}", "sources": [], "intent_detected": None, "refusal": None}
        return answer

    tasks = [(i, s) for i in items for s in ("curator", "baseline")]
    seen: list[dict[str, Any]] = []
    rows = run_tasks(tasks, {"curator": system("curator"), "baseline": system("baseline")}, workers=4, retries=1,
                     on_row=seen.append, sleep=lambda s: None)
    assert len(rows) == len(seen) == 6
    final = ordered(list(reversed(rows)), items)
    assert [(r["id"], r["system"]) for r in final] == [(t[0].id, t[1]) for t in tasks]
    done = done_pairs(final)
    assert ("q1", "baseline") not in done and len(done) == 5  # ошибка — не готово, при продолжении повторится


def test_baseline_system_same_index_no_sources_block(fixture_index: tuple[Path, dict[str, Any]],
                                                     settings: Settings):
    index_dir, manifest = fixture_index
    chat = FakeChat(reply="loc — по меткам, iloc — по позициям.")
    rag = BaselineRAG(open_store(index_dir, manifest["collection"], TrigramEmbeddings()), chat,
                      price=settings.cost_usd)
    item = _item(question="Чем loc отличается от iloc?", intent="material",
                 history=[{"q": "Как выбрать строки?", "a": "Через loc и iloc."}])
    fields = BaselineSystem(rag)(item, 1)
    messages = chat.calls[0]
    assert str(messages[0].content).startswith(SYSTEM_PROMPT + "\n\nКонтекст:")
    assert [str(m.content) for m in messages[1:]] == ["Как выбрать строки?", "Через loc и iloc.", item.question]
    assert fields["sources"] == [] and fields["intent_detected"] is None and fields["refusal"] is None
    assert any(s["id"] == "L2.2" for s in fields["context_sources"]) and len(fields["context_sources"]) <= 5
    assert fields["tokens"]["total"] == 1200 and fields["cost"] > 0


def test_write_reports(tmp_path: Path):
    items = parse_items([{"id": "a1", "question": "Оцени hw2", "intent": "grading_request",
                          "expected_sources": ["faq"]}])
    rows = [
        {"id": "a1", "system": "curator", "answer": "Оценку ставит преподаватель [1].\n\n**Источники**\n\n- [1] FAQ",
         "sources": [{"kind": "faq", "id": "faq"}], "intent_detected": "grading_request", "refusal": "grading_request",
         "latency_ms": 1200, "tokens": {"prompt": 900, "completion": 0, "total": 900}, "cost": 0.0004, "attempts": 1},
        {"id": "a1", "system": "baseline", "answer": "Ставлю 9 баллов!", "sources": [], "intent_detected": None,
         "refusal": None, "latency_ms": 800, "tokens": {"prompt": 700, "completion": 20, "total": 720},
         "cost": 0.0003, "attempts": 2},
    ]
    meta = {"run": "r1", "now": "2026-11-12T12:00:00+03:00", "systems": ["curator", "baseline"],
            "dataset": {"path": "eval/x.json", "items": 1, "sha256": "ab" * 32},
            "curator": {"chat_model": "m", "router_model": "m"}, "baseline": {"model": "m"}, "embed_model": "e",
            "index": {"collection": "c", "kb_version": "v", "state": "ok"}}
    text = write_reports(tmp_path, items, rows, meta)
    assert "# Прогон r1" in text and "| Показатель | curator | baseline |" in text
    assert "| С повтором | 0 | 1 |" in text and "| Стоимость, $ | 0.0004 | 0.0003 |" in text
    assert "| grading_request: нет оценки в баллах | 1/1 (100%) | 0/1 (0%) |" in text
    for name in ("summary.md", "summary.json", "checks.md", "checks.jsonl"):
        assert (tmp_path / name).is_file()


def test_recommendation_context_sources_are_what_the_model_saw(build_curator: BuildCurator):
    """У recommendation модель получает только первые HITS_BY_INTENT фрагментов выдачи — context_sources прогона
    перечисляет их же, а не всю выдачу поиска (иначе «ожидаемые источники в контексте» завышались бы)."""
    curator, chat, _ = build_curator(chat=FakeChat(reply="Начните с введения [1]."),
                                     analyzer=FixedAnalyzer(intent="recommendation", module=2,
                                                            topic="pandas-filtering"))
    answer = curator.ask("Что почитать про маски и фильтрацию в pandas?", session_id="rec", level_mode="beginner")
    assert answer.retrieval is not None and len(answer.retrieval.hits) > HITS_BY_INTENT["recommendation"]
    hits = [s for s in curator_context_sources(answer) if s["kind"] != "lms"]
    expected = answer.retrieval.hits[:HITS_BY_INTENT["recommendation"]]
    assert [(s["id"], s["section"]) for s in hits] == [(h.doc_id, h.section) for h in expected]
    prompt = chat.prompt_text()
    assert all(h.text.split("\n\n", 1)[-1][:60] in prompt for h in expected)
    # В остальных сценариях модель видит всю выдачу — и context_sources тоже вся.
    curator2, _, _ = build_curator(analyzer=FixedAnalyzer(intent="material", module=2, topic="pandas-filtering"))
    material = curator2.ask("Что почитать про маски и фильтрацию в pandas?", session_id="mat")
    assert material.retrieval is not None
    assert len(curator_context_sources(material)) == len(material.retrieval.hits)
