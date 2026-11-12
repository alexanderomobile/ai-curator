"""Промпты из файлов и few-shot примеры."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from curator.fewshot import FewShotLibrary, check_examples, load_examples, parse_examples
from curator.prompts import GENERATIVE_INTENTS, PROMPTS_DIR, PromptError, PromptLibrary, quote, render, tidy
from tests.conftest import FIXTURES, ROOT

# Блоки, которые собирает код (curator/curator.py): каждый должен быть в файле шаблона.
USED_BLOCKS = {
    "refusal_grading": {"main", "rule", "status", "status_revision", "criteria_step", "no_module_step",
                        "review_step", "help_step"},
    "refusal_schedule": {"main", "main_events", "rule", "deadline_open", "deadline_closed", "deadline_plain",
                         "status", "status_revision", "student_used", "student_free", "events", "steps_open",
                         "steps_closed", "steps_revision", "steps_unknown", "steps_events"},
    "revision": {"fix"},
    "refusal_off_topic": {"main", "topic", "greeting", "thanks", "ack"},
    "refusal_limits": {"too_long", "per_hour", "hourly", "daily", "empty"},
    "no_answer": {"main", "next", "sources"},
    "llm_error": {"main"},
    "search_error": {"main", "sources"},
    "context": {"header", "item", "fewshot_header", "fewshot_item"},
}


def test_library_loads_all_prompt_files():
    lib = PromptLibrary()
    for name, blocks in USED_BLOCKS.items():
        assert blocks <= set(lib.blocks(name)), name
    for intent in GENERATIVE_INTENTS:
        assert lib.scenario(intent).startswith("Сценарий:")
    assert "куратор" in lib.system(today="четверг, 12.11.2026 14:00")
    assert "начинающий" in lib.level("beginner") and "продвинутый" in lib.level("advanced")


def test_curator_uses_only_existing_blocks():
    source = (ROOT / "curator" / "curator.py").read_text(encoding="utf-8")
    used = set(re.findall(r'block\("([a-z_]+)",\s*"([a-z_]+)"', source))
    lib = PromptLibrary()
    for name, block in used:
        assert block in lib.blocks(name), f"{name}.{block}"


def test_comments_do_not_reach_prompts():
    lib = PromptLibrary()
    assert {p.stem for p in PROMPTS_DIR.glob("*.md")} == set(lib.names())
    for name in lib.names():
        assert "<!--" not in lib.template(name)
        for block in lib.blocks(name):
            assert "<!--" not in lib.template(name, block)


def test_render_requires_every_placeholder():
    assert render("Дедлайн {{when}}{{ref}}", when="пт", ref=" [1]") == "Дедлайн пт [1]"
    with pytest.raises(PromptError):
        render("Дедлайн {{when}}")


def test_fixed_scenarios_have_no_model_instruction():
    with pytest.raises(PromptError):
        PromptLibrary().scenario("grading_request")


def test_missing_prompt_file_is_reported(tmp_path: Path):
    (tmp_path / "system.md").write_text("x", encoding="utf-8")
    with pytest.raises(PromptError, match="level_beginner.md"):
        PromptLibrary(tmp_path)


def test_tidy_and_quote():
    assert tidy("а\n\n\n\nб  \n") == "а\n\nб"
    assert quote("строка 1\n\nстрока 2") == "> строка 1\n>\n> строка 2"


def test_fewshot_fixture_meets_brief():
    examples, problems = load_examples(FIXTURES / "fewshot" / "examples.yaml")
    assert problems == [] and check_examples(examples) == []


def test_fewshot_select_same_intent_and_level_plus_no_answer():
    examples, _ = load_examples(FIXTURES / "fewshot" / "examples.yaml")
    lib = FewShotLibrary(examples)
    chosen = lib.select("deadline", "advanced")
    assert [(e.scenario, e.level) for e in chosen] == [("deadline", "advanced"), ("deadline", "beginner"),
                                                       ("no_answer", "advanced")]
    assert [e.scenario for e in lib.select("material", "beginner")][:2] == ["difficulty", "difficulty"]
    assert chosen[0].sources_text().startswith("[1] FAQ")


def test_fewshot_problems_reported():
    examples, problems = parse_examples([{"intent": "deadline", "question": "q"}, "не словарь"])
    assert examples == [] and len(problems) == 2
    assert "нет полей bad_answer, why_bad, good_answer" in problems[0]
    assert check_examples([]) == ["пар 0, нужно не меньше 16",
                                  "нет сценариев: deadline, difficulty, recommendation, organizational, "
                                  "grading_request, schedule_change, off_topic, no_answer"]


def test_fewshot_tags_select_only_when_condition_holds():
    examples, problems = load_examples(FIXTURES / "fewshot" / "examples.yaml")
    assert problems == []
    lib = FewShotLibrary(examples)
    tagged = next(e for e in examples if e.tags)
    assert tagged.tags == ("revision",) and tagged.scenario == "deadline"
    assert tagged not in lib.select("deadline", "beginner")  # без условия пара не идёт
    chosen = lib.select("deadline", "advanced", tags={"revision"})
    assert chosen[0] is tagged and len(chosen) == 3  # с условием — первой, даже другого уровня
    assert [e.scenario for e in chosen] == ["deadline", "deadline", "no_answer"]


def test_real_fewshot_revision_pair_is_tagged():
    examples, _ = load_examples(ROOT / "data" / "fewshot" / "examples.yaml")
    revision = [e for e in examples if "revision" in e.tags]
    assert len(revision) == 1 and revision[0].scenario == "deadline"
    assert "5 дней с момента решения преподавателя" in revision[0].good_answer


def test_prompts_forbid_invented_reasons_and_comfort_for_advanced():
    lib = PromptLibrary()
    system = lib.system(today="четверг, 12.11.2026 14:00")
    assert "Не додумывай к правилам курса причины" in system and "последствия" in system
    assert "не заполняй догадкой «обычно…»" in system
    assert "Срок, которого в материалах нет" in lib.scenario("organizational")
    assert "без диапазонов" in system  # ссылки — по одной: [1], [2]
    advanced = lib.level("advanced")
    assert "без утешений, похвалы вопроса и вступлений" in advanced
    assert "продвинутому — сразу по существу" in lib.scenario("difficulty")
    assert "формат сдачи из данных LMS" in lib.scenario("deadline")


def test_system_prompt_on_help_channels_and_first_person():
    """R3-5, R3-6: каналы помощи — только из материалов, «в материалах нет» — только после проверки всех материалов,
    куратор говорит о себе в первом лице."""
    system = PromptLibrary().system(today="четверг, 12.11.2026 14:00")
    assert "Куда обратиться за помощью, называй только так, как сказано в блоке «Материалы»" in system
    assert "Других форматов помощи (встреч, созвонов, консультаций) и контактов не придумывай" in system
    assert "не позже какого момента появится запись вебинара), бери как есть" in system
    assert "«В материалах этого нет» говори только о том, чего действительно нет ни в одном материале" in system
    assert "ИИ-куратор этого курса — это ты" in system and "«спросите меня»" in system


def test_deadline_prompt_separates_revision_from_lateness():
    """R3-7: доработка — решение по сданной и проверенной работе, а не последствие опоздания."""
    scenario = PromptLibrary().scenario("deadline")
    assert "«На доработку» — решение преподавателя по уже сданной и проверенной работе" in scenario
    assert "не последствие опоздания" in scenario and "последствий не называй" in scenario
    assert "Не пиши, что работу, не сданную к дедлайну" in scenario
    assert "назови срок её появления из данных LMS как есть" in PromptLibrary().scenario("organizational")


def test_real_fewshot_shows_invented_lecture_duration_as_bad():
    examples, _ = load_examples(ROOT / "data" / "fewshot" / "examples.yaml")
    recommendation = [e for e in examples if e.scenario == "recommendation"]
    assert any("Время чтения раздела лекции выдумано" in e.why_bad for e in recommendation)
