"""Договор с данными курса (data/): формат совпадает с тем, что ждёт код. Тесты только читают data/.

Данные пишут отдельно от кода; если какой-то части ещё нет, её проверка пропускается, а не падает.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from curator.config import BASE_DIR, MODULE_TITLES
from curator.curator import RESOURCES_BY_INTENT, SECTIONS_BY_INTENT
from curator.fewshot import check_examples, load_examples
from curator.kb import FAQ_TOPICS, KnowledgeBase, load_kb
from curator.lms import RECORDING_DELAY, FileBackend
from curator.recommender import KIND_LABELS, Recommender, load_resources
from curator.router import RuleAnalyzer

DATA = BASE_DIR / "data"


def need(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"нет {path.relative_to(BASE_DIR)}")
    return path


@pytest.fixture(scope="module")
def real_kb() -> KnowledgeBase:
    return load_kb(need(DATA / "kb"), strict=True)


def test_kb_parses_strictly(real_kb: KnowledgeBase):
    types = {d.resource_type for d in real_kb.documents}
    assert {"lecture", "guide", "faq", "overview"} <= types


def test_every_lecture_has_sections_for_recommendations(real_kb: KnowledgeBase):
    for doc in (d for d in real_kb.documents if d.resource_type == "lecture"):
        kinds = {s.kind for s in doc.sections}
        assert {"mistakes", "if_hard", "advanced"} <= kinds, doc.doc_id


def test_every_module_has_guide_with_criteria(real_kb: KnowledgeBase):
    for module in MODULE_TITLES:
        guide = real_kb.guide(module)
        assert guide is not None, f"нет методички модуля {module}"
        assert any(s.kind == "criteria" for s in guide.sections), guide.doc_id


def test_faq_topics_cover_scenarios_used_by_code(real_kb: KnowledgeBase):
    topics = {s.topic for _, s in real_kb.sections(resource_type="faq")}
    assert topics <= set(FAQ_TOPICS)
    assert {"grading", "extensions", "recordings"} <= topics  # на них опираются фиксированные сценарии


def test_lms_files_valid_and_linked_to_kb(real_kb: KnowledgeBase):
    backend = FileBackend(need(DATA / "lms"))
    assignments = backend.assignments()
    assert {a.id for a in assignments} == {"hw1", "hw2", "hw3", "final"}
    assert all(a.deadline.utcoffset() is not None for a in assignments)
    assert all(real_kb.get(a.guide_id) is not None for a in assignments)
    assert len(backend.students()) == 5
    assert {e.type for e in backend.schedule()} >= {"webinar", "review"}


def test_resources_catalog_loads():
    resources = load_resources(need(DATA / "resources" / "resources.csv"))
    assert resources and all(r.url.startswith("https://") for r in resources)
    assert {r.level for r in resources} <= {"beginner", "advanced", "all"}


def test_resources_catalog_fields_match_code(real_kb: KnowledgeBase):
    """Поля каталога, которые код понимает только из своего списка (docs/kb-update.md, раздел 7): вид — из
    KIND_LABELS, модуль 0-4, тема — тема базы знаний, id уникальны. Неверный kind или topic иначе прошли бы молча."""
    resources = load_resources(need(DATA / "resources" / "resources.csv"))
    assert {r.kind for r in resources} <= set(KIND_LABELS)
    assert {r.module for r in resources} <= {0, *MODULE_TITLES}
    assert {r.topic for r in resources} <= set(real_kb.topics())
    assert len({r.id for r in resources}) == len(resources)


def test_fewshot_meets_brief():
    examples, problems = load_examples(need(DATA / "fewshot" / "examples.yaml"))
    assert problems == []
    assert check_examples(examples) == []


def test_seaborn_recommendation_in_rules_mode_gets_module_3(real_kb: KnowledgeBase):
    """Запасные правила (модель анализа не ответила): вопрос про seaborn от опытного студента — модуль 3, тема
    seaborn, продвинутый уровень; каталог отдаёт ресурсы модуля 3, а не установку среды из модуля 0."""
    lectures = {d.doc_id: d.topic for d in real_kb.documents if d.resource_type == "lecture"}
    a = RuleAnalyzer(list(real_kb.topics()), lectures).analyze(
        "Посоветуй что почитать по seaborn, я уже уверенно пишу на pandas", []).analysis
    assert (a.intent, a.module, a.topic, a.level_hint) == ("recommendation", 3, "seaborn-chart-choice", "advanced")
    recommender = Recommender(load_resources(need(DATA / "resources" / "resources.csv")), real_kb)
    resources = recommender.resources_for("advanced", a.module, a.topic, RESOURCES_BY_INTENT["recommendation"])
    assert resources[0].id == "res-26" and all(r.module == 3 for r in resources)
    sections = recommender.sections_for("advanced", a.module, a.topic, SECTIONS_BY_INTENT["recommendation"])
    assert [(s.doc.doc_id, s.section.kind) for s in sections] == [("L3.2", "advanced")]


def test_faq_topics_in_real_kb_have_russian_names(real_kb: KnowledgeBase):
    labels = {slug: t.label for slug, t in real_kb.topics().items() if slug in FAQ_TOPICS}
    assert len(labels) == len(FAQ_TOPICS) and all(label.startswith("FAQ: ") for label in labels.values())
    assert len(set(labels.values())) == len(labels)


def test_recording_delay_matches_faq(real_kb: KnowledgeBase):
    """Срок появления записи вебинара в фактах LMS (curator.lms.RECORDING_DELAY) — то же число, что в FAQ."""
    hours = int(RECORDING_DELAY.total_seconds() // 3600)
    texts = [section.text for _, section in real_kb.faq("recordings")]
    assert texts and any(f"в течение {hours} часов" in text for text in texts)
