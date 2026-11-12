"""Разбор базы знаний: front-matter, разделы, FAQ с темами, фрагменты и их метаданные."""
from __future__ import annotations

from pathlib import Path

import pytest

from curator.kb import (FAQ_TOPIC_LABELS, FAQ_TOPICS, KBFormatError, KnowledgeBase, chunk_documents, load_kb,
                        parse_document, section_kind, split_front_matter)

GOOD = """---
id: L9.9
title: "Тест"
module: 2
topic: test-topic
resource_type: lecture
level: all
---

# Тест

Вступление.

## Раздел

Текст раздела.
"""


def test_front_matter_parsed():
    meta, body = split_front_matter(GOOD, "x.md")
    assert meta == {"id": "L9.9", "title": "Тест", "module": 2, "topic": "test-topic", "resource_type": "lecture",
                    "level": "all"}
    assert body.lstrip().startswith("# Тест")


def test_front_matter_with_bom_and_crlf():
    meta, _ = split_front_matter("﻿" + GOOD.replace("\n", "\r\n"), "x.md")
    assert meta["id"] == "L9.9"


@pytest.mark.parametrize("broken, message", [
    (GOOD.replace("---\nid", "id", 1), "нет YAML front-matter"),
    (GOOD.replace("topic: test-topic\n", ""), "нет ключей topic"),
    (GOOD.replace("level: all", "level: expert"), "level 'expert'"),
    (GOOD.replace("module: 2", "module: 7"), "module должен быть числом 0-4"),
    (GOOD.replace("module: 2", "module: два"), "module должен быть числом 0-4"),
    (GOOD.replace("resource_type: lecture", "resource_type: video"), "resource_type 'video'"),
])
def test_front_matter_errors_name_file_and_problem(broken: str, message: str):
    with pytest.raises(KBFormatError) as err:
        parse_document(broken, "module-2/x.md")
    assert "module-2/x.md" in str(err.value)
    assert message in str(err.value)


def test_fixture_kb_metadata(kb: KnowledgeBase):
    ids = {d.doc_id: d for d in kb.documents}
    assert set(ids) == {"L1.1", "L2.2", "guide-2", "faq", "course-overview"}
    assert ids["L2.2"].module == 2 and ids["L2.2"].topic == "pandas-filtering"
    assert ids["faq"].module == 0 and ids["faq"].resource_type == "faq"


def test_section_kinds_and_levels(kb: KnowledgeBase):
    lecture = kb.get("L2.2")
    assert lecture is not None
    kinds = {s.title: (s.kind, s.level) for s in lecture.sections}
    assert kinds["Частые ошибки"] == ("mistakes", "all")
    assert kinds["Если сложно"] == ("if_hard", "beginner")
    assert kinds["Для продвинутых"] == ("advanced", "advanced")
    assert kinds[""] == ("intro", "all")
    guide = kb.guide(2)
    assert guide is not None
    criteria = [s for s in guide.sections if s.kind == "criteria"]
    assert criteria and "Группировка и выводы | 5" in criteria[0].text
    assert any(s.title == "Шаги" and "### Шаг 2. Группировка" in s.text for s in guide.sections)


@pytest.mark.parametrize("heading, kind", [("Если сложно", "if_hard"), ("Для продвинутых", "advanced"),
                                           ("Критерии оценки", "criteria"), ("Как сдать", "submission"),
                                           ("Дедлайн и перенос", "deadline"), ("Булевы маски", "content")])
def test_section_kind(heading: str, kind: str):
    assert section_kind(heading) == kind


def test_faq_questions_have_own_topics(kb: KnowledgeBase):
    faq = {s.title: s for _, s in kb.sections(resource_type="faq")}
    assert faq["Можно ли перенести дедлайн?"].topic == "extensions"
    assert faq["Кто ставит оценки за ДЗ и итоговый проект?"].topic == "grading"
    assert all("<!--" not in s.text for s in faq.values())
    assert [s.title for _, s in kb.faq("certificate")] == ["Как получить сертификат?"]


def test_faq_without_topic_comment_is_an_error():
    text = GOOD.replace("resource_type: lecture", "resource_type: faq").replace("module: 2", "module: 0")
    with pytest.raises(KBFormatError, match="нет строки <!-- topic"):
        parse_document(text, "faq/faq.md")


def test_faq_unknown_topic_is_an_error():
    text = (GOOD.replace("resource_type: lecture", "resource_type: faq").replace("module: 2", "module: 0")
            .replace("Текст раздела.", "<!-- topic: weather -->\nТекст."))
    with pytest.raises(KBFormatError, match="topic 'weather'"):
        parse_document(text, "faq/faq.md")


def test_topics_include_faq_topics_with_module_zero(kb: KnowledgeBase):
    topics = kb.topics()
    assert topics["extensions"].module == 0
    assert topics["pandas-filtering"].module == 2
    assert "Фильтрация, сортировка, пропуски" in topics["pandas-filtering"].titles


def test_faq_topics_have_russian_names(kb: KnowledgeBase):
    topics = kb.topics()
    faq = {slug: t.label for slug, t in topics.items() if t.module == 0 and slug in FAQ_TOPICS}
    assert faq["deadlines"] == "FAQ: Дедлайны" and faq["extensions"] == "FAQ: Перенос срока"
    assert all("Частые вопросы" not in label for label in faq.values())  # не общий заголовок файла FAQ
    assert len(set(faq.values())) == len(faq)  # у каждой темы своё название
    assert set(FAQ_TOPIC_LABELS) == set(FAQ_TOPICS)
    assert topics["pandas-filtering"].label == "Фильтрация, сортировка, пропуски"


def test_chunks_carry_metadata_for_filters(kb: KnowledgeBase):
    chunks = chunk_documents(kb.documents, 1200, 150)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    for chunk in chunks:
        meta = chunk.metadata
        assert {"doc_id", "title", "module", "topic", "resource_type", "level", "section"} <= set(meta)
        assert all(isinstance(v, (str, int)) for v in meta.values())  # Chroma принимает только скаляры
    faq_chunk = next(c for c in chunks if c.metadata["section"] == "Можно ли перенести дедлайн?")
    assert faq_chunk.metadata["topic"] == "extensions" and faq_chunk.metadata["module"] == 0
    assert faq_chunk.page_content.startswith("FAQ курса. Можно ли перенести дедлайн?")


def test_code_indentation_survives_and_hash_comment_is_not_a_heading(kb: KnowledgeBase):
    lecture = kb.get("L1.1")
    assert lecture is not None
    section = next(s for s in lecture.sections if s.title == "Переменные и присваивание")
    assert "\n    # комментарий внутри блока кода — не заголовок раздела\n    print(title)" in section.text
    assert all("комментарий внутри" not in s.title for s in lecture.sections)


def test_long_section_is_split_with_same_metadata(kb: KnowledgeBase):
    chunks = chunk_documents(kb.documents, 100, 10)
    parts = [c for c in chunks if c.metadata["doc_id"] == "L2.2" and c.metadata["section"] == "loc и iloc"]
    assert len(parts) >= 2
    assert all(p.page_content.startswith("Лекция L2.2 «Фильтрация, сортировка, пропуски». loc и iloc") for p in parts)


def test_duplicate_ids_strict_and_lenient(tmp_path: Path):
    (tmp_path / "a.md").write_text(GOOD, encoding="utf-8")
    (tmp_path / "b.md").write_text(GOOD, encoding="utf-8")
    (tmp_path / "c.md").write_text("без front-matter", encoding="utf-8")
    with pytest.raises(KBFormatError):
        load_kb(tmp_path, strict=True)
    lenient = load_kb(tmp_path, strict=False)
    assert len(lenient.documents) == 1
    assert len(lenient.problems) == 2
    assert any("уже занят" in p for p in lenient.problems)


def test_guide_sections_and_module_overview(kb: KnowledgeBase):
    sections = kb.guide_sections(2, ("goal", "data", "steps"))
    assert [(d.doc_id, s.kind) for d, s in sections] == [("guide-2", "goal"), ("guide-2", "steps")]  # данных нет
    assert kb.guide_sections(3, ("goal",)) == []
    found = kb.module_overview(2)
    assert found is not None and found[0].doc_id == "course-overview"
    assert found[1].title.startswith("Модуль 2. Pandas") and "hw2" in found[1].text
    assert kb.module_overview(3) is None
