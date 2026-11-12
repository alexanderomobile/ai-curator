"""База знаний: разбор front-matter, разделы markdown и фрагменты для индекса.

Формат (BRIEF, раздел 3): каждый файл начинается с YAML front-matter (id, title, module, topic, resource_type,
level). FAQ — один файл, вопрос — заголовок `## ...`, сразу под ним `<!-- topic: deadlines -->`.

Разделы режет MarkdownHeaderTextSplitter, длинные разделы — RecursiveCharacterTextSplitter. У сплиттера заголовков
две особенности, которые здесь обходятся: он срезает отступы строк (в коде на Python это ломает смысл) и склеивает
абзацы через «  \\n». Поэтому ведущие пробелы перед разбором заменяются невидимым символом и потом возвращаются,
а «  \\n» снова становится пустой строкой между абзацами.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

RESOURCE_TYPES = ("lecture", "guide", "faq", "overview")
LEVELS = ("beginner", "advanced", "all")
FAQ_TOPICS = ("deadlines", "extensions", "grading", "certificate", "recordings", "submission", "support",
              "schedule", "plagiarism", "access", "tools")
# Названия тем FAQ для интерфейса, отчёта и промпта анализа: у всех вопросов FAQ один файл и один заголовок,
# поэтому название темы берётся отсюда, а не из title документа.
FAQ_TOPIC_LABELS = {
    "deadlines": "Дедлайны", "extensions": "Перенос срока", "grading": "Оценки и проверка ДЗ",
    "certificate": "Сертификат", "recordings": "Записи вебинаров", "submission": "Сдача ДЗ",
    "support": "Поддержка и помощь", "schedule": "Расписание", "plagiarism": "Плагиат и самостоятельность",
    "access": "Доступ к курсу", "tools": "Инструменты: Python, Colab, Jupyter",
}
REQUIRED_KEYS = ("id", "title", "module", "topic", "resource_type", "level")
MODULES = (0, 1, 2, 3, 4)

RESOURCE_LABELS = {"lecture": "Лекция", "guide": "Методичка", "faq": "FAQ", "overview": "О курсе"}

# Вид раздела по заголовку: рекомендации ссылаются на «Если сложно» / «Для продвинутых», отказ в оценке —
# на «Критерии оценки» методички.
_SECTION_KINDS: tuple[tuple[str, str], ...] = (
    ("если сложно", "if_hard"),
    ("для продвинутых", "advanced"),
    ("частые ошибки", "mistakes"),
    ("типичные ошибки", "mistakes"),
    ("критери", "criteria"),
    ("как сдать", "submission"),
    ("сдача", "submission"),
    ("дедлайн", "deadline"),
    ("срок", "deadline"),
    ("цель", "goal"),
    ("шаги", "steps"),
    ("данные", "data"),
)
_KIND_LEVEL = {"if_hard": "beginner", "advanced": "advanced"}

_FRONT = re.compile(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_TOPIC_COMMENT = re.compile(r"<!--\s*topic:\s*([a-z0-9_-]+)\s*-->", re.IGNORECASE)
_INDENT = re.compile(r"^[ \t]+", re.MULTILINE)
_PAD = "⠀"  # «пустой шрифт Брайля»: печатный символ, его не срезают ни strip(), ни isprintable()
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


class KBFormatError(ValueError):
    """Файл базы знаний не по формату BRIEF; в тексте — какой файл и что не так."""


@dataclass(frozen=True)
class Section:
    title: str  # заголовок раздела `##`; "" — вступление до первого раздела
    kind: str  # intro | content | if_hard | advanced | mistakes | criteria | submission | deadline | ... | faq
    text: str
    topic: str  # у вопроса FAQ — свой topic, у остальных — topic документа
    level: str


@dataclass(frozen=True)
class KBDocument:
    doc_id: str
    title: str
    module: int
    topic: str
    resource_type: str
    level: str
    path: str  # относительно папки kb, через «/»
    body: str
    sha256: str
    sections: tuple[Section, ...] = ()

    @property
    def label(self) -> str:
        return doc_label(self.resource_type, self.doc_id, self.title)


def doc_label(resource_type: str, doc_id: str, title: str) -> str:
    """Как документ называется в ответе и в блоке «Источники»."""
    if resource_type == "lecture":
        return f"Лекция {doc_id} «{title}»"
    if resource_type == "faq":
        return "FAQ курса"
    if resource_type == "guide" and not title.lower().startswith("методичка"):
        return f"Методичка «{title}»"
    return title


@dataclass(frozen=True)
class TopicInfo:
    slug: str
    module: int
    titles: tuple[str, ...]

    @property
    def label(self) -> str:
        """Понятное название темы: «FAQ: Дедлайны», «Фильтрация, сортировка, пропуски»."""
        return "; ".join(self.titles) or self.slug


@dataclass
class KnowledgeBase:
    documents: list[KBDocument]
    problems: list[str] = field(default_factory=list)

    def get(self, doc_id: str) -> KBDocument | None:
        return next((d for d in self.documents if d.doc_id == doc_id), None)

    def sections(self, *, module: int | None = None, kind: str | None = None, resource_type: str | None = None,
                 topic: str | None = None) -> list[tuple[KBDocument, Section]]:
        found: list[tuple[KBDocument, Section]] = []
        for doc in self.documents:
            if module is not None and doc.module != module:
                continue
            if resource_type is not None and doc.resource_type != resource_type:
                continue
            for section in doc.sections:
                if kind is not None and section.kind != kind:
                    continue
                if topic is not None and section.topic != topic:
                    continue
                found.append((doc, section))
        return found

    def faq(self, topic: str) -> list[tuple[KBDocument, Section]]:
        return self.sections(resource_type="faq", topic=topic)

    def guide(self, module: int) -> KBDocument | None:
        return next((d for d in self.documents if d.resource_type == "guide" and d.module == module), None)

    def guide_sections(self, module: int, kinds: Iterable[str]) -> list[tuple[KBDocument, Section]]:
        """Разделы методички модуля заданных видов (goal, data, steps, ...) в порядке kinds, по первому
        разделу каждого вида."""
        guide = self.guide(module)
        if guide is None:
            return []
        found: list[tuple[KBDocument, Section]] = []
        for kind in kinds:
            section = next((s for s in guide.sections if s.kind == kind), None)
            if section is not None:
                found.append((guide, section))
        return found

    def module_overview(self, module: int) -> tuple[KBDocument, Section] | None:
        """Раздел обзора курса про модуль: заголовок «Модуль N. …» (формат обзора — kb-update.md)."""
        pattern = re.compile(rf"^\s*модуль\s*№?\s*{module}\b", re.IGNORECASE)
        for doc in self.documents:
            if doc.resource_type != "overview":
                continue
            section = next((s for s in doc.sections if pattern.match(s.title)), None)
            if section is not None:
                return doc, section
        return None

    def topics(self) -> dict[str, TopicInfo]:
        """Темы базы для анализа запроса и аналитики: slug → модуль и названия документов (у тем FAQ —
        русское название темы, а не общий заголовок файла FAQ)."""
        titles: dict[str, list[str]] = {}
        modules: dict[str, int] = {}
        for doc in self.documents:
            slugs = {s.topic for s in doc.sections} if doc.resource_type == "faq" else {doc.topic}
            for slug in slugs:
                title = f"FAQ: {FAQ_TOPIC_LABELS.get(slug, slug)}" if doc.resource_type == "faq" else doc.title
                titles.setdefault(slug, [])
                if title not in titles[slug]:
                    titles[slug].append(title)
                modules.setdefault(slug, doc.module)
        return {slug: TopicInfo(slug, modules[slug], tuple(t)) for slug, t in sorted(titles.items())}

    def titles(self) -> dict[str, str]:
        return {d.doc_id: d.title for d in self.documents}


def split_front_matter(text: str, where: str = "") -> tuple[dict[str, Any], str]:
    match = _FRONT.match(text)
    if not match:
        raise KBFormatError(f"{where}: нет YAML front-matter между строками «---» в начале файла")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise KBFormatError(f"{where}: front-matter не разбирается как YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise KBFormatError(f"{where}: front-matter должен быть словарём ключ: значение")
    return {str(k): v for k, v in meta.items()}, text[match.end():]


def validate_meta(meta: dict[str, Any], where: str = "") -> None:
    missing = [k for k in REQUIRED_KEYS if meta.get(k) in (None, "")]
    if missing:
        raise KBFormatError(f"{where}: в front-matter нет ключей {', '.join(missing)}")
    module = meta["module"]
    if isinstance(module, bool) or not isinstance(module, int) or module not in MODULES:
        raise KBFormatError(f"{where}: module должен быть числом 0-4, а не {module!r}")
    if meta["resource_type"] not in RESOURCE_TYPES:
        raise KBFormatError(f"{where}: resource_type {meta['resource_type']!r} — допустимо {', '.join(RESOURCE_TYPES)}")
    if meta["level"] not in LEVELS:
        raise KBFormatError(f"{where}: level {meta['level']!r} — допустимо {', '.join(LEVELS)}")


def section_kind(heading: str) -> str:
    low = heading.lower()
    for marker, kind in _SECTION_KINDS:
        if marker in low:
            return kind
    return "content"


def _protect(text: str) -> str:
    return _INDENT.sub(lambda m: _PAD * len(m.group(0).replace("\t", "    ")), text)


def _restore(text: str) -> str:
    return text.replace("  \n", "\n\n").replace(_PAD, " ").strip()


@dataclass(frozen=True)
class _Piece:
    h2: str
    h3: str
    text: str


def _pieces(body: str) -> list[_Piece]:
    splitter = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=True)
    out: list[_Piece] = []
    for part in splitter.split_text(_protect(body)):
        text = _restore(part.page_content)
        if not text:
            continue
        out.append(_Piece(str(part.metadata.get("h2", "")), str(part.metadata.get("h3", "")), text))
    return out


def _sections(meta: dict[str, Any], body: str, where: str) -> tuple[Section, ...]:
    """Разделы `##` документа; подразделы `###` входят в свой раздел."""
    grouped: dict[str, list[str]] = {}
    for piece in _pieces(body):
        chunk = f"### {piece.h3}\n\n{piece.text}" if piece.h3 else piece.text
        grouped.setdefault(piece.h2, []).append(chunk)
    sections: list[Section] = []
    for h2, parts in grouped.items():
        text = "\n\n".join(parts)
        if meta["resource_type"] == "faq":
            if not h2:
                continue  # вступление FAQ — не вопрос
            topic_match = _TOPIC_COMMENT.search(text)
            if not topic_match:
                raise KBFormatError(f"{where}: у вопроса «{h2}» нет строки <!-- topic: ... -->")
            topic = topic_match.group(1).lower()
            if topic not in FAQ_TOPICS:
                raise KBFormatError(f"{where}: topic {topic!r} у вопроса «{h2}» — допустимо {', '.join(FAQ_TOPICS)}")
            text = _TOPIC_COMMENT.sub("", text).strip()
            sections.append(Section(h2, "faq", text, topic, str(meta["level"])))
            continue
        kind = section_kind(h2) if h2 else "intro"
        level = _KIND_LEVEL.get(kind, str(meta["level"]))
        sections.append(Section(h2, kind, _TOPIC_COMMENT.sub("", text).strip(), str(meta["topic"]), level))
    return tuple(sections)


def parse_document(text: str, rel_path: str) -> KBDocument:
    meta, body = split_front_matter(text, rel_path)
    validate_meta(meta, rel_path)
    sections = _sections(meta, body, rel_path)
    if meta["resource_type"] == "faq" and not sections:
        raise KBFormatError(f"{rel_path}: в FAQ нет ни одного вопроса `## ...`")
    return KBDocument(
        doc_id=str(meta["id"]), title=str(meta["title"]), module=int(meta["module"]), topic=str(meta["topic"]),
        resource_type=str(meta["resource_type"]), level=str(meta["level"]), path=rel_path, body=body,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), sections=sections,
    )


def kb_files(kb_dir: Path) -> list[Path]:
    return sorted(p for p in kb_dir.rglob("*.md") if p.is_file())


def file_hashes(kb_dir: Path) -> dict[str, str]:
    return {p.relative_to(kb_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in kb_files(kb_dir)}


def load_kb(kb_dir: Path, strict: bool = True) -> KnowledgeBase:
    """strict=True (индексация) — первая же ошибка формата останавливает; False (приложение) — файл пропускается,
    ошибка попадает в problems и показывается в интерфейсе."""
    if not kb_dir.is_dir():
        raise KBFormatError(f"нет папки базы знаний {kb_dir}")
    documents: list[KBDocument] = []
    problems: list[str] = []
    seen: dict[str, str] = {}
    for path in kb_files(kb_dir):
        rel = path.relative_to(kb_dir).as_posix()
        try:
            doc = parse_document(path.read_text(encoding="utf-8"), rel)
            if doc.doc_id in seen:
                raise KBFormatError(f"{rel}: id {doc.doc_id!r} уже занят файлом {seen[doc.doc_id]}")
        except KBFormatError as exc:
            if strict:
                raise
            problems.append(str(exc))
            continue
        seen[doc.doc_id] = rel
        documents.append(doc)
    if strict and not documents:
        raise KBFormatError(f"в {kb_dir} нет ни одного .md файла")
    return KnowledgeBase(documents, problems)


def chunk_documents(docs: Iterable[KBDocument], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Фрагменты для индекса с метаданными doc_id, title, module, topic, resource_type, level, section.

    В текст фрагмента впереди добавляется «Название. Раздел»: так фрагмент из середины лекции находится по теме
    лекции, а модель видит, откуда он.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", " ", ""])
    out: list[Document] = []
    for doc in docs:
        index = 0
        for section in doc.sections:
            heading = section.title or doc.title
            for part in splitter.split_text(section.text):
                metadata: dict[str, str | int] = {
                    "doc_id": doc.doc_id, "title": doc.title, "module": doc.module, "topic": section.topic,
                    "resource_type": doc.resource_type, "level": section.level, "section": section.title,
                    "section_kind": section.kind, "source": doc.path, "chunk": index,
                }
                prefix = f"{doc.label}. {heading}" if heading != doc.title else doc.label
                out.append(Document(page_content=f"{prefix}\n\n{part}", metadata=metadata,
                                    id=f"{doc.doc_id}::{index:03d}"))
                index += 1
    return out
