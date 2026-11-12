"""Рекомендации: дополнительные ресурсы из resources.csv и разделы лекций «Если сложно» / «Для продвинутых».

Выбор детерминированный: модуль и тема из анализа запроса, уровень студента. Модель потом объясняет выбор,
но список и ссылки даёт код — внешние адреса берутся только из каталога, придумать свой модель не может.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .kb import KBDocument, KnowledgeBase, Section

RESOURCE_FIELDS = ("id", "title", "url", "module", "topic", "level", "kind", "minutes", "description")
KIND_LABELS = {"docs": "документация", "tutorial": "учебник", "article": "статья", "course": "курс",
               "book": "книга", "catalog": "каталог", "video": "видео", "practice": "практика"}


@dataclass(frozen=True)
class Resource:
    id: str
    title: str
    url: str
    module: int
    topic: str
    level: str  # beginner | advanced | all
    kind: str
    minutes: int | None
    description: str

    def summary(self) -> str:
        minutes = f", ~{self.minutes} мин" if self.minutes else ""
        kind = KIND_LABELS.get(self.kind, self.kind)
        return f"{self.title} ({kind}{minutes}) — {self.url}\n{self.description}"


def load_resources(path: Path) -> list[Resource]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [f for f in RESOURCE_FIELDS if f not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: нет столбцов {', '.join(missing)}")
        out = []
        for row in reader:
            minutes = row["minutes"].strip()
            out.append(Resource(
                id=row["id"].strip(), title=row["title"].strip(), url=row["url"].strip(),
                module=int(row["module"] or 0), topic=row["topic"].strip(), level=row["level"].strip() or "all",
                kind=row["kind"].strip(), minutes=int(minutes) if minutes.isdigit() else None,
                description=row["description"].strip(),
            ))
    return out


@dataclass(frozen=True)
class SectionRef:
    doc: KBDocument
    section: Section


class Recommender:
    def __init__(self, resources: list[Resource], kb: KnowledgeBase):
        self.resources = resources
        self.kb = kb

    def resources_for(self, level: str, module: int | None, topic: str | None, k: int = 3) -> list[Resource]:
        """Тема важнее модуля, модуль важнее общего; ресурс чужого уровня не предлагается."""
        scored: list[tuple[int, int, str, Resource]] = []
        for res in self.resources:
            if res.level not in ("all", level):
                continue
            score = 0
            if topic and res.topic == topic:
                score += 4
            if module is not None and res.module == module:
                score += 3
            elif res.module == 0:
                score += 1
            elif module is not None:
                continue  # ресурс другого модуля
            if res.level == level:
                score += 1
            scored.append((-score, res.minutes or 999, res.id, res))
        scored.sort(key=lambda item: item[:3])
        return [item[3] for item in scored[:k]]

    def sections_for(self, level: str, module: int | None, topic: str | None, k: int = 2) -> list[SectionRef]:
        """Начинающему — «Если сложно» лекций по теме или модулю, продвинутому — «Для продвинутых»."""
        kind = "advanced" if level == "advanced" else "if_hard"
        found = self.kb.sections(kind=kind, resource_type="lecture", topic=topic) if topic else []
        if not found and module is not None:
            found = self.kb.sections(kind=kind, resource_type="lecture", module=module)
        found.sort(key=lambda pair: pair[0].doc_id)
        return [SectionRef(doc, section) for doc, section in found[:k]]
