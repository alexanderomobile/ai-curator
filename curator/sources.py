"""Источники ответа: нумерованные материалы для промпта, ссылки [n] в тексте и блок «Источники» в конце.

Блок «Источники» всегда собирает код, а не модель: из тех материалов, на которые модель сослалась, а если она не
сослалась ни на один — из всех, что ей дали. Ссылки на номера, которых не было в материалах, из текста удаляются,
а свой список источников, если модель его всё же написала, отрезается — и отдельной строкой «Источники», и в виде
«**Источники:** [1] …» на той же строке. Показанные источники перенумеровываются подряд в порядке первого
упоминания ([4], [6] → [1], [2]), ссылки в тексте меняются по той же карте.

Ссылка может быть перечислением и диапазоном: «[1, 2]», «[1-3]», «[1–5]», «[2, 4-5]» разворачиваются в отдельные
номера («[1][2][3]»), несуществующие номера отбрасываются. Скобка с нулём («[0-10]», «[0, 1, 2]») или с обратным
диапазоном («[5-1]») — не ссылка: это шкала баллов или список Python, и она остаётся в тексте как есть. Так же
остаётся скобка из нескольких номеров, среди которых нет ни одного номера материала («[10, 20, 30]» при трёх
материалах): это список в тексте ответа, а не ссылка. Одиночный несуществующий номер («[9]») вырезается.
Ограничение: список из номеров материалов («[1, 2, 3]» при трёх материалах) от ссылки не отличить.

Пробелы чинятся только там, где ссылку вырезали: «лекции [9]» перед точкой → «лекции.», «L2.1 [9] и» → «L2.1 и»,
«[1], [9], [9].» → «[1].». Остальной текст не трогается: «файл .ipynb», «метод .copy()», «a , b» остаются как есть.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .kb import doc_label
from .retriever import Hit

KIND_LABELS = {"lecture": "лекция", "guide": "методичка", "faq": "FAQ", "overview": "о курсе", "lms": "LMS",
               "resource": "доп. ресурс"}
# Ссылка — «[n]» не сразу после латинской буквы или скобки: `arr[1]` и `f(x)[0]` в тексте — это код, а не
# источник. После кириллицы и цифры («в лекции[1]», «L2.2[1]») — ссылка. Внутри — номера через запятую или точку
# с запятой, каждый — число или диапазон через дефис или тире.
_NUM = r"\d{1,2}(?:\s*[-–—]\s*\d{1,2})?"
_CITE = re.compile(rf"(?<![A-Za-z_)])\[({_NUM}(?:\s*[,;]\s*{_NUM})*)\]")
_PART = re.compile(r"^(\d{1,2})(?:\s*[-–—]\s*(\d{1,2}))?$")


def cite_numbers(inner: str) -> list[int] | None:
    """Номера ссылки из текста между скобками: «1, 3-5» → [1, 3, 4, 5]. None — это не ссылка: в скобке есть ноль
    («0-10», «0, 1, 2», «0») или обратный диапазон («5-1»)."""
    numbers: list[int] = []
    for part in re.split(r"\s*[,;]\s*", inner.strip()):
        match = _PART.match(part.strip())
        if not match:
            return None
        first = int(match.group(1))
        if first < 1:
            return None
        if match.group(2) is None:
            numbers.append(first)
            continue
        last = int(match.group(2))
        if last < first:
            return None
        numbers.extend(range(first, last + 1))
    return numbers


def _list_literal(numbers: list[int], known: Callable[[int], bool]) -> bool:
    """Скобка из нескольких номеров, ни один из которых не номер материала, — список в тексте («[10, 20, 30]»)."""
    return len(numbers) > 1 and not any(known(n) for n in numbers)
_CODE = re.compile(r"```[\s\S]*?(?:```|$)|`[^`\n]+`")
# Строка-заголовок списка источников: «Источники», «**Источники:**», «### Источник:», «*Sources:* [1] …».
# Текст на той же строке допустим только после двоеточия: «Источник данных — Kaggle» — обычное предложение.
_MODEL_SOURCES = re.compile(r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?[*_]{0,2}(?:источники?|список источников|sources?)"
                            r"[*_]{0,2}[ \t]*(?::[*_]{0,2}(?:[ \t].*)?|[*_]{0,2})[ \t]*$")
# Место вырезанной ссылки: сначала ссылка заменяется меткой, потом метка убирается вместе с лишними пробелами.
# Подряд идущие метки через запятую («[8], [9]») — одно место; запятая перед ним после оставшейся ссылки
# («[1], [9]») уходит вместе с ним.
_GONE = "\x00"
_GONE_RUN = re.compile(r"(?:(?<=\])[ \t]*[,;])?([ \t]*)\x00(?:[ \t]*[,;]?[ \t]*\x00)*([ \t]*)")
_CLOSING = frozenset(".,;:!?)»…\n")


def _close_gaps(text: str) -> str:
    """Убрать места вырезанных ссылок. Перед знаком препинания, концом и в начале строки пробел не нужен; между
    словами остаётся один пробел. Пробелы в остальном тексте не меняются. Метки ставятся только вне кода, поэтому
    функция идёт по всему тексту: на границе с кодом («`arr[1]` [9] тут») пробел между словами сохраняется."""
    def _sub(match: re.Match[str]) -> str:
        after = match.string[match.end():match.end() + 1]
        before = match.string[match.start() - 1:match.start()] if match.start() else ""
        if not after or after in _CLOSING or not before or before == "\n":
            return ""
        return " " if (match.group(1) or match.group(2)) else ""

    return _GONE_RUN.sub(_sub, text) if _GONE in text else text


@dataclass(frozen=True)
class SourceRef:
    kind: str  # lecture | guide | faq | overview | lms | resource
    ref_id: str  # doc_id, id ресурса или «lms»
    title: str
    section: str = ""
    url: str = ""
    module: int | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.kind, self.ref_id, self.section

    def label(self) -> str:
        if self.kind == "resource":
            return f"[{self.title}]({self.url})" if self.url else self.title
        if self.kind == "lms":
            return self.title
        if self.kind == "faq":
            return f"FAQ: «{self.section}»" if self.section else "FAQ курса"
        base = doc_label(self.kind, self.ref_id, self.title)
        return f"{base}, раздел «{self.section}»" if self.section and self.section != self.title else base

    def journal_entry(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.ref_id, "section": self.section}


@dataclass
class ContextItem:
    n: int
    source: SourceRef
    text: str


@dataclass
class Context:
    items: list[ContextItem] = field(default_factory=list)

    def add(self, source: SourceRef, text: str) -> int:
        """Номер материала; тот же раздел второй раз не заводится — текст дописывается к первому."""
        for item in self.items:
            if item.source.key == source.key:
                if text and text not in item.text:
                    item.text = f"{item.text}\n\n{text}"
                return item.n
        item = ContextItem(len(self.items) + 1, source, text)
        self.items.append(item)
        return item.n

    def ref(self, source: SourceRef, text: str = "") -> str:
        """Добавить и вернуть ссылку « [n]» для шаблона."""
        return f" [{self.add(source, text)}]"

    def find(self, source: SourceRef) -> ContextItem | None:
        return next((item for item in self.items if item.source.key == source.key), None)

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)


def hit_source(hit: Hit) -> SourceRef:
    meta = hit.metadata
    module = meta.get("module")
    return SourceRef(kind=str(meta.get("resource_type", "lecture")), ref_id=hit.doc_id,
                     title=str(meta.get("title", "")), section=hit.section,
                     module=int(module) if isinstance(module, int) else None)


def _chunk_body(text: str) -> str:
    """Текст фрагмента без первой строки «Лекция …. Раздел», которую индексация добавляет впереди."""
    return text.split("\n\n", 1)[-1]


def add_hits(context: Context, hits: list[Hit]) -> None:
    """Фрагменты одного раздела — один материал: разделы в порядке выдачи, фрагменты внутри — по порядку
    в документе. Заголовок «Лекция …. Раздел» у второго и дальше фрагментов раздела не повторяется. Если раздел
    уже в контексте (его положил код — например, шаги методички), дописывается только то, чего в нём ещё нет."""
    groups: dict[tuple[str, str, str], list[Hit]] = {}
    for hit in hits:
        groups.setdefault(hit_source(hit).key, []).append(hit)
    for group in groups.values():
        group.sort(key=lambda h: int(h.metadata.get("chunk", 0)))
        source = hit_source(group[0])
        existing = context.find(source)
        if existing is None:
            texts = [group[0].text] + [_chunk_body(h.text) for h in group[1:]]
            context.add(source, "\n\n".join(texts))
            continue
        for hit in group:
            body = _chunk_body(hit.text)
            if body.strip() not in existing.text:
                context.add(source, body)


def strip_model_sources(text: str) -> str:
    """Отрезать список источников, который модель написала сама, вопреки инструкции."""
    matches = list(_MODEL_SOURCES.finditer(text))
    if matches and matches[-1].start() > len(text) * 0.4:
        return text[: matches[-1].start()].rstrip()
    return text


def _outside_code(text: str, fn: Callable[[str], str]) -> str:
    """Применить fn только к тексту вне блоков и вставок кода: `lst[7]` в коде — не ссылка на источник."""
    pieces: list[str] = []
    last = 0
    for code in _CODE.finditer(text):
        pieces.append(fn(text[last:code.start()]))
        pieces.append(code.group(0))
        last = code.end()
    pieces.append(fn(text[last:]))
    return "".join(pieces)


def process_citations(text: str, count: int, *, keep_lists: bool = True) -> tuple[str, list[int]]:
    """Убрать ссылки на несуществующие номера; вернуть текст и номера в порядке первого упоминания.
    «[1, 2]» и «[1-2]» превращаются в «[1][2]». keep_lists — скобка из нескольких номеров вне 1..count остаётся
    как есть (это список в тексте, «[10, 20, 30]»); False — вырезается и она (слепые пакеты судей убирают всё)."""
    cited: list[int] = []

    def _sub(match: re.Match[str]) -> str:
        numbers = cite_numbers(match.group(1))
        if numbers is None or (keep_lists and _list_literal(numbers, lambda n: 1 <= n <= count)):
            return match.group(0)
        valid = [n for n in numbers if 1 <= n <= count]
        for n in valid:
            if n not in cited:
                cited.append(n)
        return "".join(f"[{n}]" for n in dict.fromkeys(valid)) or _GONE

    marked = _outside_code(text, lambda part: _CITE.sub(_sub, part))
    return _close_gaps(marked), cited


def citation_numbers(text: str, count: int) -> list[int]:
    """Номера источников, на которые ссылается текст (вне кода), по возрастанию."""
    found: set[int] = set()

    def _collect(part: str) -> str:
        for match in _CITE.finditer(part):
            found.update(n for n in cite_numbers(match.group(1)) or [] if 1 <= n <= count)
        return part

    _outside_code(text, _collect)
    return sorted(found)


def sources_block(items: list[ContextItem], empty_text: str = "") -> str:
    lines = ["**Источники**", ""]
    if not items:
        lines.append(empty_text or "в материалах курса ответа не нашлось.")
    for item in items:
        lines.append(f"- [{item.n}] {item.source.label()}")
    return "\n".join(lines)


def used_items(context: Context, cited: list[int]) -> list[ContextItem]:
    """Процитированные материалы; если модель не сослалась ни на один — все, что ей дали."""
    if not cited:
        return list(context.items)
    by_number = {item.n: item for item in context.items}
    return [by_number[n] for n in sorted(cited) if n in by_number]


def _has_grouped(text: str) -> bool:
    """Есть ли вне кода ссылка-перечисление или диапазон («[1, 2]», «[1-3]»), которую надо развернуть."""
    found = False

    def _check(part: str) -> str:
        nonlocal found
        found = found or any(cite_numbers(m.group(1)) is not None and not m.group(1).strip().isdigit()
                             for m in _CITE.finditer(part))
        return part

    _outside_code(text, _check)
    return found


def _first_mentions(text: str) -> list[int]:
    order: list[int] = []

    def _collect(part: str) -> str:
        for match in _CITE.finditer(part):
            for n in cite_numbers(match.group(1)) or []:
                if n not in order:
                    order.append(n)
        return part

    _outside_code(text, _collect)
    return order


def renumber(text: str, items: list[ContextItem]) -> tuple[str, list[ContextItem]]:
    """Номера источников подряд с 1 в порядке первого упоминания в тексте (неупомянутые — следом, по старому
    номеру); ссылки в тексте переписываются по той же карте. Ссылка на номер, которого среди items нет, убирается."""
    by_number = {item.n: item for item in items}
    mentions = _first_mentions(text)
    order = [n for n in mentions if n in by_number]
    order += sorted(n for n in by_number if n not in order)
    mapping = {old: new for new, old in enumerate(order, 1)}
    changed = (any(old != new for old, new in mapping.items()) or any(n not in mapping for n in mentions)
               or _has_grouped(text))

    def _sub(match: re.Match[str]) -> str:
        numbers = cite_numbers(match.group(1))
        if numbers is None or _list_literal(numbers, lambda n: n in mapping):
            return match.group(0)
        return "".join(f"[{mapping[n]}]" for n in dict.fromkeys(numbers) if n in mapping) or _GONE

    new_text = _close_gaps(_outside_code(text, lambda part: _CITE.sub(_sub, part))) if changed else text
    return new_text, [ContextItem(mapping[old], by_number[old].source, by_number[old].text) for old in order]
