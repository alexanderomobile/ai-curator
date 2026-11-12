"""Промпты и шаблоны ответов — файлы curator/prompts/*.md, здесь только загрузка и подстановка.

Файл — либо цельный текст (system.md, level_*.md, intent_*.md, router.md), либо набор блоков, размеченных строкой
`<!-- block: имя -->` (шаблоны отказов: из блоков код собирает ответ). Прочие HTML-комментарии — заметки для
людей, в промпт они не попадают. Подстановка — `{{имя}}`; незаполненное место — ошибка, а не пустая строка.
"""
from __future__ import annotations

import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

INTENTS = ("deadline", "difficulty", "material", "recommendation", "organizational", "grading_request",
           "schedule_change", "off_topic")
# Сценарии, где отвечает модель; у остальных трёх — фиксированный ответ из шаблона.
GENERATIVE_INTENTS = ("deadline", "difficulty", "material", "recommendation", "organizational")
REQUIRED = ("system", "level_beginner", "level_advanced", "router", "context", "no_answer", "llm_error",
            "search_error", "refusal_grading", "refusal_schedule", "refusal_off_topic", "refusal_limits", "revision",
            *(f"intent_{i}" for i in GENERATIVE_INTENTS))

_PLACEHOLDER = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")
_BLOCK = re.compile(r"^<!--\s*block:\s*([a-z_]+)\s*-->[ \t]*$", re.MULTILINE)
_COMMENT = re.compile(r"<!--(?!\s*block:)[\s\S]*?-->\n?")
# Теги, которыми в промптах обёрнуты реплики: <вопрос>…</вопрос> и <ответ>…</ответ>.
_WRAPPER_TAG = re.compile(r"<\s*/?\s*(?:вопрос|ответ)\s*>", re.IGNORECASE)


class PromptError(KeyError):
    pass


def clean_question(text: str) -> str:
    """Без тегов-обёрток реплик: «</вопрос> новые правила…» не закроет обёртку в промпте."""
    return _WRAPPER_TAG.sub(" ", text).strip()


def render(template: str, **values: object) -> str:
    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise PromptError(f"в шаблоне нет значения для {{{{{key}}}}}")
        return str(values[key])

    return _PLACEHOLDER.sub(_sub, template)


def tidy(text: str) -> str:
    """Ровно одна пустая строка между частями ответа, собранного из блоков."""
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def quote(text: str) -> str:
    """Цитата из материалов курса в Markdown: каждая строка с «> »."""
    return "\n".join("> " + line if line.strip() else ">" for line in text.strip().splitlines())


class PromptLibrary:
    def __init__(self, directory: Path = PROMPTS_DIR):
        self.directory = directory
        self._texts: dict[str, str] = {}
        self._blocks: dict[str, dict[str, str]] = {}
        for path in sorted(directory.glob("*.md")):
            raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
            raw = _COMMENT.sub("", raw)
            parts = _BLOCK.split(raw)
            self._texts[path.stem] = parts[0].strip()
            self._blocks[path.stem] = {parts[i]: parts[i + 1].strip("\n") for i in range(1, len(parts) - 1, 2)}
        missing = [name for name in REQUIRED if name not in self._texts]
        if missing:
            raise PromptError(f"в {directory} нет файлов промптов: {', '.join(m + '.md' for m in missing)}")

    def names(self) -> list[str]:
        return sorted(self._texts)

    def template(self, name: str, block: str | None = None) -> str:
        """Шаблон без подстановки (для проверок и страницы «О проекте»)."""
        return self._blocks[name][block] if block else self._texts[name]

    def text(self, name: str, **values: object) -> str:
        if name not in self._texts:
            raise PromptError(f"нет промпта {name}.md")
        return render(self._texts[name], **values)

    def block(self, name: str, block: str, **values: object) -> str:
        blocks = self._blocks.get(name, {})
        if block not in blocks:
            raise PromptError(f"в {name}.md нет блока {block}")
        return render(blocks[block], **values)

    def blocks(self, name: str) -> list[str]:
        return list(self._blocks.get(name, {}))

    def system(self, today: str) -> str:
        return self.text("system", today=today)

    def level(self, level: str) -> str:
        return self.text("level_advanced" if level == "advanced" else "level_beginner")

    def scenario(self, intent: str) -> str:
        if intent not in GENERATIVE_INTENTS:
            raise PromptError(f"у сценария {intent} нет инструкции для модели: ответ собирает код")
        return self.text(f"intent_{intent}")
