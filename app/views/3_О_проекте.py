"""О проекте: как устроен куратор, что умеет и чего не делает, состояние индекса, ссылки."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import common  # noqa: E402
from curator.ingest import index_status  # noqa: E402
from curator.prompts import PromptLibrary  # noqa: E402
from curator.retriever import MODULE_BONUS, PREFER_BONUS  # noqa: E402

# Документация проекта: пути в репозитории (тест проверяет, что файлы есть) и описание архитектуры в Google Docs.
ARCHITECTURE_DOC_URL = "https://docs.google.com/document/d/1W8aHKrNGa0gk_KIkr4i4ijbSL2v7D042FHnHj1NocvU/edit"
DOCS = [("docs/architecture.md", "архитектура промптов и RAG"),
        ("docs/kb-update.md", "обновление базы знаний и переиндексация"),
        ("docs/analytics.md", "логирование запросов и структура отчёта"),
        ("docs/lms-stub.md", "заглушка LMS: API и выгрузка JSON/CSV"),
        ("docs/diagram.svg", "схема: промпты и RAG")]


def _bonus(value: float) -> str:
    """Прибавка к близости из curator/retriever.py — как на странице: «+0,05»."""
    return f"+{value:.2f}".replace(".", ",")


st.set_page_config(page_title="О проекте", page_icon="ℹ️", layout="wide")
st.title("О проекте")

settings = common.settings()
st.markdown(
    "**ИИ-куратор** отвечает студентам онлайн-курса «Python для анализа данных» (вымышленная школа «Академия "
    "Данных») на вопросы по материалам, срокам и правилам курса и подсказывает, что изучить дальше, с учётом "
    "уровня подготовки. Учебный проект; все данные — вымышленные."
)

st.subheader("Как устроен ответ")
st.markdown(
    "1. **Лимиты демо и обезличивание.** Длина вопроса, вопросов в час на посетителя, общий потолок в час и в день; "
    "почта, ники и ссылки Telegram, телефоны и длинные номера, номера документов и имя после «меня зовут» "
    "вырезаются до отправки в OpenAI и до записи в журнал.\n"
    "2. **Анализ запроса.** Модель со структурированным выводом определяет сценарий (дедлайн, сложность, материал, "
    "рекомендация, организационный, просьба оценить, просьба перенести, не по курсу), модуль, тему, где искать "
    "и подсказку уровня. Последние реплики диалога учитываются. Если модель не ответила — правила.\n"
    "3. **Ограничения кодом.** Просьба оценить, перенести срок и вопрос не по курсу получают фиксированный ответ "
    "из шаблона: отказ и что можно сделать (критерии из методички, правило переноса из FAQ, окно подачи заявки "
    "по данным LMS). Модель ответа в этих сценариях не вызывается.\n"
    "4. **Данные LMS.** Для вопросов о сроках код сам берёт задания и события и считает, сколько осталось "
    "до дедлайна (дни и часы), по московскому времени.\n"
    "5. **Поиск (RAG).** Chroma, эмбеддинги OpenAI text-embedding-3-large, фрагменты по заголовкам markdown с "
    "метаданными (модуль, тема, тип ресурса, уровень, раздел). Модуль, тип и тема из настроек слева — жёсткий "
    "фильтр (модуль слева уступает модулю, прямо названному в вопросе: «hw3», «модуль 2»); если в нём нашлось "
    f"меньше {settings.min_filtered_hits} фрагментов, он ослабляется: без темы → только модуль → без фильтра. "
    "Модуль, типы и тема из анализа запроса — не фильтр, а приоритет: таким фрагментам "
    f"прибавка к близости ({_bonus(PREFER_BONUS)}, совпал только модуль — {_bonus(MODULE_BONUS)}), и заметно "
    "более близкий фрагмент вне них тоже попадёт в ответ. Эмбеддинг вопроса считается один раз, все шаги поиска "
    "идут по готовому вектору. Ниже порога близости — «в материалах курса этого нет».\n"
    "6. **Рекомендации.** 2-3 пункта: ресурсы из каталога по уровню, модулю и теме и разделы лекций "
    "«Если сложно» / «Для продвинутых».\n"
    "7. **Ответ.** gpt-4.1-mini: роль наставника, инструкция уровня, сценарий, 2-3 примера хороших и плохих "
    "ответов, материалы с номерами [1], [2]. Блок «Источники» в конце собирает код.\n"
    f"8. **Журнал и аналитика.** SQLite, {settings.retention_days} дней, отчёт по темам: "
    "`python -m curator.report --days 7`."
)

diagram = ROOT / "docs" / "diagram.svg"
if diagram.is_file():
    st.image(str(diagram), caption="Схема: промпты и RAG")

st.subheader("Состояние")
status = index_status(settings.kb_dir, settings.index_dir, settings.embed_model, settings.chunk_size,
                      settings.chunk_overlap)
(st.success if status.ok else st.warning)(status.message)
kb = common.current_kb()
if kb:
    by_type: dict[str, int] = {}
    for doc in kb.documents:
        by_type[doc.resource_type] = by_type.get(doc.resource_type, 0) + 1
    st.markdown("База знаний: " + ", ".join(f"{k} — {v}" for k, v in sorted(by_type.items()))
                + f"; тем: {len(kb.topics())}.")
    for problem in kb.problems:
        st.warning(problem)
st.markdown(
    f"Модель ответа: `{settings.chat_model}` (temperature {settings.temperature}), анализ запроса: "
    f"`{settings.router_model}`, эмбеддинги: `{settings.embed_model}`. Порог близости {settings.min_similarity}, "
    f"фрагментов в ответе до {settings.top_k}. LMS: {'HTTP ' + settings.lms_url if settings.lms_url else 'файлы'}."
)
st.markdown(f"Промпты (`curator/prompts/`): {', '.join(f'`{n}.md`' for n in PromptLibrary().names())}.")

st.subheader("Обновление базы знаний")
st.markdown(
    "Материалы — markdown-файлы в `data/kb` с front-matter (id, title, module, topic, resource_type, level). "
    "После правки базы индекс пересобирается одной командой, приложение подхватывает новую коллекцию само:\n\n"
    "```bash\n"
    "python -m curator.ingest           # полная пересборка индекса Chroma\n"
    "python -m curator.ingest --check   # только проверить, не устарел ли индекс\n"
    "```\n\n"
    f"Подробно — в [docs/kb-update.md]({settings.repo_url}/blob/main/docs/kb-update.md)."
)

st.subheader("Ссылки")
st.markdown("\n".join([f"- Репозиторий: {settings.repo_url}",
                       f"- [Архитектура промптов и RAG-схема (Google Docs)]({ARCHITECTURE_DOC_URL})"]
                      + [f"- [{title}]({settings.repo_url}/blob/main/{path})" for path, title in DOCS]))
