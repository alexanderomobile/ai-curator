"""Настройки куратора. Секреты и пути читаются из окружения: `.env` в корне проекта или EnvironmentFile службы.

Время курса — московское (МСК): дни до дедлайна, «сегодня» в лимитах и в отчёте считаются в Europe/Moscow.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

# На Windows zoneinfo берёт базу часовых поясов из пакета tzdata (он в requirements.in).
MSK = ZoneInfo("Europe/Moscow")

SOURCES = ("live", "eval", "demo")  # live — живые посетители, eval — прогоны оценки, demo — кнопки-примеры
# Модули курса (BRIEF, раздел 1): для выбора в интерфейсе и для промпта анализа запроса.
MODULE_TITLES = {1: "Основы Python", 2: "Pandas: работа с таблицами", 3: "Визуализация", 4: "Итоговый проект"}


class ConfigError(RuntimeError):
    """Настройка, без которой куратор не запустится; текст — что именно поправить."""


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def _on_windows() -> bool:
    """Отдельная функция, чтобы тест мог изобразить Windows, не подменяя os.name: от os.name зависит, какой класс
    путей создаёт pathlib, и подмена ломает Path() на Linux."""
    return os.name == "nt"


def path_is_safe_for_chroma(path: Path) -> bool:
    """Можно ли держать индекс Chroma по этому пути.

    На Windows hnswlib внутри Chroma открывает файлы узкими (ANSI) строками: в папку с кириллицей он молча не пишет
    data_level0.bin и остальные файлы индекса, а при следующем открытии Chroma падает с «Error loading hnsw index».
    Проверено на chromadb 1.5.9: в `PEcf13. Учебный проект ИИ-куратор\\...` индекс не открывается, в ASCII-пути — да.
    На Linux путь в UTF-8, ограничения нет.
    """
    return not _on_windows() or str(path).isascii()


def default_index_dir(base_dir: Path = BASE_DIR) -> Path:
    """data/index в репозитории, а если путь к нему не ASCII (Windows) — папка вне репозитория."""
    inside = base_dir / "data" / "index"
    if path_is_safe_for_chroma(inside.resolve()):
        return inside
    candidates: list[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "ai-curator" / "index")
    candidates.append(Path(Path.home().anchor or "C:\\") / "ai-curator-index")
    for candidate in candidates:
        if path_is_safe_for_chroma(candidate):
            return candidate
    return candidates[-1]


def ensure_index_dir_usable(path: Path) -> None:
    if not path_is_safe_for_chroma(path.resolve()):
        raise ConfigError(
            f"Путь к индексу «{path}» содержит не-ASCII символы. На Windows Chroma (hnswlib) не может записать "
            "в такую папку файлы индекса. Задайте переменную CURATOR_INDEX_DIR с ASCII-путём, "
            "например CURATOR_INDEX_DIR=C:\\ai-curator-index, и пересоберите индекс: python -m curator.ingest"
        )


def _now_override() -> datetime | None:
    """CURATOR_NOW=2026-11-12T12:00 — «текущее время» для демо и записи скриншотов; без зоны считается МСК."""
    value = os.environ.get("CURATOR_NOW", "").strip()
    if not value:
        return None
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo else moment.replace(tzinfo=MSK)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""
    chat_model: str = "gpt-4.1-mini"
    router_model: str = "gpt-4.1-mini"
    embed_model: str = "text-embedding-3-large"
    temperature: float = 0.2
    max_tokens: int = 900
    llm_timeout_s: float = 40.0
    embed_timeout_s: float = 15.0
    data_dir: Path = BASE_DIR / "data"
    index_dir: Path = field(default_factory=default_index_dir)
    collection_prefix: str = "kb"
    journal_path: Path = BASE_DIR / "data" / "journal.db"
    exports_dir: Path = BASE_DIR / "data" / "lms" / "exports"
    lms_url: str = ""
    lms_timeout_s: float = 5.0
    chunk_size: int = 1200
    chunk_overlap: int = 150
    top_k: int = 5
    min_similarity: float = 0.30
    min_filtered_hits: int = 2
    history_turns: int = 2
    max_question_chars: int = 1500
    rate_per_hour: int = 20
    hourly_cap: int = 60
    daily_cap: int = 300
    retention_days: int = 90
    user_hash_salt: str = ""
    now_override: datetime | None = None
    show_live_texts: bool = False
    default_source: str = "live"
    price_input_per_1m: float = 0.40
    price_output_per_1m: float = 1.60
    public_contacts: tuple[str, ...] = ("support@academy-data.example",)
    repo_url: str = "https://github.com/alexanderomobile/ai-curator"

    @property
    def kb_dir(self) -> Path:
        return self.data_dir / "kb"

    @property
    def lms_dir(self) -> Path:
        return self.data_dir / "lms"

    @property
    def resources_path(self) -> Path:
        return self.data_dir / "resources" / "resources.csv"

    @property
    def fewshot_path(self) -> Path:
        return self.data_dir / "fewshot" / "examples.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def now(self) -> datetime:
        """Текущее время в МСК (или CURATOR_NOW)."""
        return self.now_override.astimezone(MSK) if self.now_override else datetime.now(MSK)

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.price_input_per_1m + completion_tokens * self.price_output_per_1m) / 1_000_000


def load_settings() -> Settings:
    chat_model = os.environ.get("CHAT_MODEL", "gpt-4.1-mini")
    data_dir = _path("CURATOR_DATA_DIR", BASE_DIR / "data")
    source = os.environ.get("CURATOR_SOURCE", "live")
    if source not in SOURCES:
        raise ConfigError(f"CURATOR_SOURCE={source!r}: допустимо {', '.join(SOURCES)}")
    return Settings(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        chat_model=chat_model,
        router_model=os.environ.get("ROUTER_MODEL", chat_model),
        # Смена модели эмбеддингов требует пересборки индекса: ingest пишет модель в манифест, фабрика сверяет.
        embed_model=os.environ.get("EMBED_MODEL", "text-embedding-3-large"),
        temperature=_float("TEMPERATURE", 0.2),
        # Ответ начинающему с шагами и кодом длиннее, чем в FAQ-ассистенте (там 400).
        max_tokens=_int("MAX_TOKENS", 900),
        llm_timeout_s=_float("LLM_TIMEOUT_S", 40.0),
        # Эмбеддинги без таймаута (у OpenAIEmbeddings по умолчанию None — «ждать вечно») блокировали бы сессию
        # на зависшем соединении. Запрос эмбеддинга вопроса занимает доли секунды; 15 с с запасом хватает и на пачку
        # из 64 фрагментов при индексации (curator.ingest берёт те же настройки). Повторов — 2, как у модели ответа.
        embed_timeout_s=_float("EMBED_TIMEOUT_S", 15.0),
        data_dir=data_dir,
        index_dir=_path("CURATOR_INDEX_DIR", default_index_dir()),
        journal_path=_path("CURATOR_JOURNAL", data_dir / "journal.db"),
        exports_dir=_path("CURATOR_EXPORTS_DIR", data_dir / "lms" / "exports"),
        # Пусто — LMS читается из файлов data/lms; адрес — из HTTP-заглушки (curator/lms_stub_api.py).
        lms_url=os.environ.get("LMS_URL", "").strip().rstrip("/"),
        lms_timeout_s=_float("LMS_TIMEOUT_S", 5.0),
        # Лекция 700-1200 слов, раздел `##` — 300-1800 знаков (медиана ~760): 1200 знаков дают почти всегда один
        # фрагмент на раздел, длиннее 1200 знаков — 13 из 110 разделов лекций, они режутся на два.
        chunk_size=_int("CHUNK_SIZE", 1200),
        chunk_overlap=_int("CHUNK_OVERLAP", 150),
        top_k=_int("TOP_K", 5),
        # Порог косинусной близости для text-embedding-3-large: взят из FAQ-ассистента (PEcf11, eval/retrieval.py:
        # у вопросов с ответом в базе лучшая оценка не ниже 0,316). Для этой базы уточняется на этапе eval/.
        min_similarity=_float("MIN_SIMILARITY", 0.30),
        # Только для фильтра интерфейса (модуль, тип, тема, выбранные студентом): если в нём нашлось меньше
        # фрагментов выше порога, он ослабляется по шагам — без темы, только модуль, без фильтра. Модуль, тип и тема
        # из анализа запроса — предпочтение при ранжировании, к ним этот порог не относится (curator/retriever.py).
        min_filtered_hits=_int("MIN_FILTERED_HITS", 2),
        history_turns=_int("HISTORY_TURNS", 2),
        # Лимиты демо: студент может вставить в вопрос трейсбек или кусок кода, поэтому длина больше, чем в FAQ (1000).
        max_question_chars=_int("MAX_QUESTION_CHARS", 1500),
        # Вопросов в час на посетителя (адрес клиента из заголовков туннеля), а не на диалог: «Новый диалог» и
        # перезагрузка страницы лимит не сбрасывают.
        rate_per_hour=_int("RATE_PER_HOUR", 20),
        # Общий потолок в час на всех посетителей: дневной лимит нельзя выбрать за минуты (300 / 60 = не меньше
        # пяти часов), даже если адрес клиента подменяют.
        hourly_cap=_int("HOURLY_CAP", 60),
        daily_cap=_int("DAILY_CAP", 300),
        retention_days=_int("RETENTION_DAYS", 90),
        # Без соли код сессии всё равно не обратим (идентификатор сессии случайный), но меняется после перезапуска.
        user_hash_salt=os.environ.get("USER_HASH_SALT", "") or secrets.token_hex(16),
        now_override=_now_override(),
        # На публичном демо тексты вопросов живых посетителей не показываются — только агрегаты.
        show_live_texts=_flag("SHOW_LIVE_TEXTS", False),
        default_source=source,
        # Цены gpt-4.1-mini за 1 млн токенов (USD) на момент написания; при смене модели — поправить в окружении.
        price_input_per_1m=_float("PRICE_INPUT_PER_1M", 0.40),
        price_output_per_1m=_float("PRICE_OUTPUT_PER_1M", 1.60),
    )
