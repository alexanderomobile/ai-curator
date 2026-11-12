"""Общее для страниц Streamlit: кэш сервисов, подписи, оформление графиков, ключ посетителя для лимитов.

Страницы Аналитика и LMS не требуют ключа OpenAI и индекса: журнал и LMS собираются отдельно от куратора.
Сервисы куратора пересобираются сами, когда переиндексация (python -m curator.ingest) переключила коллекцию.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import sys
import uuid
from collections.abc import Collection, Mapping
from numbers import Real
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Any  # noqa: E402

import streamlit as st  # noqa: E402

from curator import factory  # noqa: E402
from curator.config import MODULE_TITLES, ConfigError, Settings, load_settings  # noqa: E402
from curator.ingest import read_manifest  # noqa: E402
from curator.journal import Journal  # noqa: E402
from curator.kb import KBFormatError, KnowledgeBase, load_kb  # noqa: E402
from curator.lms import LMSClient, make_backend  # noqa: E402
from curator.privacy import user_hash  # noqa: E402

LEVEL_OPTIONS = {"Авто": "auto", "Начинающий": "beginner", "Продвинутый": "advanced"}
LEVEL_LABELS = {"beginner": "начинающий", "advanced": "продвинутый"}
LEVEL_SOURCE_LABELS = {"manual": "выбран вручную", "hint": "по вопросу", "student": "уровень студента",
                       "default": "по умолчанию"}
TYPE_OPTIONS = {"Любые": None, "Лекции": "lecture", "Методички ДЗ": "guide", "FAQ": "faq", "О курсе": "overview"}
# Категориальная палитра (dataviz, references/palette.md): порядок слотов фиксирован, цвет закреплён за сущностью.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
# Прокси, от которых принимаются заголовки с адресом посетителя (client_address). Читается здесь, а не в
# curator.config: настройка нужна только интерфейсу. .env подхватывается — его загружает импорт curator.config.
TRUSTED_PROXIES_ENV = "CURATOR_TRUSTED_PROXIES"
DEFAULT_TRUSTED_PROXIES = ("127.0.0.1", "::1")

log = logging.getLogger("curator.app")


def settings() -> Settings:
    return load_settings()


def module_label(module: int | None) -> str:
    return "Все модули" if module is None else f"{module}. {MODULE_TITLES.get(module, '')}"


def _index_key(s: Settings) -> str:
    manifest = read_manifest(s.index_dir) or {}
    return f"{s.index_dir}|{manifest.get('collection', '')}|{s.data_dir}|{s.journal_path}"


@st.cache_resource(show_spinner="Загружаю базу знаний…")
def _services(key: str) -> factory.Services:  # key — только для кэша: новая коллекция → новые сервисы
    return factory.build_services(load_settings())


def services() -> factory.Services | None:
    """Сервисы куратора или None (сообщение об ошибке уже показано)."""
    s = load_settings()
    try:
        return _services(_index_key(s))
    except ConfigError as exc:
        st.error(str(exc))
        return None


@st.cache_resource
def journal(path: str, retention_days: int) -> Journal:
    return Journal(Path(path), retention_days)


@st.cache_resource
def lms(url: str, directory: str, timeout: float) -> LMSClient:
    return LMSClient(make_backend(url, Path(directory), timeout))


@st.cache_resource
def knowledge_base(kb_dir: str, stamp: str) -> KnowledgeBase | None:  # stamp — версия индекса для кэша
    try:
        return load_kb(Path(kb_dir), strict=False)
    except KBFormatError:
        return None


def current_journal() -> Journal:
    s = load_settings()
    return journal(str(s.journal_path), s.retention_days)


def current_lms() -> LMSClient:
    s = load_settings()
    return lms(s.lms_url, str(s.lms_dir), s.lms_timeout_s)


def _canonical_ip(value: str) -> str | None:
    """Адрес в одном написании (IPv4 внутри IPv6 «::ffff:a.b.c.d» — как IPv4) или None, если это не IP-адрес."""
    try:
        address = ipaddress.ip_address(value.strip())
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return str(address.ipv4_mapped)
    return str(address)


def parse_trusted_proxies(raw: str) -> frozenset[str]:
    """Список адресов через запятую. Не IP-адрес (опечатка, сеть вида 10.0.0.0/8) пропускается с предупреждением
    в журнал службы: доверять заголовкам от непонятно кого нельзя."""
    trusted: set[str] = set()
    for part in raw.split(","):
        if not part.strip():
            continue
        address = _canonical_ip(part)
        if address is None:
            log.warning("%s: «%s» — не IP-адрес, пропущен", TRUSTED_PROXIES_ENV, part.strip())
            continue
        trusted.add(address)
    return frozenset(trusted)


def trusted_proxies() -> frozenset[str]:
    """Доверенные прокси из CURATOR_TRUSTED_PROXIES (через запятую). Не задано или пусто — 127.0.0.1 и ::1;
    заданный список заменяет значение по умолчанию."""
    raw = os.environ.get(TRUSTED_PROXIES_ENV, "").strip()
    return parse_trusted_proxies(raw or ",".join(DEFAULT_TRUSTED_PROXIES))


def client_address(headers: Mapping[str, str], ip_address: object,
                   trusted: Collection[str] | None = None) -> str | None:
    """Адрес посетителя — ключ почасового лимита.

    Демо открыто через Cloudflare Tunnel: cloudflared живёт в другом контейнере и подключается к Streamlit по LAN
    со своего адреса — он и есть st.context.ip_address. Настоящий адрес посетителя приходит в заголовках:
    Cf-Connecting-Ip (его ставит Cloudflare), затем X-Forwarded-For — последний адрес цепочки (его дописал ближайший
    прокси, а начало цепочки может прислать сам посетитель).

    Заголовкам верим, только если соединение пришло от доверенного прокси (trusted, по умолчанию
    CURATOR_TRUSTED_PROXIES) или с localhost: Streamlit отдаёт тогда ip_address = None. Иначе заголовки
    игнорируются и ключом служит адрес соединения: Streamlit слушает LAN, и клиент из LAN, подключившись напрямую,
    подделал бы Cf-Connecting-Ip и получал бы новый лимит на каждый вопрос. Сам адрес соединения клиент из LAN
    не подменит: сервер Streamlit (uvicorn) переписывает его из X-Forwarded-For только для соединений с 127.0.0.1
    и ::1 (FORWARDED_ALLOW_IPS по умолчанию; в службе её не задавать).

    ip_address не строка и не None (заглушка в тестах AppTest) — источник неизвестен: адреса нет, заголовкам
    не верим, сработает запасной ключ вкладки."""
    if ip_address is None:
        peer = None
    elif isinstance(ip_address, str) and ip_address.strip():
        peer = ip_address.strip()
        allowed = trusted_proxies() if trusted is None else parse_trusted_proxies(",".join(trusted))
        if _canonical_ip(peer) not in allowed:
            return peer
    else:
        return None
    lower = {k.lower(): v for k, v in headers.items()}
    cf = lower.get("cf-connecting-ip", "").strip()
    if cf:
        return cf
    chain = [part.strip() for part in lower.get("x-forwarded-for", "").split(",") if part.strip()]
    if chain:
        return chain[-1]
    return peer


def limit_key(salt: str) -> str:
    """Ключ почасового лимита: адрес посетителя (обезличенный), иначе — идентификатор вкладки, который «Новый
    диалог» не меняет. Адрес в журнал не пишется и живёт только в памяти лимитов."""
    try:
        address = client_address(dict(st.context.headers), st.context.ip_address)
    except Exception:  # контекста запроса нет (например, в тестах без сервера)
        address = None
    if address:
        return "ip:" + user_hash(salt, address)
    st.session_state.setdefault("limit_id", uuid.uuid4().hex)
    return "tab:" + str(st.session_state["limit_id"])


def current_kb() -> KnowledgeBase | None:
    s = load_settings()
    manifest = read_manifest(s.index_dir) or {}
    return knowledge_base(str(s.kb_dir), str(manifest.get("kb_version", "")))


def style(fig: Any, height: int = 320, counts: str | None = "y") -> Any:
    """Тонкие столбцы со скруглением, сетка только по значениям, подписи осей — цветом текста темы.
    counts — ось со штуками (вопросы, ответы): деления только целые, без «0,5 вопроса»."""
    fig.update_layout(height=height, margin={"l": 8, "r": 8, "t": 36, "b": 8}, bargap=0.35, barcornerradius=4,
                      legend={"orientation": "h", "y": -0.2, "title": None}, hovermode="closest")
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(title=None, zeroline=False)
    if counts:
        top = 0.0
        for trace in fig.data:
            values = getattr(trace, counts, None)  # у plotly это массив numpy или None
            numbers = [float(v) for v in (list(values) if values is not None else []) if isinstance(v, Real)]
            top += max(numbers, default=0.0)  # для столбцов с накоплением — верхняя оценка суммы
        update = fig.update_yaxes if counts == "y" else fig.update_xaxes
        update(tickformat=",d", rangemode="tozero", **({"dtick": 1} if top <= 8 else {}))
    if counts == "x":
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=False)
    return fig
