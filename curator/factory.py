"""Сборка настоящих зависимостей куратора: OpenAI (LangChain), Chroma, LMS, журнал, лимиты.

Тесты и Streamlit-тесты подменяют make_chat / make_router_llm / make_embeddings фейками — остальная сборка та же.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from .config import MODULE_TITLES, MSK, ConfigError, Settings, load_settings
from .curator import Curator
from .fewshot import FewShotLibrary, check_examples, load_examples
from .ingest import IndexStatus, index_status, open_store
from .journal import Journal
from .kb import KnowledgeBase, load_kb
from .limits import HOUR, DemoLimits
from .lms import LMSClient, make_backend
from .memory import DialogMemory
from .prompts import PromptLibrary
from .recommender import Recommender, load_resources
from .retriever import Retriever
from .router import LLMAnalyzer, RuleAnalyzer, modules_text

ROUTER_MAX_TOKENS = 300
ROUTER_TIMEOUT_S = 15.0


def make_chat(settings: Settings) -> BaseChatModel:
    return ChatOpenAI(model=settings.chat_model, temperature=settings.temperature,
                      max_completion_tokens=settings.max_tokens, timeout=settings.llm_timeout_s, max_retries=2,
                      api_key=SecretStr(settings.openai_api_key))


def make_router_llm(settings: Settings) -> BaseChatModel:
    """Анализ запроса — короткий детерминированный вызов: temperature 0, один повтор, свой таймаут."""
    return ChatOpenAI(model=settings.router_model, temperature=0, max_completion_tokens=ROUTER_MAX_TOKENS,
                      timeout=ROUTER_TIMEOUT_S, max_retries=1, api_key=SecretStr(settings.openai_api_key))


def make_embeddings(settings: Settings) -> Embeddings:
    """Эмбеддинги с таймаутом: без него OpenAIEmbeddings ждёт зависшее соединение бесконечно."""
    return OpenAIEmbeddings(model=settings.embed_model, api_key=SecretStr(settings.openai_api_key), max_retries=2,
                            timeout=settings.embed_timeout_s)


@dataclass
class Services:
    settings: Settings
    curator: Curator
    journal: Journal
    lms: LMSClient
    kb: KnowledgeBase
    index: IndexStatus
    limits: DemoLimits
    problems: list[str] = field(default_factory=list)  # что не так с данными: показать в интерфейсе


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or load_settings()
    if not settings.openai_api_key:
        raise ConfigError("Нет OPENAI_API_KEY: задайте его в .env в корне проекта или в окружении службы.")
    status = index_status(settings.kb_dir, settings.index_dir, settings.embed_model, settings.chunk_size,
                          settings.chunk_overlap)
    if not status.usable or status.manifest is None:
        raise ConfigError(status.message)
    if status.manifest.get("embed_model") != settings.embed_model:
        raise ConfigError(status.message)  # векторы другой модели несравнимы с вопросом — устаревший не годится
    kb = load_kb(settings.kb_dir, strict=False)
    prompts = PromptLibrary()
    embeddings = make_embeddings(settings)
    store = open_store(settings.index_dir, str(status.manifest["collection"]), embeddings)
    retriever = Retriever(store, settings.top_k, settings.min_similarity, settings.min_filtered_hits)
    topics = kb.topics()
    rules = RuleAnalyzer(list(topics), {d.doc_id: d.topic for d in kb.documents if d.resource_type == "lecture"})
    analyzer = LLMAnalyzer(make_router_llm(settings), prompts, modules=modules_text(kb, MODULE_TITLES),
                           topics=topics, fallback=rules, history_turns=settings.history_turns)
    lms = LMSClient(make_backend(settings.lms_url, settings.lms_dir, settings.lms_timeout_s))
    journal = Journal(settings.journal_path, settings.retention_days)
    examples, fewshot_problems = load_examples(settings.fewshot_path)
    problems = list(kb.problems) + fewshot_problems + check_examples(examples)
    try:
        resources = load_resources(settings.resources_path)
    except ValueError as exc:
        resources = []
        problems.append(str(exc))
    now = datetime.now(MSK)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    # Лимиты — по настоящим часам, даже если CURATOR_NOW застыл на демо-дате: иначе окно часа не сдвигалось бы.
    # После перезапуска счётчики берутся из журнала: вопросы, прошедшие лимиты, за сегодня и за последний час.
    limits = DemoLimits(settings.max_question_chars, settings.rate_per_hour, settings.daily_cap,
                        hourly_cap=settings.hourly_cap, daily_used=journal.count_since(midnight),
                        recent=journal.stamps_since(now - HOUR))
    curator = Curator(settings=settings, analyzer=analyzer, retriever=retriever, llm=make_chat(settings), lms=lms,
                      kb=kb, recommender=Recommender(resources, kb), fewshot=FewShotLibrary(examples),
                      prompts=prompts, memory=DialogMemory(turns=max(settings.history_turns, 1)), journal=journal,
                      limits=limits, kb_version=str(status.manifest.get("kb_version", "")))
    return Services(settings, curator, journal, lms, kb, status, limits, problems)
