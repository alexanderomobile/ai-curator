"""Индексация (манифест, устаревание, пересборка), путь индекса на Windows, поиск с фильтром по метаданным."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from curator import config
from curator.config import ConfigError, default_index_dir, ensure_index_dir_usable, path_is_safe_for_chroma
from curator.ingest import build_index, chroma_client, index_status, open_store, read_manifest
from curator.kb import file_hashes
from curator.retriever import Retriever, SearchFilter
from tests.conftest import EMBED_MODEL, FIXTURES
from tests.fakes import TrigramEmbeddings


@pytest.fixture(scope="module")
def det_index(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    """Индекс на DeterministicFakeEmbedding из langchain_core: близость случайна, проверяются только фильтры."""
    index_dir = tmp_path_factory.mktemp("det-index")
    manifest = build_index(FIXTURES / "kb", index_dir, DeterministicFakeEmbedding(size=64), embed_model="det",
                           chunk_size=1200, chunk_overlap=150)
    return index_dir, manifest


def det_retriever(det_index: tuple[Path, dict[str, Any]], min_hits: int = 2, top_k: int = 5) -> Retriever:
    index_dir, manifest = det_index
    store = open_store(index_dir, manifest["collection"], DeterministicFakeEmbedding(size=64))
    return Retriever(store, top_k=top_k, min_similarity=-1.0, min_filtered_hits=min_hits)


def test_manifest_records_files_model_and_chunks(fixture_index: tuple[Path, dict[str, Any]]):
    index_dir, manifest = fixture_index
    assert read_manifest(index_dir) == manifest
    assert manifest["files"] == file_hashes(FIXTURES / "kb")
    assert manifest["embed_model"] == EMBED_MODEL
    assert manifest["documents"] == 5 and manifest["chunks"] > 10
    assert set(manifest["by_type"]) == {"lecture", "guide", "faq", "overview"}
    status = index_status(FIXTURES / "kb", index_dir, EMBED_MODEL, 1200, 150)
    assert status.ok, status.message


def test_index_status_detects_stale_and_missing(tmp_path: Path):
    kb_dir = tmp_path / "kb"
    shutil.copytree(FIXTURES / "kb", kb_dir)
    index_dir = tmp_path / "index"
    assert index_status(kb_dir, index_dir, "m", 1200, 150).state == "missing"
    build_index(kb_dir, index_dir, TrigramEmbeddings(), embed_model="m", chunk_size=1200, chunk_overlap=150)
    assert index_status(kb_dir, index_dir, "m", 1200, 150).ok

    lecture = kb_dir / "module-2" / "lecture-2.2.md"
    lecture.write_text(lecture.read_text(encoding="utf-8") + "\nНовый абзац.\n", encoding="utf-8")
    (kb_dir / "module-1" / "lecture-1.1.md").unlink()
    status = index_status(kb_dir, index_dir, "m", 1200, 150)
    assert status.state == "stale" and status.usable
    assert status.changed == ["module-2/lecture-2.2.md"] and status.removed == ["module-1/lecture-1.1.md"]
    assert "python -m curator.ingest" in status.message

    assert index_status(kb_dir, index_dir, "другая-модель", 1200, 150).state == "stale"
    assert index_status(kb_dir, index_dir, "m", 800, 150).state == "stale"


def test_rebuild_switches_collection_and_cleans_old(tmp_path: Path):
    index_dir = tmp_path / "index"
    first = build_index(FIXTURES / "kb", index_dir, TrigramEmbeddings(), embed_model="m", chunk_size=1200,
                        chunk_overlap=150)
    second = build_index(FIXTURES / "kb", index_dir, TrigramEmbeddings(), embed_model="m", chunk_size=1200,
                         chunk_overlap=150, prefix="kb")
    assert first["collection"] != second["collection"]
    names = [c.name for c in chroma_client(index_dir).list_collections()]
    assert names == [second["collection"]]
    store = open_store(index_dir, second["collection"], TrigramEmbeddings())
    assert len(store.get()["ids"]) == second["chunks"]


def test_failed_rebuild_keeps_working_index(tmp_path: Path):
    index_dir = tmp_path / "index"
    good = build_index(FIXTURES / "kb", index_dir, TrigramEmbeddings(), embed_model="m", chunk_size=1200,
                       chunk_overlap=150)

    class Broken(TrigramEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise ConnectionError("сеть упала посреди индексации")

    with pytest.raises(ConnectionError):
        build_index(FIXTURES / "kb", index_dir, Broken(), embed_model="m", chunk_size=1200, chunk_overlap=150)
    assert read_manifest(index_dir) == good
    assert [c.name for c in chroma_client(index_dir).list_collections()] == [good["collection"]]


def test_filter_by_module_keeps_module_and_general_docs(det_index: tuple[Path, dict[str, Any]]):
    result = det_retriever(det_index, top_k=50).search("маски", SearchFilter(module=2))
    assert result.hits and not result.relaxed
    assert {h.metadata["module"] for h in result.hits} <= {2, 0}
    assert any(h.metadata["module"] == 2 for h in result.hits)


def test_filter_by_resource_type_and_topic(det_index: tuple[Path, dict[str, Any]]):
    r = det_retriever(det_index, top_k=50)
    faq = r.search("перенос", SearchFilter(resource_types=("faq",)))
    assert faq.hits and {h.metadata["resource_type"] for h in faq.hits} == {"faq"}
    one = det_retriever(det_index, min_hits=1, top_k=50)
    topic = one.search("перенос", SearchFilter(module=0, resource_types=("faq",), topic="extensions"))
    assert [h.metadata["section"] for h in topic.hits] == ["Можно ли перенести дедлайн?"]
    assert not topic.relaxed


def test_filter_relaxes_when_too_few_hits(det_index: tuple[Path, dict[str, Any]]):
    result = det_retriever(det_index).search("графики", SearchFilter(module=3, resource_types=("lecture",),
                                                                    topic="matplotlib-basics"))
    assert result.relaxed
    assert result.filter_used == SearchFilter(module=3)  # шаг «только модуль»: FAQ и обзор (module=0) находятся
    assert result.steps[0].endswith(": 0")
    assert result.hits


def test_threshold_drops_unrelated(retriever: Retriever):
    assert retriever.search("Можно ли перенести дедлайн?").hits[0].section == "Можно ли перенести дедлайн?"
    assert retriever.search("zzzz qqqq xxxx").hits == []


def test_several_queries_take_best_score(retriever: Retriever):
    alone = retriever.search("zzzz qqqq xxxx")
    both = retriever.search(["zzzz qqqq xxxx", "Можно ли перенести дедлайн?"])
    assert alone.hits == [] and both.hits[0].section == "Можно ли перенести дедлайн?"


def test_where_syntax():
    assert SearchFilter().where() is None
    assert SearchFilter(module=2).where() == {"module": {"$in": [2, 0]}}
    assert SearchFilter(module=2, resource_types=("lecture",), topic="t").where() == {"$and": [
        {"module": {"$in": [2, 0]}}, {"resource_type": {"$in": ["lecture"]}}, {"topic": "t"}]}
    assert SearchFilter(module=1, resource_types=("faq",)).relaxed() == [
        SearchFilter(1, ("faq",)), SearchFilter(1), SearchFilter()]


def test_cyrillic_index_path_rejected_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(config, "_on_windows", lambda: True)  # не os.name: от него зависит класс путей pathlib
    assert not path_is_safe_for_chroma(Path("D:/Учебный проект/data/index"))
    with pytest.raises(ConfigError, match="CURATOR_INDEX_DIR"):
        ensure_index_dir_usable(Path("D:/Учебный проект/data/index"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    fallback = default_index_dir(Path("D:/Учебный проект/ai-curator"))
    assert str(fallback).isascii() and "Учебный" not in str(fallback)
    assert default_index_dir(Path("D:/ascii/ai-curator")) == Path("D:/ascii/ai-curator/data/index")


def test_any_path_allowed_on_linux(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config.os, "name", "posix")
    assert path_is_safe_for_chroma(Path("/opt/Учебный проект/index"))


@pytest.mark.skipif(os.name != "nt", reason="проверка ограничения hnswlib на Windows")
def test_project_default_index_dir_is_ascii_on_windows():
    assert str(default_index_dir().resolve()).isascii()
