"""Индексация базы знаний в Chroma и проверка, не устарел ли индекс.

    python -m curator.ingest           # полная пересборка: проверка базы → новая коллекция → манифест → старые удаляются
    python -m curator.ingest --check   # только сверить индекс с data/kb (код выхода 1, если устарел или его нет)

Пересборка всегда полная: база маленькая (десятки файлов), а частичное обновление легко оставляет в индексе
фрагменты удалённого раздела. Новая коллекция собирается рядом со старой под своим именем; манифест переключается
на неё, только когда все фрагменты записаны. Если эмбеддинги упали на середине, рабочий индекс остаётся прежним.

Манифест (index_dir/manifest.json) хранит хеши файлов базы: по нему приложение и `--check` видят, что база
поменялась после сборки индекса. Модель эмбеддингов тоже записана: индекс, собранный другой моделью, непригоден.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from .config import MSK, ConfigError, ensure_index_dir_usable
from .kb import chunk_documents, file_hashes, load_kb

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
BATCH = 64
# Косинусная близость: similarity = 1 - distance, порог MIN_SIMILARITY задаётся в этих единицах.
COLLECTION_CONFIGURATION: dict[str, Any] = {"hnsw": {"space": "cosine"}}
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def chroma_client(index_dir: Path) -> ClientAPI:
    """Один и тот же набор настроек везде: Chroma не даёт открыть одну папку с разными настройками в одном процессе.
    Телеметрия выключена — ни приложение, ни тесты не ходят в сеть ради Chroma."""
    ensure_index_dir_usable(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(index_dir),
                                     settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False))


def open_store(index_dir: Path, collection: str, embeddings: Embeddings) -> Chroma:
    return Chroma(client=chroma_client(index_dir), collection_name=collection, embedding_function=embeddings,
                  collection_configuration=COLLECTION_CONFIGURATION,  # pyright: ignore[reportArgumentType]
                  create_collection_if_not_exists=False)


def read_manifest(index_dir: Path) -> dict[str, Any] | None:
    path = index_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def kb_version(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()[:10]


@dataclass
class IndexStatus:
    state: str  # ok | missing | stale
    message: str
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    manifest: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.state == "ok"

    @property
    def usable(self) -> bool:
        """Устаревший индекс работает (отвечает по старой версии базы), отсутствующий — нет."""
        return self.state != "missing"


def index_status(kb_dir: Path, index_dir: Path, embed_model: str, chunk_size: int, chunk_overlap: int) -> IndexStatus:
    manifest = read_manifest(index_dir)
    if manifest is None or not manifest.get("collection"):
        return IndexStatus("missing", f"Индекса нет в {index_dir}. Соберите его: python -m curator.ingest")
    if manifest.get("embed_model") != embed_model:
        return IndexStatus("stale", f"Индекс собран моделью {manifest.get('embed_model')}, а в настройках "
                                    f"{embed_model}. Пересоберите: python -m curator.ingest", manifest=manifest)
    if (manifest.get("chunk_size"), manifest.get("chunk_overlap")) != (chunk_size, chunk_overlap):
        return IndexStatus("stale", "Поменялись настройки нарезки (CHUNK_SIZE/CHUNK_OVERLAP). "
                                    "Пересоберите: python -m curator.ingest", manifest=manifest)
    built: dict[str, str] = dict(manifest.get("files") or {})
    current = file_hashes(kb_dir) if kb_dir.is_dir() else {}
    added = sorted(set(current) - set(built))
    removed = sorted(set(built) - set(current))
    changed = sorted(k for k in set(current) & set(built) if current[k] != built[k])
    if added or removed or changed:
        parts = []
        for label, items in (("изменены", changed), ("добавлены", added), ("удалены", removed)):
            if items:
                shown = ", ".join(items[:4]) + (f" и ещё {len(items) - 4}" if len(items) > 4 else "")
                parts.append(f"{label}: {shown}")
        return IndexStatus("stale", "Индекс устарел — после сборки в базе знаний " + "; ".join(parts)
                           + ". Пересоберите: python -m curator.ingest", added, changed, removed, manifest)
    return IndexStatus("ok", f"Индекс актуален: {manifest.get('chunks')} фрагментов, собран {manifest.get('built_at')}",
                       manifest=manifest)


def _write_manifest(index_dir: Path, manifest: dict[str, Any]) -> None:
    tmp = index_dir / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, index_dir / MANIFEST_NAME)


def build_index(kb_dir: Path, index_dir: Path, embeddings: Embeddings, *, embed_model: str, chunk_size: int,
                chunk_overlap: int, prefix: str = "kb", now: datetime | None = None) -> dict[str, Any]:
    """Полная пересборка. Возвращает манифест новой коллекции."""
    kb = load_kb(kb_dir, strict=True)  # ошибка формата — до того, как тратить деньги на эмбеддинги
    chunks = chunk_documents(kb.documents, chunk_size, chunk_overlap)
    hashes = file_hashes(kb_dir)
    moment = (now or datetime.now(MSK)).astimezone(MSK)
    # Метка времени — для людей, случайный хвост — чтобы две сборки в одну секунду не попали в одну коллекцию
    # (тогда неудачная сборка удалила бы рабочую).
    name = f"{prefix}-{moment:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
    if name in {c.name for c in chroma_client(index_dir).list_collections()}:
        raise RuntimeError(f"коллекция {name} уже есть — повторите сборку")
    client = chroma_client(index_dir)
    store = Chroma(client=client, collection_name=name, embedding_function=embeddings,
                   collection_configuration=COLLECTION_CONFIGURATION)  # pyright: ignore[reportArgumentType]
    try:
        for start in range(0, len(chunks), BATCH):
            part = chunks[start:start + BATCH]
            store.add_documents(part, ids=[str(d.id) for d in part])
    except BaseException:
        client.delete_collection(name)
        raise
    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "built_at": moment.isoformat(timespec="seconds"),
        "collection": name,
        "embed_model": embed_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "documents": len(kb.documents),
        "chunks": len(chunks),
        "by_type": dict(Counter(str(c.metadata["resource_type"]) for c in chunks)),
        "kb_version": kb_version(hashes),
        "files": hashes,
    }
    _write_manifest(index_dir, manifest)
    for collection in client.list_collections():
        if collection.name != name and collection.name.startswith(prefix + "-"):
            client.delete_collection(collection.name)
    remove_orphan_segments(index_dir)
    return manifest


def remove_orphan_segments(index_dir: Path) -> int:
    """Папки HNSW удалённых коллекций: Chroma удаляет коллекцию из своей базы, а файлы оставляет на диске,
    и с каждой пересборкой индекс рос бы на размер коллекции. Удаляются только папки с именем-UUID, которых нет
    в таблице segments. Если схема Chroma поменяется и запрос не пройдёт — ничего не трогаем."""
    try:
        con = sqlite3.connect(f"file:{(index_dir / 'chroma.sqlite3').as_posix()}?mode=ro", uri=True)
        try:
            alive = {str(row[0]) for row in con.execute("SELECT id FROM segments")}
        finally:
            con.close()
    except sqlite3.Error:
        return 0
    removed = 0
    for path in index_dir.iterdir():
        if path.is_dir() and _UUID.fullmatch(path.name) and path.name not in alive:
            shutil.rmtree(path, ignore_errors=True)  # на Windows файл может держать этот же процесс — не страшно
            removed += not path.exists()
    return removed


def main(argv: list[str] | None = None) -> int:
    from .config import BASE_DIR, load_settings
    from .factory import make_embeddings

    parser = argparse.ArgumentParser(prog="python -m curator.ingest", description="Индексация базы знаний в Chroma")
    parser.add_argument("--check", action="store_true", help="только сверить индекс с базой знаний")
    args = parser.parse_args(argv)
    settings = load_settings()
    try:
        ensure_index_dir_usable(settings.index_dir)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not settings.index_dir.resolve().is_relative_to(BASE_DIR):
        print(f"Индекс лежит вне репозитория: {settings.index_dir} "
              "(путь проекта не ASCII, а Chroma на Windows в такой путь не пишет; задаётся CURATOR_INDEX_DIR)")
    status = index_status(settings.kb_dir, settings.index_dir, settings.embed_model, settings.chunk_size,
                          settings.chunk_overlap)
    if args.check:
        print(status.message)
        return 0 if status.ok else 1
    if not settings.openai_api_key:
        print("Нет OPENAI_API_KEY (в .env или окружении): без него эмбеддинги не посчитать", file=sys.stderr)
        return 2
    started = time.perf_counter()
    manifest = build_index(settings.kb_dir, settings.index_dir, make_embeddings(settings),
                           embed_model=settings.embed_model, chunk_size=settings.chunk_size,
                           chunk_overlap=settings.chunk_overlap, prefix=settings.collection_prefix)
    by_type = ", ".join(f"{k}: {v}" for k, v in sorted(manifest["by_type"].items()))
    print(f"Готово за {time.perf_counter() - started:.1f} с: {manifest['documents']} документов, "
          f"{manifest['chunks']} фрагментов ({by_type}); коллекция {manifest['collection']} в {settings.index_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
