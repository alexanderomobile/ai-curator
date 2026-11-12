#!/usr/bin/env bash
# Проверка одной командой: типы (basedpyright), затем тесты (pytest). Сеть и ключ OpenAI не нужны:
# модель, анализ запроса и эмбеддинги в тестах — фейки, данные — tests/fixtures.
#   bash check.sh             # поставить зависимости из requirements*.txt и проверить
#   bash check.sh --no-sync   # окружение уже готово — только проверить
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
if [ ! -x "$PY" ]; then
  uv venv --quiet --python 3.12 .venv
  if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
fi
if [ "${1:-}" != "--no-sync" ]; then
  uv pip install --quiet --python "$PY" -r requirements.txt -r requirements-dev.txt
fi

echo "== basedpyright: curator app eval tests"
"$PY" -m basedpyright curator app eval tests
echo "== pytest"
PYTHONIOENCODING=utf-8 "$PY" -m pytest -p no:cacheprovider
