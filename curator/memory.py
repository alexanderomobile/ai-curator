"""Память диалога: последние реплики каждой сессии в памяти процесса.

Нужна, чтобы куратор понимал уточнения («а по второму модулю?»): последние реплики передаются в анализ запроса
и в промпт ответа. Вместе с репликой хранится её разбор (сценарий и модуль) — правила-запаска наследуют его,
если уточнение само ни на что не указывает. Молчащая дольше ttl сессия забывается, при переполнении вытесняется
самая старая. Долговременно переписка не хранится: журнал пишет только обезличенный вопрос, без ответа.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    question: str
    answer: str  # тело ответа без блока «Источники»
    intent: str = ""
    module: int | None = None


@dataclass
class _Dialog:
    turns: deque[Turn]
    touched: float = field(default_factory=time.monotonic)


class DialogMemory:
    def __init__(self, turns: int = 2, ttl_seconds: float = 3600, max_sessions: int = 2000,
                 clock: Callable[[], float] = time.monotonic):
        self.turns = turns
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self.clock = clock
        self._dialogs: OrderedDict[str, _Dialog] = OrderedDict()
        self._lock = threading.Lock()

    def _drop_expired(self, now: float) -> None:
        for key in [k for k, d in self._dialogs.items() if now - d.touched > self.ttl]:
            del self._dialogs[key]

    def history(self, session_id: str) -> list[Turn]:
        """Реплики от старых к новым."""
        with self._lock:
            self._drop_expired(self.clock())
            dialog = self._dialogs.get(session_id)
            return list(dialog.turns) if dialog else []

    def add(self, session_id: str, turn: Turn) -> None:
        with self._lock:
            now = self.clock()
            self._drop_expired(now)
            dialog = self._dialogs.get(session_id)
            if dialog is None:
                dialog = _Dialog(deque(maxlen=max(self.turns, 1)))
                self._dialogs[session_id] = dialog
            dialog.turns.append(turn)
            dialog.touched = now
            self._dialogs.move_to_end(session_id)
            while len(self._dialogs) > self.max_sessions:
                self._dialogs.popitem(last=False)

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._dialogs.pop(session_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._dialogs)
