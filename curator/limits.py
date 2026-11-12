"""Лимиты публичного демо: длина вопроса, вопросов в час на посетителя, общий потолок в час и общий дневной потолок.

Защищают ключ OpenAI от расхода. Счётчики в памяти процесса (Streamlit — один процесс); после перезапуска дневной
счётчик и окно последнего часа берутся из журнала, чтобы перезапуск не обнулял потолки. День — по московскому
времени. Ключ почасового лимита — посетитель (адрес клиента из заголовков туннеля), а не диалог: «Новый диалог»
и перезагрузка страницы его не сбрасывают. Общий потолок в час не даёт выбрать дневной лимит за минуты.
Прогоны оценки (source=eval) лимитам не подчиняются.
"""
from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from .config import MSK

HOUR = timedelta(hours=1)
MAX_TRACKED_CLIENTS = 5000  # забытые посетители не копятся бесконечно


@dataclass(frozen=True)
class LimitDecision:
    ok: bool
    reason: str = ""  # too_long | per_hour | hourly | daily | empty
    values: tuple[tuple[str, object], ...] = ()  # подстановки в шаблон refusal_limits.md

    @property
    def refusal(self) -> str:
        return f"limit_{self.reason}" if self.reason else ""


def _wait_minutes(oldest: datetime, now: datetime) -> int:
    return max(1, math.ceil((HOUR - (now - oldest)).total_seconds() / 60))


class DemoLimits:
    def __init__(self, max_chars: int, per_hour: int, daily_cap: int,
                 clock: Callable[[], datetime] = lambda: datetime.now(MSK), daily_used: int = 0,
                 hourly_cap: int | None = None, recent: Iterable[datetime] = ()):
        """hourly_cap=None — без общего потолка в час. recent — когда были вопросы за последний час (из журнала)."""
        self.max_chars = max_chars
        self.per_hour = per_hour
        self.daily_cap = daily_cap
        self.hourly_cap = hourly_cap
        self.clock = clock
        self._clients: dict[str, deque[datetime]] = {}
        self._all: deque[datetime] = deque(sorted(recent))
        self._day: date = clock().astimezone(MSK).date()
        self._daily = daily_used
        self._lock = threading.Lock()

    def check(self, client: str, question: str) -> LimitDecision:
        """Проверить и, если можно, сразу засчитать вопрос (одним шагом под замком — без гонок)."""
        text = question.strip()
        if not text:
            return LimitDecision(False, "empty")
        if len(text) > self.max_chars:
            return LimitDecision(False, "too_long", (("limit", self.max_chars), ("length", len(text))))
        with self._lock:
            now = self.clock()
            today = now.astimezone(MSK).date()
            if today != self._day:
                self._day, self._daily = today, 0
            if self._daily >= self.daily_cap:
                return LimitDecision(False, "daily", (("limit", self.daily_cap),))
            while self._all and now - self._all[0] >= HOUR:
                self._all.popleft()
            stamps = self._clients.setdefault(client, deque())
            while stamps and now - stamps[0] >= HOUR:
                stamps.popleft()
            if len(stamps) >= self.per_hour:
                return LimitDecision(False, "per_hour", (("limit", self.per_hour),
                                                         ("minutes", _wait_minutes(stamps[0], now))))
            if self.hourly_cap is not None and len(self._all) >= self.hourly_cap:
                return LimitDecision(False, "hourly", (("limit", self.hourly_cap),
                                                       ("minutes", _wait_minutes(self._all[0], now))))
            stamps.append(now)
            self._all.append(now)
            self._daily += 1
            if len(self._clients) > MAX_TRACKED_CLIENTS:
                for key in [k for k, v in self._clients.items() if not v or now - v[-1] >= HOUR]:
                    del self._clients[key]
            return LimitDecision(True)

    @property
    def daily_used(self) -> int:
        return self._daily
