"""Лимиты демо: длина, вопросов в час на сессию, дневной потолок с полуночью по Москве."""
from __future__ import annotations

from datetime import datetime, timedelta

from curator.config import MSK
from curator.limits import DemoLimits


class Clock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now


START = datetime(2026, 11, 12, 22, 0, tzinfo=MSK)


def test_length_and_empty():
    limits = DemoLimits(max_chars=10, per_hour=5, daily_cap=10, clock=Clock(START))
    assert limits.check("s", "   ").reason == "empty"
    too_long = limits.check("s", "x" * 11)
    assert too_long.reason == "too_long" and dict(too_long.values) == {"limit": 10, "length": 11}
    assert too_long.refusal == "limit_too_long"
    assert limits.check("s", "x" * 10).ok
    assert limits.daily_used == 1  # отказ по длине не тратит дневной лимит


def test_per_hour_sliding_window_is_per_session():
    clock = Clock(START)
    limits = DemoLimits(max_chars=100, per_hour=2, daily_cap=100, clock=clock)
    assert limits.check("a", "q").ok and limits.check("a", "q").ok
    blocked = limits.check("a", "q")
    assert blocked.reason == "per_hour" and dict(blocked.values)["minutes"] == 60
    assert limits.check("b", "q").ok  # у другой сессии свой счётчик
    clock.now += timedelta(minutes=59)
    assert dict(limits.check("a", "q").values)["minutes"] == 1
    clock.now += timedelta(minutes=1)
    assert limits.check("a", "q").ok


def test_daily_cap_resets_at_moscow_midnight():
    clock = Clock(START)
    limits = DemoLimits(max_chars=100, per_hour=100, daily_cap=2, clock=clock, daily_used=1)
    assert limits.check("a", "q").ok
    assert limits.check("b", "q").reason == "daily"
    clock.now = datetime(2026, 11, 13, 0, 1, tzinfo=MSK)
    assert limits.check("b", "q").ok and limits.daily_used == 1


def test_global_hourly_cap_across_clients():
    clock = Clock(START)
    limits = DemoLimits(max_chars=100, per_hour=5, daily_cap=100, clock=clock, hourly_cap=3)
    assert all(limits.check(client, "q").ok for client in ("a", "b", "c"))
    blocked = limits.check("d", "q")  # у посетителя d свой лимит не тронут, но общий потолок часа выбран
    assert blocked.reason == "hourly" and dict(blocked.values) == {"limit": 3, "minutes": 60}
    assert limits.daily_used == 3  # отказ по потолку дневной лимит не тратит
    clock.now += timedelta(minutes=60)
    assert limits.check("d", "q").ok


def test_restart_restores_last_hour_and_day():
    clock = Clock(START)
    recent = [START - timedelta(minutes=50), START - timedelta(minutes=10)]
    limits = DemoLimits(max_chars=100, per_hour=5, daily_cap=10, clock=clock, hourly_cap=2, daily_used=9,
                        recent=recent)
    blocked = limits.check("a", "q")
    assert blocked.reason == "hourly" and dict(blocked.values)["minutes"] == 10  # первый вопрос часа — 50 мин назад
    clock.now += timedelta(minutes=10)
    assert limits.check("a", "q").ok and limits.daily_used == 10
    assert limits.check("b", "q").reason == "daily"


def test_no_hourly_cap_by_default():
    limits = DemoLimits(max_chars=100, per_hour=100, daily_cap=1000, clock=Clock(START))
    assert all(limits.check(str(n), "q").ok for n in range(200))
