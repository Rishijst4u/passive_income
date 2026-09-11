"""Small, replaceable calendar support for NSE equity trading days."""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Protocol


class TradingDayType(str, Enum):
    WEEKEND = "weekend"
    MARKET_HOLIDAY = "market_holiday"
    TRADING_DAY = "trading_day"


class TradingCalendar(Protocol):
    def day_type(self, trading_date: date) -> TradingDayType: ...

    def is_trading_day(self, trading_date: date) -> bool: ...


# NSE's published 2026 Capital Market (Equities) trading holidays.
NSE_EQUITY_HOLIDAYS_2026 = frozenset(
    {
        date(2026, 1, 15), date(2026, 1, 26), date(2026, 2, 19),
        date(2026, 3, 3), date(2026, 3, 19), date(2026, 3, 26),
        date(2026, 3, 31), date(2026, 4, 1), date(2026, 4, 3),
        date(2026, 4, 14), date(2026, 5, 1), date(2026, 5, 28),
        date(2026, 6, 26), date(2026, 8, 26), date(2026, 9, 14),
        date(2026, 10, 2), date(2026, 10, 20), date(2026, 11, 10),
        date(2026, 11, 24), date(2026, 12, 25),
    }
)


@dataclass(frozen=True)
class NseTradingCalendar:
    holidays: frozenset[date] = NSE_EQUITY_HOLIDAYS_2026

    def day_type(self, trading_date: date) -> TradingDayType:
        if trading_date.weekday() >= 5:
            return TradingDayType.WEEKEND
        if trading_date in self.holidays:
            return TradingDayType.MARKET_HOLIDAY
        return TradingDayType.TRADING_DAY

    def is_trading_day(self, trading_date: date) -> bool:
        return self.day_type(trading_date) is TradingDayType.TRADING_DAY


DEFAULT_NSE_TRADING_CALENDAR = NseTradingCalendar()
