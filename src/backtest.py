"""Deterministic baseline execution engine for Experiment #001."""
from dataclasses import dataclass
from datetime import time
from typing import Callable

import pandas as pd

from .config import Settings, load_settings
from .strategy import Signal, generate_signal

ASIA_KOLKATA = "Asia/Kolkata"
SESSION_CLOSE = time(15, 30)


@dataclass(frozen=True)
class PositionSize:
    quantity: int
    risk_per_share: float
    planned_risk: float


@dataclass
class Trade:
    symbol: str
    signal_time: object
    entry_time: object
    entry: float
    stop: float
    target: float
    quantity: int
    exit_time: object
    exit: float
    pnl: float
    reason: str


@dataclass(frozen=True)
class RejectedSignal:
    symbol: str
    timestamp: object
    reason: str


@dataclass(frozen=True)
class BacktestResult:
    trades: list[Trade]
    rejected_signals: list[RejectedSignal]


@dataclass
class _OpenPosition:
    signal: Signal
    entry_time: object
    entry: float
    stop: float
    target: float
    quantity: int


SignalGenerator = Callable[[pd.DataFrame, str, Settings], Signal | None]


def calculate_position_size(entry: float, stop: float, settings: Settings) -> PositionSize:
    """Size integer shares under configured risk and position-value limits."""
    risk_per_share = entry - stop
    if risk_per_share <= 0 or entry <= 0:
        return PositionSize(0, risk_per_share, 0.0)
    max_risk = settings.capital * settings.max_risk_per_trade_pct / 100
    quantity = max(0, min(int(max_risk // risk_per_share), int(settings.max_position_value // entry)))
    return PositionSize(quantity, risk_per_share, quantity * risk_per_share)


def _local_timestamp(timestamp: object) -> pd.Timestamp:
    value = pd.Timestamp(timestamp)
    return value.tz_localize(ASIA_KOLKATA) if value.tzinfo is None else value.tz_convert(ASIA_KOLKATA)


def _entry_allowed(timestamp: object, settings: Settings) -> bool:
    current = _local_timestamp(timestamp).time()
    return time.fromisoformat(settings.entry_start) <= current <= time.fromisoformat(settings.entry_end)


def _close(position: _OpenPosition, timestamp: object, price: float, reason: str) -> Trade:
    return Trade(position.signal.symbol, position.signal.timestamp, position.entry_time,
                 position.entry, position.stop, position.target, position.quantity,
                 timestamp, price, (price - position.entry) * position.quantity, reason)


def run_backtest(
    df: pd.DataFrame, symbol: str = "TEST", settings: Settings | None = None,
    signal_generator: SignalGenerator = generate_signal,
) -> BacktestResult:
    """Enter at the next bar open; ambiguous stop/target bars conservatively stop out."""
    settings = settings or load_settings()
    trades: list[Trade] = []
    rejected: list[RejectedSignal] = []
    position: _OpenPosition | None = None
    pending: Signal | None = None
    daily_pnl: dict[object, float] = {}
    daily_trades: dict[object, int] = {}

    for i in range(len(df)):
        bar, timestamp = df.iloc[i], _local_timestamp(df.index[i])
        day = timestamp.date()

        if position and _local_timestamp(position.entry_time).date() != day:
            previous_time = _local_timestamp(df.index[i - 1])
            trade = _close(position, previous_time, float(df.iloc[i - 1].close), "SESSION_CLOSE")
            trades.append(trade)
            daily_pnl[previous_time.date()] = daily_pnl.get(previous_time.date(), 0.0) + trade.pnl
            position = None

        if pending:
            signal, pending = pending, None
            entry, stop = float(bar.open), float(signal.stop)
            size = calculate_position_size(entry, stop, settings)
            max_daily_loss = settings.capital * settings.max_daily_loss_pct / 100
            if timestamp.date() != _local_timestamp(signal.timestamp).date():
                reason = "NO_OVERNIGHT_ENTRY"
            elif not _entry_allowed(timestamp, settings):
                reason = "OUTSIDE_ENTRY_WINDOW"
            elif position:
                reason = "POSITION_OPEN"
            elif size.quantity == 0:
                reason = "INVALID_POSITION_SIZE"
            elif daily_trades.get(day, 0) >= settings.max_trades_per_day:
                reason = "MAX_TRADES_PER_DAY"
            elif daily_pnl.get(day, 0.0) - size.planned_risk < -max_daily_loss:
                reason = "DAILY_LOSS_LIMIT"
            else:
                target = entry + settings.risk_reward_min * size.risk_per_share
                position = _OpenPosition(signal, timestamp, entry, stop, target, size.quantity)
                daily_trades[day] = daily_trades.get(day, 0) + 1
                reason = None
            if reason:
                rejected.append(RejectedSignal(symbol, signal.timestamp, reason))

        if position:
            stop_hit, target_hit = float(bar.low) <= position.stop, float(bar.high) >= position.target
            if stop_hit and target_hit:
                trade = _close(position, timestamp, position.stop, "STOP_TARGET_AMBIGUITY_STOP")
            elif stop_hit:
                trade = _close(position, timestamp, position.stop, "STOP")
            elif target_hit:
                trade = _close(position, timestamp, position.target, "TARGET")
            elif timestamp.time() >= SESSION_CLOSE:
                trade = _close(position, timestamp, float(bar.close), "SESSION_CLOSE")
            else:
                trade = None
            if trade:
                trades.append(trade)
                daily_pnl[day] = daily_pnl.get(day, 0.0) + trade.pnl
                position = None

        signal = signal_generator(df.iloc[: i + 1], symbol, settings)
        if not signal:
            continue
        if i == len(df) - 1:
            rejected.append(RejectedSignal(symbol, signal.timestamp, "NO_NEXT_BAR"))
        elif not _entry_allowed(signal.timestamp, settings):
            rejected.append(RejectedSignal(symbol, signal.timestamp, "OUTSIDE_ENTRY_WINDOW"))
        elif position or pending:
            rejected.append(RejectedSignal(symbol, signal.timestamp, "POSITION_OPEN"))
        else:
            pending = signal

    if position:
        final_time = _local_timestamp(df.index[-1])
        trades.append(_close(position, final_time, float(df.iloc[-1].close), "SESSION_CLOSE"))
    return BacktestResult(trades, rejected)


def run_simple_backtest(df: pd.DataFrame, symbol="TEST") -> list[Trade]:
    """Compatibility wrapper for the existing CLI and dashboard."""
    return run_backtest(df, symbol).trades
