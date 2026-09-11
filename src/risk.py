"""Deterministic risk controls for Experiment #001."""

from dataclasses import dataclass
from datetime import time
from typing import Literal

import pandas as pd

from .config import Settings


ASIA_KOLKATA = "Asia/Kolkata"


@dataclass(frozen=True)
class PositionSize:
    """Calculated position size before a trade is opened."""

    quantity: int
    risk_per_share: float
    planned_risk: float
    position_value: float


@dataclass(frozen=True)
class RiskDecision:
    """Result of a deterministic risk check."""

    approved: bool
    reason: str
    position_size: PositionSize


@dataclass
class RiskState:
    """Mutable state maintained during one backtest/trading session."""

    realized_pnl: float = 0.0
    trades_today: int = 0
    position_open: bool = False


def calculate_position_size(
    entry: float,
    stop: float,
    settings: Settings,
) -> PositionSize:
    """Calculate integer quantity subject to risk and position-value limits."""

    risk_per_share = float(entry) - float(stop)

    if entry <= 0 or risk_per_share <= 0:
        return PositionSize(
            quantity=0,
            risk_per_share=risk_per_share,
            planned_risk=0.0,
            position_value=0.0,
        )

    max_risk = (
        settings.capital
        * settings.max_risk_per_trade_pct
        / 100.0
    )

    risk_quantity = int(max_risk // risk_per_share)
    value_quantity = int(settings.max_position_value // entry)

    quantity = max(0, min(risk_quantity, value_quantity))

    return PositionSize(
        quantity=quantity,
        risk_per_share=risk_per_share,
        planned_risk=quantity * risk_per_share,
        position_value=quantity * entry,
    )


def _local_timestamp(timestamp: object) -> pd.Timestamp:
    """Return timestamp converted to Asia/Kolkata."""

    value = pd.Timestamp(timestamp)

    if value.tzinfo is None:
        return value.tz_localize(ASIA_KOLKATA)

    return value.tz_convert(ASIA_KOLKATA)


def is_entry_time_allowed(
    timestamp: object,
    settings: Settings,
) -> bool:
    """Return whether a signal timestamp falls inside the configured entry window."""

    current_time = _local_timestamp(timestamp).time()

    start = time.fromisoformat(settings.entry_start)
    end = time.fromisoformat(settings.entry_end)

    return start <= current_time <= end


def check_entry(
    timestamp: object,
    entry: float,
    stop: float,
    settings: Settings,
    state: RiskState,
) -> RiskDecision:
    """Apply all deterministic entry risk controls."""

    position_size = calculate_position_size(entry, stop, settings)

    if not is_entry_time_allowed(timestamp, settings):
        return RiskDecision(
            approved=False,
            reason="OUTSIDE_ENTRY_WINDOW",
            position_size=position_size,
        )

    if state.position_open:
        return RiskDecision(
            approved=False,
            reason="POSITION_OPEN",
            position_size=position_size,
        )

    if state.trades_today >= settings.max_trades_per_day:
        return RiskDecision(
            approved=False,
            reason="MAX_TRADES_PER_DAY",
            position_size=position_size,
        )

    if position_size.quantity <= 0:
        return RiskDecision(
            approved=False,
            reason="INVALID_POSITION_SIZE",
            position_size=position_size,
        )

    max_daily_loss = (
        settings.capital
        * settings.max_daily_loss_pct
        / 100.0
    )

    # Conservative entry check:
    # assume the entire planned risk is lost when determining
    # whether this trade could breach the daily loss limit.
    projected_pnl = state.realized_pnl - position_size.planned_risk

    if projected_pnl < -max_daily_loss:
        return RiskDecision(
            approved=False,
            reason="DAILY_LOSS_LIMIT",
            position_size=position_size,
        )

    return RiskDecision(
        approved=True,
        reason="APPROVED",
        position_size=position_size,
    )


def record_entry(state: RiskState) -> None:
    """Record an approved trade entry."""

    state.position_open = True
    state.trades_today += 1


def record_exit(state: RiskState, pnl: float) -> None:
    """Record a completed trade and realized P&L."""

    state.position_open = False
    state.realized_pnl += float(pnl)


def reset_day(state: RiskState) -> None:
    """Reset daily risk counters while preserving account P&L."""

    state.trades_today = 0
    state.position_open = False