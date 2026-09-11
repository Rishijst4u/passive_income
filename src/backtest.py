"""Deterministic baseline execution engine for Experiment #001."""

from dataclasses import dataclass
from datetime import time
from typing import Callable

import pandas as pd

from .config import Settings, load_settings
from .risk import (
    RiskState,
    check_entry,
    record_entry,
    record_exit,
    reset_day,
)
from .strategy import Signal, generate_signal


ASIA_KOLKATA = "Asia/Kolkata"
SESSION_CLOSE = time(15, 30)


@dataclass
class Trade:
    """Completed paper trade."""

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
    """Signal rejected by deterministic execution/risk controls."""

    symbol: str
    timestamp: object
    reason: str


@dataclass(frozen=True)
class BacktestResult:
    """Complete backtest output."""

    trades: list[Trade]
    rejected_signals: list[RejectedSignal]


@dataclass
class _OpenPosition:
    """Internal representation of an open position."""

    signal: Signal
    entry_time: object
    entry: float
    stop: float
    target: float
    quantity: int


SignalGenerator = Callable[[pd.DataFrame, str, Settings], Signal | None]


def _local_timestamp(timestamp):
    """Return timestamp normalized to Asia/Kolkata."""
    value = pd.Timestamp(timestamp)

    if value.tzinfo is None:
        return value.tz_localize(ASIA_KOLKATA)

    return value.tz_convert(ASIA_KOLKATA)


def _close(
    position: _OpenPosition,
    timestamp,
    price: float,
    reason: str,
) -> Trade:
    """Close a position and calculate gross P&L."""
    exit_price = float(price)

    pnl = (
        exit_price - position.entry
    ) * position.quantity

    return Trade(
        symbol=position.signal.symbol,
        signal_time=position.signal.timestamp,
        entry_time=position.entry_time,
        entry=position.entry,
        stop=position.stop,
        target=position.target,
        quantity=position.quantity,
        exit_time=timestamp,
        exit=exit_price,
        pnl=pnl,
        reason=reason,
    )


def run_backtest(
    df: pd.DataFrame,
    symbol: str = "TEST",
    settings: Settings | None = None,
    signal_generator: SignalGenerator = generate_signal,
) -> BacktestResult:
    """
    Run the deterministic baseline backtest.

    Execution assumptions:
    - Strategy signals are generated from the current bar.
    - A signal enters at the NEXT bar open.
    - Risk controls are evaluated using the actual entry timestamp.
    - Only one position can be open at a time.
    - No overnight positions are allowed.
    - Stop/target ambiguity is resolved conservatively in favour of the stop.
    - Position sizing is delegated to the deterministic risk engine.
    """
    settings = settings or load_settings()

    if df.empty:
        return BacktestResult(
            trades=[],
            rejected_signals=[],
        )

    data = df.copy()

    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"])

    data = data.sort_values("timestamp").reset_index(drop=True)

    trades: list[Trade] = []
    rejected: list[RejectedSignal] = []

    risk_state = RiskState()
    position: _OpenPosition | None = None
    pending_signal: Signal | None = None

    current_date = None

    for i in range(len(data)):
        row = data.iloc[i]

        timestamp = _local_timestamp(row["timestamp"])
        trading_date = timestamp.date()

        # ---------------------------------------------------------
        # New trading day
        # ---------------------------------------------------------
        if current_date is None:
            current_date = trading_date

        elif trading_date != current_date:
            # No overnight positions are permitted.
            if position is not None:
                previous_timestamp = _local_timestamp(
                    data.iloc[i - 1]["timestamp"]
                )

                previous_close = float(data.iloc[i - 1]["close"])

                trade = _close(
                    position=position,
                    timestamp=previous_timestamp,
                    price=previous_close,
                    reason="SESSION_CLOSE",
                )

                trades.append(trade)
                record_exit(risk_state, trade.pnl)

                position = None

            pending_signal = None
            reset_day(risk_state)
            current_date = trading_date

        # ---------------------------------------------------------
        # Enter pending signal at current bar OPEN
        # ---------------------------------------------------------
        if pending_signal is not None and position is None:
            entry_time = timestamp
            entry_price = float(row["open"])

            signal = pending_signal

            # Signal's stop is calculated by the strategy.
            stop_price = float(signal.stop)

            risk_decision = check_entry(
                timestamp=entry_time,
                entry=entry_price,
                stop=stop_price,
                settings=settings,
                state=risk_state,
            )

            if not risk_decision.approved:
                rejected.append(
                    RejectedSignal(
                        symbol=symbol,
                        timestamp=signal.timestamp,
                        reason=risk_decision.reason,
                    )
                )

                pending_signal = None

            else:
                position_size = risk_decision.position_size

                risk_per_share = position_size.risk_per_share

                target_price = (
                    entry_price
                    + (
                        risk_per_share
                        * settings.risk_reward_min
                    )
                )

                position = _OpenPosition(
                    signal=signal,
                    entry_time=entry_time,
                    entry=entry_price,
                    stop=stop_price,
                    target=target_price,
                    quantity=position_size.quantity,
                )

                record_entry(risk_state)

                pending_signal = None

        # ---------------------------------------------------------
        # Manage open position
        # ---------------------------------------------------------
        if position is not None:
            high = float(row["high"])
            low = float(row["low"])

            stop_hit = low <= position.stop
            target_hit = high >= position.target

            # Conservative handling:
            # If both levels were touched in the same candle,
            # assume the stop was hit first.
            if stop_hit and target_hit:
                trade = _close(
                    position=position,
                    timestamp=timestamp,
                    price=position.stop,
                    reason="STOP_TARGET_AMBIGUITY_STOP",
                )

                trades.append(trade)
                record_exit(risk_state, trade.pnl)
                position = None

            elif stop_hit:
                trade = _close(
                    position=position,
                    timestamp=timestamp,
                    price=position.stop,
                    reason="STOP",
                )

                trades.append(trade)
                record_exit(risk_state, trade.pnl)
                position = None

            elif target_hit:
                trade = _close(
                    position=position,
                    timestamp=timestamp,
                    price=position.target,
                    reason="TARGET",
                )

                trades.append(trade)
                record_exit(risk_state, trade.pnl)
                position = None

            # Session-close protection.
            elif timestamp.time() >= SESSION_CLOSE:
                close_price = float(row["close"])

                trade = _close(
                    position=position,
                    timestamp=timestamp,
                    price=close_price,
                    reason="SESSION_CLOSE",
                )

                trades.append(trade)
                record_exit(risk_state, trade.pnl)
                position = None

        # ---------------------------------------------------------
        # Generate signal for NEXT bar
        # ---------------------------------------------------------
        if position is None and pending_signal is None:
            signal = signal_generator(
                data.iloc[: i + 1],
                symbol,
                settings,
            )

            if signal is not None:
                # A signal on the final bar cannot be executed because
                # there is no next bar.
                if i < len(data) - 1:
                    pending_signal = signal

    # -------------------------------------------------------------
    # Force-close any remaining position at final available close.
    # -------------------------------------------------------------
    if position is not None:
        final_row = data.iloc[-1]

        final_timestamp = _local_timestamp(
            final_row["timestamp"]
        )

        final_close = float(final_row["close"])

        trade = _close(
            position=position,
            timestamp=final_timestamp,
            price=final_close,
            reason="SESSION_CLOSE",
        )

        trades.append(trade)
        record_exit(risk_state, trade.pnl)

        position = None

    return BacktestResult(
        trades=trades,
        rejected_signals=rejected,
    )


def run_simple_backtest(
    df: pd.DataFrame,
    symbol: str = "TEST",
    settings: Settings | None = None,
) -> list[Trade]:
    """
    Backward-compatible helper returning only completed trades.
    """
    result = run_backtest(
        df=df,
        symbol=symbol,
        settings=settings,
    )

    return result.trades