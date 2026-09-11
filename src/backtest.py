"""Deterministic baseline execution engine for Experiment #001."""

from dataclasses import dataclass
from datetime import time
from typing import Callable

import pandas as pd

from .config import Settings, load_settings
from .costs import CostSettings, calculate_trade_costs
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
    symbol: str
    signal_time: object
    entry_time: object
    entry: float
    stop: float
    target: float
    quantity: int
    exit_time: object
    exit: float

    gross_pnl: float
    total_cost: float
    net_pnl: float

    brokerage: float
    stt: float
    exchange_transaction_charges: float
    sebi_charges: float
    stamp_duty: float
    gst: float
    ipft: float
    slippage: float

    reason: str

    @property
    def pnl(self):
        """Backward-compatible alias for net P&L."""
        return self.net_pnl


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


SignalGenerator = Callable[
    [pd.DataFrame, str, Settings],
    Signal | None,
]


def _local_timestamp(timestamp):
    """Return timestamp normalized to Asia/Kolkata."""
    value = pd.Timestamp(timestamp)

    if value.tzinfo is None:
        return value.tz_localize(ASIA_KOLKATA)

    return value.tz_convert(ASIA_KOLKATA)


def _prepare_backtest_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize backtest input so timestamps are always in a column.

    Supported inputs:
    1. DataFrame containing a 'timestamp' column.
    2. DataFrame with a DatetimeIndex.

    The returned DataFrame always contains a 'timestamp' column.
    """
    if df.empty:
        return df.copy()

    data = df.copy()

    if "timestamp" not in data.columns:
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()

            # reset_index() normally creates a column called 'index'
            # unless the DatetimeIndex already has a name.
            if "timestamp" not in data.columns:
                if "index" in data.columns:
                    data = data.rename(columns={"index": "timestamp"})
                else:
                    raise ValueError(
                        "Unable to identify timestamp after resetting DatetimeIndex."
                    )
        else:
            raise ValueError(
                "Backtest data must contain a 'timestamp' column "
                "or use a DatetimeIndex."
            )

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    return data.sort_values("timestamp").reset_index(drop=True)


def _close_position(
    position: _OpenPosition,
    exit_time,
    exit_price: float,
    reason: str,
    cost_settings: CostSettings,
) -> Trade:
    """Close an open position and calculate trading costs."""
    costs = calculate_trade_costs(
        entry_price=position.entry,
        exit_price=exit_price,
        quantity=position.quantity,
        settings=cost_settings,
    )

    return Trade(
        symbol=position.signal.symbol,
        signal_time=position.signal.timestamp,
        entry_time=position.entry_time,
        entry=position.entry,
        stop=position.stop,
        target=position.target,
        quantity=position.quantity,
        exit_time=exit_time,
        exit=exit_price,
        gross_pnl=costs.gross_pnl,
        total_cost=costs.total_cost,
        net_pnl=costs.net_pnl,
        brokerage=costs.brokerage,
        stt=costs.stt,
        exchange_transaction_charges=costs.exchange_transaction_charges,
        sebi_charges=costs.sebi_charges,
        stamp_duty=costs.stamp_duty,
        gst=costs.gst,
        ipft=costs.ipft,
        slippage=costs.slippage,
        reason=reason,
    )


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    settings: Settings | None = None,
    signal_generator: SignalGenerator = generate_signal,
    cost_settings: CostSettings | None = None,
) -> BacktestResult:
    """
    Run the deterministic Experiment #001 backtest.

    Execution assumptions:
    - Signal is generated from the current bar.
    - Entry occurs at the next bar open.
    - Risk controls are checked at actual entry.
    - One position at a time.
    - No overnight positions.
    - If stop and target are both touched in one bar, stop wins.
    - Positions are closed at session close.
    - Trading costs are applied to each completed trade.
    """
    settings = settings or load_settings()
    cost_settings = cost_settings or CostSettings()

    if df.empty:
        return BacktestResult(trades=[], rejected_signals=[])

    data = _prepare_backtest_data(df)

    risk_state = RiskState()
    position: _OpenPosition | None = None
    pending_signal: Signal | None = None

    current_date = None
    trades: list[Trade] = []
    rejected_signals: list[RejectedSignal] = []

    for i in range(len(data)):
        row = data.iloc[i]

        timestamp = _local_timestamp(row["timestamp"])
        trading_date = timestamp.date()

        # ------------------------------------------------------------
        # New trading day
        # ------------------------------------------------------------
        if current_date != trading_date:
            if position is not None:
                previous_row = data.iloc[i - 1]

                previous_timestamp = _local_timestamp(
                    previous_row["timestamp"]
                )

                trade = _close_position(
                    position=position,
                    exit_time=previous_timestamp,
                    exit_price=float(previous_row["close"]),
                    reason="SESSION_CLOSE",
                    cost_settings=cost_settings,
                )

                trades.append(trade)
                record_exit(risk_state, trade.net_pnl)

                position = None

            pending_signal = None
            reset_day(risk_state)
            current_date = trading_date

        # ------------------------------------------------------------
        # Execute pending signal at current bar open
        # ------------------------------------------------------------
        if pending_signal is not None and position is None:
            entry = float(row["open"])
            stop = float(pending_signal.stop)

            decision = check_entry(
                timestamp=timestamp,
                entry=entry,
                stop=stop,
                settings=settings,
                state=risk_state,
            )

            pending_signal = None

            if not decision.approved:
                rejected_signals.append(
                    RejectedSignal(
                        symbol=symbol,
                        timestamp=timestamp,
                        reason=decision.reason,
                    )
                )
            else:
                risk_per_share = decision.position_size.risk_per_share

                target = (
                    entry
                    + risk_per_share * settings.risk_reward_min
                )

                position = _OpenPosition(
                    signal=pending_signal
                    if pending_signal is not None
                    else signal_generator(
                        data.iloc[: i + 1],
                        symbol,
                        settings,
                    ),
                    entry_time=timestamp,
                    entry=entry,
                    stop=stop,
                    target=target,
                    quantity=decision.position_size.quantity,
                )

                record_entry(risk_state)

        # ------------------------------------------------------------
        # Manage open position
        # ------------------------------------------------------------
        if position is not None:
            low = float(row["low"])
            high = float(row["high"])

            stop_hit = low <= position.stop
            target_hit = high >= position.target

            if stop_hit and target_hit:
                trade = _close_position(
                    position=position,
                    exit_time=timestamp,
                    exit_price=position.stop,
                    reason="STOP_TARGET_AMBIGUITY_STOP",
                    cost_settings=cost_settings,
                )

                trades.append(trade)
                record_exit(risk_state, trade.net_pnl)
                position = None

            elif stop_hit:
                trade = _close_position(
                    position=position,
                    exit_time=timestamp,
                    exit_price=position.stop,
                    reason="STOP",
                    cost_settings=cost_settings,
                )

                trades.append(trade)
                record_exit(risk_state, trade.net_pnl)
                position = None

            elif target_hit:
                trade = _close_position(
                    position=position,
                    exit_time=timestamp,
                    exit_price=position.target,
                    reason="TARGET",
                    cost_settings=cost_settings,
                )

                trades.append(trade)
                record_exit(risk_state, trade.net_pnl)
                position = None

            elif timestamp.time() >= SESSION_CLOSE:
                trade = _close_position(
                    position=position,
                    exit_time=timestamp,
                    exit_price=float(row["close"]),
                    reason="SESSION_CLOSE",
                    cost_settings=cost_settings,
                )

                trades.append(trade)
                record_exit(risk_state, trade.net_pnl)
                position = None

        # ------------------------------------------------------------
        # Generate signal for next bar
        # ------------------------------------------------------------
        if position is None and pending_signal is None:
            if i < len(data) - 1:
                signal = signal_generator(
                    data.iloc[: i + 1],
                    symbol,
                    settings,
                )

                if signal is not None:
                    pending_signal = signal

    # ------------------------------------------------------------
    # Force close final position
    # ------------------------------------------------------------
    if position is not None:
        final_row = data.iloc[-1]
        final_timestamp = _local_timestamp(final_row["timestamp"])

        trade = _close_position(
            position=position,
            exit_time=final_timestamp,
            exit_price=float(final_row["close"]),
            reason="SESSION_CLOSE",
            cost_settings=cost_settings,
        )

        trades.append(trade)
        record_exit(risk_state, trade.net_pnl)

    return BacktestResult(
        trades=trades,
        rejected_signals=rejected_signals,
    )


def run_simple_backtest(
    df: pd.DataFrame,
    symbol: str = "TEST",
    settings: Settings | None = None,
) -> list[Trade]:
    """Backward-compatible helper returning only completed trades."""
    return run_backtest(
        df=df,
        symbol=symbol,
        settings=settings,
    ).trades