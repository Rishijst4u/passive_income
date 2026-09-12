"""Shared-account portfolio backtest engine for Experiment #001."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Callable

import pandas as pd

from .backtest import RejectedSignal, Trade
from .config import Settings, load_settings
from .costs import CostSettings, calculate_trade_costs
from .risk import calculate_position_size
from .strategy import Signal, generate_signal


ASIA_KOLKATA = "Asia/Kolkata"

# Dhan development dataset ends at 15:10.
SESSION_CLOSE = time(15, 10)


@dataclass
class _PendingSignal:
    symbol: str
    signal: Signal


@dataclass
class _OpenPosition:
    symbol: str
    signal_time: object
    entry_time: object
    entry: float
    stop: float
    target: float
    quantity: int


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """Complete shared-account portfolio backtest result."""

    trades: list[Trade]
    rejected_signals: list[RejectedSignal]


SignalGenerator = Callable[
    [pd.DataFrame, str, Settings],
    Signal | None,
]


def _local_timestamp(timestamp):
    """Normalize a timestamp to Asia/Kolkata."""
    value = pd.Timestamp(timestamp)

    if value.tzinfo is None:
        return value.tz_localize(ASIA_KOLKATA)

    return value.tz_convert(ASIA_KOLKATA)


def _signal_score(signal: object) -> float:
    """
    Extract a deterministic ranking score from a signal.

    The baseline Signal object may not expose a score, so zero is
    used as the neutral fallback.
    """
    for attribute in (
        "score",
        "strategy_score",
        "signal_score",
        "strength",
    ):
        value = getattr(signal, attribute, None)

        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0

    return 0.0


def _prepare_symbol_data(
    frame: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """Normalize and validate the minimum structure needed by the engine."""
    if frame.empty:
        return frame.copy()

    data = frame.copy()

    if "timestamp" not in data.columns:
        raise ValueError(
            f"{symbol}: data must contain a 'timestamp' column"
        )

    required_columns = {
        "open",
        "high",
        "low",
        "close",
    }

    missing = required_columns.difference(data.columns)

    if missing:
        raise ValueError(
            f"{symbol}: missing required columns: "
            f"{sorted(missing)}"
        )

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    data["timestamp"] = data["timestamp"].map(
        _local_timestamp
    )

    data = (
        data.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    if "symbol" not in data.columns:
        data["symbol"] = symbol

    return data


def _build_timeline(
    data_by_symbol: dict[str, pd.DataFrame],
) -> dict[object, list[tuple[str, int]]]:
    """
    Build a global chronological event timeline.

    Each timestamp maps to all symbols having a bar at that timestamp.
    Symbols are sorted alphabetically so tie-breaking is deterministic.
    """
    timeline: dict[object, list[tuple[str, int]]] = {}

    for symbol in sorted(data_by_symbol):
        frame = _prepare_symbol_data(
            data_by_symbol[symbol],
            symbol,
        )

        for index, timestamp in enumerate(
            frame["timestamp"]
        ):
            timeline.setdefault(
                timestamp,
                [],
            ).append(
                (
                    symbol,
                    index,
                )
            )

    for timestamp in timeline:
        timeline[timestamp].sort(
            key=lambda item: item[0]
        )

    return dict(
        sorted(
            timeline.items(),
            key=lambda item: item[0],
        )
    )


def _close_position(
    position: _OpenPosition,
    timestamp,
    price: float,
    reason: str,
    cost_settings: CostSettings,
) -> Trade:
    """Close a portfolio position and calculate all trading costs."""
    exit_price = float(price)

    costs = calculate_trade_costs(
        entry_price=position.entry,
        exit_price=exit_price,
        quantity=position.quantity,
        settings=cost_settings,
    )

    return Trade(
        symbol=position.symbol,
        signal_time=position.signal_time,
        entry_time=position.entry_time,
        entry=position.entry,
        stop=position.stop,
        target=position.target,
        quantity=position.quantity,
        exit_time=timestamp,
        exit=exit_price,
        gross_pnl=costs.gross_pnl,
        total_cost=costs.total_cost,
        net_pnl=costs.net_pnl,
        brokerage=costs.brokerage,
        stt=costs.stt,
        exchange_transaction_charges=(
            costs.exchange_transaction_charges
        ),
        sebi_charges=costs.sebi_charges,
        stamp_duty=costs.stamp_duty,
        gst=costs.gst,
        ipft=costs.ipft,
        slippage=costs.slippage,
        reason=reason,
    )


def _candidate_is_executable(
    timestamp,
    signal: Signal,
) -> bool:
    """
    Determine whether a signal is eligible to be executed
    on the next bar.
    """
    if timestamp.time() > time(14, 30):
        return False

    return True


def run_portfolio_backtest(
    data_by_symbol: dict[str, pd.DataFrame],
    settings: Settings | None = None,
    signal_generator: SignalGenerator = generate_signal,
    cost_settings: CostSettings | None = None,
) -> PortfolioBacktestResult:
    """
    Run a shared-account portfolio backtest.

    Important portfolio rules:

    - One shared account across every symbol.
    - One position open at a time.
    - Maximum three trades per trading day.
    - Maximum 0.5% risk per trade.
    - Maximum ₹10,000 position value.
    - Maximum 1% daily loss.
    - Signals are generated from bar N.
    - Entry occurs at bar N+1 OPEN.
    - Highest-scoring candidate wins when multiple symbols
      generate signals for the same next bar.
    - Stop/target ambiguity is resolved in favour of the stop.
    - No overnight positions.
    - Dhan development-session cutoff is 15:10.
    """

    settings = settings or load_settings()
    cost_settings = cost_settings or CostSettings()

    if not data_by_symbol:
        return PortfolioBacktestResult(
            trades=[],
            rejected_signals=[],
        )

    prepared: dict[str, pd.DataFrame] = {}

    for symbol, frame in data_by_symbol.items():
        canonical_symbol = str(symbol).strip().upper()

        prepared[canonical_symbol] = _prepare_symbol_data(
            frame,
            canonical_symbol,
        )

    timeline = _build_timeline(prepared)

    if not timeline:
        return PortfolioBacktestResult(
            trades=[],
            rejected_signals=[],
        )

    trades: list[Trade] = []
    rejected: list[RejectedSignal] = []

    # Shared portfolio state.
    realized_pnl = 0.0
    daily_realized_pnl = 0.0
    current_date = None
    trades_today = 0

    position: _OpenPosition | None = None

    # Candidate signals generated on the current bar and eligible
    # for execution on the next chronological bar.
    pending_candidates: list[_PendingSignal] = []

    timeline_items = list(timeline.items())

    for timeline_index, (timestamp, bars) in enumerate(
        timeline_items
    ):
        trading_date = timestamp.date()

        # -------------------------------------------------------------
        # New trading day
        # -------------------------------------------------------------
        if current_date is None:
            current_date = trading_date

        elif trading_date != current_date:
            # The dataset is intraday-only, so a position should have
            # already been closed at the previous session cutoff.
            #
            # If not, close it at the last available bar belonging to
            # that previous trading day rather than carrying it
            # overnight.
            if position is not None:
                previous_timestamp, previous_bars = (
                    timeline_items[timeline_index - 1]
                )

                previous_symbol = position.symbol

                previous_row_index = next(
                    (
                        index
                        for symbol, index in previous_bars
                        if symbol == previous_symbol
                    ),
                    None,
                )

                if previous_row_index is not None:
                    previous_row = prepared[
                        previous_symbol
                    ].iloc[previous_row_index]

                    trade = _close_position(
                        position=position,
                        timestamp=previous_timestamp,
                        price=float(previous_row["close"]),
                        reason="SESSION_CLOSE",
                        cost_settings=cost_settings,
                    )

                    trades.append(trade)

                    realized_pnl += trade.net_pnl
                    daily_realized_pnl += trade.net_pnl

                position = None

            pending_candidates.clear()

            current_date = trading_date
            daily_realized_pnl = 0.0
            trades_today = 0

        # -------------------------------------------------------------
        # Execute the best candidate scheduled for THIS bar.
        #
        # Candidates were generated on the previous chronological
        # timestamp, therefore this is the next-bar-open execution.
        # -------------------------------------------------------------
        if pending_candidates and position is None:
            executable_candidates = []

            for candidate in pending_candidates:
                symbol = candidate.symbol

                current_bar_index = next(
                    (
                        index
                        for candidate_symbol, index in bars
                        if candidate_symbol == symbol
                    ),
                    None,
                )

                if current_bar_index is None:
                    continue

                executable_candidates.append(
                    (
                        _signal_score(candidate.signal),
                        symbol,
                        candidate.signal,
                        current_bar_index,
                    )
                )

            pending_candidates.clear()

            if executable_candidates:
                # Highest score wins.
                # Alphabetical symbol order provides deterministic
                # tie-breaking.
                executable_candidates.sort(
                    key=lambda item: (
                        -item[0],
                        item[1],
                    )
                )

                (
                    _score,
                    selected_symbol,
                    selected_signal,
                    selected_index,
                ) = executable_candidates[0]

                row = prepared[
                    selected_symbol
                ].iloc[selected_index]

                entry_time = timestamp
                entry_price = float(row["open"])
                stop_price = float(
                    selected_signal.stop
                )

                # -----------------------------------------------------
                # Portfolio-level risk controls.
                # -----------------------------------------------------
                if entry_time.time() < time(9, 30):
                    rejected.append(
                        RejectedSignal(
                            symbol=selected_symbol,
                            timestamp=selected_signal.timestamp,
                            reason="OUTSIDE_ENTRY_WINDOW",
                        )
                    )

                elif entry_time.time() > time(14, 30):
                    rejected.append(
                        RejectedSignal(
                            symbol=selected_symbol,
                            timestamp=selected_signal.timestamp,
                            reason="OUTSIDE_ENTRY_WINDOW",
                        )
                    )

                elif trades_today >= settings.max_trades_per_day:
                    rejected.append(
                        RejectedSignal(
                            symbol=selected_symbol,
                            timestamp=selected_signal.timestamp,
                            reason="MAX_TRADES_PER_DAY",
                        )
                    )

                else:
                    position_size = calculate_position_size(
                        entry=entry_price,
                        stop=stop_price,
                        settings=settings,
                    )

                    if position_size.quantity <= 0:
                        rejected.append(
                            RejectedSignal(
                                symbol=selected_symbol,
                                timestamp=selected_signal.timestamp,
                                reason="INVALID_POSITION_SIZE",
                            )
                        )

                    else:
                        max_daily_loss = (
                            settings.capital
                            * settings.max_daily_loss_pct
                            / 100.0
                        )

                        projected_daily_pnl = (
                            daily_realized_pnl
                            - position_size.planned_risk
                        )

                        if (
                            projected_daily_pnl
                            < -max_daily_loss
                        ):
                            rejected.append(
                                RejectedSignal(
                                    symbol=selected_symbol,
                                    timestamp=selected_signal.timestamp,
                                    reason="DAILY_LOSS_LIMIT",
                                )
                            )

                        else:
                            risk_per_share = (
                                position_size.risk_per_share
                            )

                            target_price = (
                                entry_price
                                + (
                                    risk_per_share
                                    * settings.risk_reward_min
                                )
                            )

                            position = _OpenPosition(
                                symbol=selected_symbol,
                                signal_time=(
                                    selected_signal.timestamp
                                ),
                                entry_time=entry_time,
                                entry=entry_price,
                                stop=stop_price,
                                target=target_price,
                                quantity=(
                                    position_size.quantity
                                ),
                            )

                            trades_today += 1

        # -------------------------------------------------------------
        # Manage open position using the current bar.
        # -------------------------------------------------------------
        if position is not None:
            position_symbol = position.symbol

            current_position_index = next(
                (
                    index
                    for symbol, index in bars
                    if symbol == position_symbol
                ),
                None,
            )

            if current_position_index is not None:
                row = prepared[
                    position_symbol
                ].iloc[current_position_index]

                high = float(row["high"])
                low = float(row["low"])

                stop_hit = low <= position.stop
                target_hit = high >= position.target

                if stop_hit and target_hit:
                    trade = _close_position(
                        position=position,
                        timestamp=timestamp,
                        price=position.stop,
                        reason=(
                            "STOP_TARGET_AMBIGUITY_STOP"
                        ),
                        cost_settings=cost_settings,
                    )

                    trades.append(trade)
                    realized_pnl += trade.net_pnl
                    daily_realized_pnl += trade.net_pnl
                    position = None

                elif stop_hit:
                    trade = _close_position(
                        position=position,
                        timestamp=timestamp,
                        price=position.stop,
                        reason="STOP",
                        cost_settings=cost_settings,
                    )

                    trades.append(trade)
                    realized_pnl += trade.net_pnl
                    daily_realized_pnl += trade.net_pnl
                    position = None

                elif target_hit:
                    trade = _close_position(
                        position=position,
                        timestamp=timestamp,
                        price=position.target,
                        reason="TARGET",
                        cost_settings=cost_settings,
                    )

                    trades.append(trade)
                    realized_pnl += trade.net_pnl
                    daily_realized_pnl += trade.net_pnl
                    position = None

                elif timestamp.time() >= SESSION_CLOSE:
                    trade = _close_position(
                        position=position,
                        timestamp=timestamp,
                        price=float(row["close"]),
                        reason="SESSION_CLOSE",
                        cost_settings=cost_settings,
                    )

                    trades.append(trade)
                    realized_pnl += trade.net_pnl
                    daily_realized_pnl += trade.net_pnl
                    position = None

        # -------------------------------------------------------------
        # Generate signals for the NEXT chronological bar.
        #
        # We only generate candidates when there is no open position.
        # Each symbol gets the data available through the current bar,
        # preventing look-ahead.
        # -------------------------------------------------------------
        if (
            position is None
            and timestamp.time() < time(14, 30)
        ):
            next_timestamp = (
                timeline_items[timeline_index + 1][0]
                if timeline_index + 1
                < len(timeline_items)
                else None
            )

            if next_timestamp is not None:
                next_date = next_timestamp.date()

                if next_date == trading_date:
                    candidates_for_next_bar = []

                    for symbol in sorted(prepared):
                        frame = prepared[symbol]

                        current_rows = frame[
                            frame["timestamp"] <= timestamp
                        ]

                        if current_rows.empty:
                            continue

                        signal = signal_generator(
                            current_rows,
                            symbol,
                            settings,
                        )

                        if signal is None:
                            continue

                        if not _candidate_is_executable(
                            timestamp,
                            signal,
                        ):
                            continue

                        candidates_for_next_bar.append(
                            _PendingSignal(
                                symbol=symbol,
                                signal=signal,
                            )
                        )

                    pending_candidates = (
                        candidates_for_next_bar
                    )

    # -------------------------------------------------------------
    # Final safety close.
    # -------------------------------------------------------------
    if position is not None:
        final_timestamp = timeline_items[-1][0]

        final_symbol = position.symbol

        final_index = next(
            (
                index
                for symbol, index in timeline_items[-1][1]
                if symbol == final_symbol
            ),
            None,
        )

        if final_index is not None:
            final_row = prepared[
                final_symbol
            ].iloc[final_index]

            trade = _close_position(
                position=position,
                timestamp=final_timestamp,
                price=float(final_row["close"]),
                reason="SESSION_CLOSE",
                cost_settings=cost_settings,
            )

            trades.append(trade)

    return PortfolioBacktestResult(
        trades=trades,
        rejected_signals=rejected,
    )