"""Tests for Experiment #001 reporting helpers."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.report import (
    format_metrics_report,
    rejected_signals_to_dataframe,
    trades_to_dataframe,
)


def make_trade():
    """Create a minimal Trade-like object."""

    return SimpleNamespace(
        symbol="RELIANCE",
        signal_time=pd.Timestamp("2026-01-05 10:00:00+05:30"),
        entry_time=pd.Timestamp("2026-01-05 10:05:00+05:30"),
        entry=100.0,
        stop=99.0,
        target=102.0,
        quantity=10,
        exit_time=pd.Timestamp("2026-01-05 10:15:00+05:30"),
        exit=102.0,
        gross_pnl=20.0,
        total_cost=2.0,
        net_pnl=18.0,
        brokerage=1.0,
        stt=0.5,
        exchange_transaction_charges=0.1,
        sebi_charges=0.01,
        stamp_duty=0.1,
        gst=0.2,
        ipft=0.01,
        slippage=0.08,
        reason="TARGET",
    )


def test_trades_to_dataframe():
    result = SimpleNamespace(
        trades=[make_trade()],
        rejected_signals=[],
    )

    df = trades_to_dataframe(result)

    assert len(df) == 1
    assert df.loc[0, "symbol"] == "RELIANCE"
    assert df.loc[0, "quantity"] == 10
    assert df.loc[0, "net_pnl"] == pytest.approx(18.0)
    assert df.loc[0, "reason"] == "TARGET"


def test_trades_to_dataframe_empty():
    result = SimpleNamespace(
        trades=[],
        rejected_signals=[],
    )

    df = trades_to_dataframe(result)

    assert df.empty


def test_rejected_signals_to_dataframe():
    result = SimpleNamespace(
        trades=[],
        rejected_signals=[
            SimpleNamespace(
                symbol="RELIANCE",
                timestamp=pd.Timestamp(
                    "2026-01-05 10:00:00+05:30"
                ),
                reason="MAX_TRADES_PER_DAY",
            )
        ],
    )

    df = rejected_signals_to_dataframe(result)

    assert len(df) == 1
    assert df.loc[0, "symbol"] == "RELIANCE"
    assert df.loc[0, "reason"] == "MAX_TRADES_PER_DAY"


def test_rejected_signals_to_dataframe_empty():
    result = SimpleNamespace(
        trades=[],
        rejected_signals=[],
    )

    df = rejected_signals_to_dataframe(result)

    assert df.empty
    assert list(df.columns) == [
        "symbol",
        "timestamp",
        "reason",
    ]


def test_format_metrics_report():
    metrics = SimpleNamespace(
        starting_capital=50000.0,
        total_trades=10,
        winning_trades=6,
        losing_trades=4,
        breakeven_trades=0,
        win_rate=60.0,
        gross_pnl=1000.0,
        total_costs=150.0,
        net_pnl=850.0,
        return_pct=1.7,
        average_winner=200.0,
        average_loser=-125.0,
        expectancy_per_trade=85.0,
        payoff_ratio=1.6,
        profit_factor=2.4,
        max_drawdown=-250.0,
        max_drawdown_pct=-0.5,
        max_consecutive_losses=2,
        average_daily_pnl=85.0,
    )

    report = format_metrics_report(
        metrics,
        rejected_signals=3,
    )

    assert "Experiment #001" in report
    assert "VWAP + Momentum + Volume Breakout v1" in report
    assert "Starting Capital : ₹50,000.00" in report
    assert "Total Trades     : 10" in report
    assert "Rejected Signals : 3" in report
    assert "Net P&L          : ₹850.00" in report
    assert "Max Drawdown %   : -0.50%" in report