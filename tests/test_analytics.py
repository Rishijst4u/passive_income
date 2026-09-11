"""Tests for backtest performance analytics."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.analytics import (
    calculate_max_consecutive_losses,
    calculate_max_drawdown,
    calculate_metrics,
)


def make_trade(
    net_pnl: float,
    gross_pnl: float | None = None,
    total_cost: float = 0.0,
    exit_time: str = "2026-01-05 10:00:00+05:30",
):
    """Create a minimal Trade-like object for analytics tests."""
    if gross_pnl is None:
        gross_pnl = net_pnl + total_cost

    return SimpleNamespace(
        net_pnl=net_pnl,
        gross_pnl=gross_pnl,
        total_cost=total_cost,
        exit_time=pd.Timestamp(exit_time),
    )


def test_max_drawdown():
    pnl_series = pd.Series([100.0, 50.0, -200.0, 50.0, -100.0])

    max_drawdown, max_drawdown_pct = calculate_max_drawdown(
        pnl_series,
        starting_capital=50000.0,
    )

    assert max_drawdown == pytest.approx(-250.0)
    assert max_drawdown_pct == pytest.approx(
        (-250.0 / 50150.0) * 100.0
    )


def test_max_drawdown_empty_series():
    pnl_series = pd.Series(dtype=float)

    max_drawdown, max_drawdown_pct = calculate_max_drawdown(
        pnl_series,
        starting_capital=50000.0,
    )

    assert max_drawdown == 0.0
    assert max_drawdown_pct == 0.0


def test_max_drawdown_uses_equity_not_cumulative_pnl():
    pnl_series = pd.Series([100.0, -50.0, -25.0])

    max_drawdown, max_drawdown_pct = calculate_max_drawdown(
        pnl_series,
        starting_capital=50000.0,
    )

    assert max_drawdown == pytest.approx(-75.0)
    assert max_drawdown_pct == pytest.approx(
        (-75.0 / 50100.0) * 100.0
    )


def test_max_drawdown_no_loss():
    pnl_series = pd.Series([100.0, 50.0, 25.0])

    max_drawdown, max_drawdown_pct = calculate_max_drawdown(
        pnl_series,
        starting_capital=50000.0,
    )

    assert max_drawdown == 0.0
    assert max_drawdown_pct == 0.0


def test_max_drawdown_requires_positive_capital():
    pnl_series = pd.Series([100.0])

    with pytest.raises(ValueError):
        calculate_max_drawdown(
            pnl_series,
            starting_capital=0.0,
        )


def test_max_consecutive_losses():
    pnl_series = pd.Series(
        [100.0, -50.0, -25.0, 10.0, -10.0, -20.0, -30.0]
    )

    assert calculate_max_consecutive_losses(pnl_series) == 3


def test_max_consecutive_losses_empty():
    pnl_series = pd.Series(dtype=float)

    assert calculate_max_consecutive_losses(pnl_series) == 0


def test_max_consecutive_losses_breakeven_breaks_sequence():
    pnl_series = pd.Series([-10.0, -20.0, 0.0, -30.0])

    assert calculate_max_consecutive_losses(pnl_series) == 2


def test_empty_metrics():
    metrics = calculate_metrics(
        [],
        starting_capital=50000.0,
    )

    assert metrics.starting_capital == 50000.0
    assert metrics.total_trades == 0
    assert metrics.winning_trades == 0
    assert metrics.losing_trades == 0
    assert metrics.breakeven_trades == 0
    assert metrics.win_rate == 0.0
    assert metrics.gross_pnl == 0.0
    assert metrics.total_costs == 0.0
    assert metrics.net_pnl == 0.0
    assert metrics.return_pct == 0.0
    assert metrics.max_drawdown == 0.0
    assert metrics.max_drawdown_pct == 0.0


def test_metrics_win_rate_and_pnl():
    trades = [
        make_trade(100.0, gross_pnl=110.0, total_cost=10.0),
        make_trade(-50.0, gross_pnl=-40.0, total_cost=10.0),
        make_trade(0.0, gross_pnl=5.0, total_cost=5.0),
        make_trade(150.0, gross_pnl=160.0, total_cost=10.0),
    ]

    metrics = calculate_metrics(
        trades,
        starting_capital=50000.0,
    )

    assert metrics.total_trades == 4
    assert metrics.winning_trades == 2
    assert metrics.losing_trades == 1
    assert metrics.breakeven_trades == 1
    assert metrics.win_rate == pytest.approx(50.0)

    assert metrics.gross_pnl == pytest.approx(235.0)
    assert metrics.total_costs == pytest.approx(35.0)
    assert metrics.net_pnl == pytest.approx(200.0)
    assert metrics.return_pct == pytest.approx(0.4)


def test_metrics_average_winner_and_loser():
    trades = [
        make_trade(100.0),
        make_trade(200.0),
        make_trade(-50.0),
        make_trade(-100.0),
    ]

    metrics = calculate_metrics(
        trades,
        starting_capital=50000.0,
    )

    assert metrics.average_winner == pytest.approx(150.0)
    assert metrics.average_loser == pytest.approx(-75.0)


def test_metrics_payoff_ratio_and_profit_factor():
    trades = [
        make_trade(100.0),
        make_trade(200.0),
        make_trade(-50.0),
        make_trade(-100.0),
    ]

    metrics = calculate_metrics(
        trades,
        starting_capital=50000.0,
    )

    assert metrics.payoff_ratio == pytest.approx(2.0)
    assert metrics.profit_factor == pytest.approx(2.0)


def test_metrics_expectancy_and_drawdown():
    trades = [
        make_trade(100.0),
        make_trade(50.0),
        make_trade(-200.0),
        make_trade(50.0),
        make_trade(-100.0),
    ]

    metrics = calculate_metrics(
        trades,
        starting_capital=50000.0,
    )

    assert metrics.net_pnl == pytest.approx(-100.0)
    assert metrics.expectancy_per_trade == pytest.approx(-20.0)

    assert metrics.max_drawdown == pytest.approx(-250.0)
    assert metrics.max_drawdown_pct == pytest.approx(
        (-250.0 / 50150.0) * 100.0
    )


def test_metrics_average_daily_pnl():
    trades = [
        make_trade(
            100.0,
            exit_time="2026-01-05 10:00:00+05:30",
        ),
        make_trade(
            -50.0,
            exit_time="2026-01-05 11:00:00+05:30",
        ),
        make_trade(
            200.0,
            exit_time="2026-01-06 10:00:00+05:30",
        ),
    ]

    metrics = calculate_metrics(
        trades,
        starting_capital=50000.0,
    )

    # Day 1 = +50
    # Day 2 = +200
    # Average = +125
    assert metrics.average_daily_pnl == pytest.approx(125.0)


def test_metrics_requires_positive_capital():
    trades = [make_trade(100.0)]

    with pytest.raises(ValueError):
        calculate_metrics(
            trades,
            starting_capital=0.0,
        )