"""Tests for Experiment #001 portfolio diagnostics."""

from pathlib import Path

import pandas as pd

from src.portfolio_diagnostics import (
    DiagnosticConfig,
    _calculate_mae_mfe,
    _prepare_trades,
    analyze_by_day_of_week,
    analyze_by_exit_reason,
    analyze_by_hour,
    analyze_by_symbol,
    analyze_rejections,
    calculate_consecutive_losses,
    create_trade_sequence,
)


def make_trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "RELIANCE",
                "signal_time": (
                    "2026-08-03 10:00:00+05:30"
                ),
                "entry_time": (
                    "2026-08-03 10:05:00+05:30"
                ),
                "entry": 100.0,
                "stop": 99.0,
                "target": 102.0,
                "quantity": 5,
                "exit_time": (
                    "2026-08-03 10:15:00+05:30"
                ),
                "exit": 102.0,
                "gross_pnl": 10.0,
                "total_cost": 2.0,
                "net_pnl": 8.0,
                "reason": "TARGET",
            },
            {
                "symbol": "HDFCBANK",
                "signal_time": (
                    "2026-08-04 11:00:00+05:30"
                ),
                "entry_time": (
                    "2026-08-04 11:05:00+05:30"
                ),
                "entry": 200.0,
                "stop": 198.0,
                "target": 204.0,
                "quantity": 5,
                "exit_time": (
                    "2026-08-04 11:20:00+05:30"
                ),
                "exit": 198.0,
                "gross_pnl": -10.0,
                "total_cost": 2.0,
                "net_pnl": -12.0,
                "reason": "STOP",
            },
            {
                "symbol": "RELIANCE",
                "signal_time": (
                    "2026-08-05 12:00:00+05:30"
                ),
                "entry_time": (
                    "2026-08-05 12:05:00+05:30"
                ),
                "entry": 100.0,
                "stop": 99.0,
                "target": 102.0,
                "quantity": 5,
                "exit_time": (
                    "2026-08-05 12:30:00+05:30"
                ),
                "exit": 99.0,
                "gross_pnl": -5.0,
                "total_cost": 2.0,
                "net_pnl": -7.0,
                "reason": "STOP",
            },
        ]
    )


def test_prepare_trades_adds_risk_fields():
    result = _prepare_trades(make_trades())

    assert "risk_per_share" in result.columns
    assert "planned_risk" in result.columns
    assert "position_value" in result.columns
    assert "holding_minutes" in result.columns
    assert "net_r_multiple" in result.columns

    assert result.iloc[0]["risk_per_share"] == 1.0
    assert result.iloc[0]["planned_risk"] == 5.0
    assert result.iloc[0]["position_value"] == 500.0


def test_prepare_trades_calculates_holding_time():
    result = _prepare_trades(make_trades())

    assert result.iloc[0]["holding_minutes"] == 10.0


def test_symbol_analysis():
    trades = _prepare_trades(make_trades())

    result = analyze_by_symbol(trades)

    assert set(result["symbol"]) == {
        "RELIANCE",
        "HDFCBANK",
    }

    reliance = result.loc[
        result["symbol"] == "RELIANCE"
    ].iloc[0]

    assert reliance["trades"] == 2
    assert reliance["net_pnl"] == 1.0


def test_exit_reason_analysis():
    trades = _prepare_trades(make_trades())

    result = analyze_by_exit_reason(trades)

    stop = result.loc[
        result["reason"] == "STOP"
    ].iloc[0]

    target = result.loc[
        result["reason"] == "TARGET"
    ].iloc[0]

    assert stop["trades"] == 2
    assert target["trades"] == 1


def test_hour_analysis():
    trades = _prepare_trades(make_trades())

    result = analyze_by_hour(trades)

    assert set(result["entry_hour"]) == {
        10,
        11,
        12,
    }


def test_day_of_week_analysis():
    trades = _prepare_trades(make_trades())

    result = analyze_by_day_of_week(trades)

    assert list(result["day_of_week"]) == [
        "Monday",
        "Tuesday",
        "Wednesday",
    ]


def test_rejection_analysis():
    rejections = pd.DataFrame(
        {
            "symbol": [
                "RELIANCE",
                "HDFCBANK",
                "SBIN",
                "TCS",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2026-08-03 10:00:00+05:30",
                    "2026-08-03 10:05:00+05:30",
                    "2026-08-03 10:10:00+05:30",
                    "2026-08-03 10:15:00+05:30",
                ]
            ),
            "reason": [
                "MAX_TRADES_PER_DAY",
                "MAX_TRADES_PER_DAY",
                "OUTSIDE_ENTRY_WINDOW",
                "MAX_TRADES_PER_DAY",
            ],
        }
    )

    result = analyze_rejections(rejections)

    maximum = result.loc[
        result["reason"] == "MAX_TRADES_PER_DAY"
    ].iloc[0]

    assert maximum["count"] == 3
    assert maximum["percentage"] == 75.0


def test_consecutive_losses():
    trades = _prepare_trades(make_trades())

    assert calculate_consecutive_losses(trades) == 2


def test_trade_sequence():
    trades = _prepare_trades(make_trades())

    result = create_trade_sequence(trades)

    assert list(result["trade_number"]) == [1, 2, 3]
    assert list(result["loss_streak"]) == [0, 1, 2]
    assert result.iloc[-1]["cumulative_net_pnl"] == -11.0


def test_mae_mfe():
    trades = _prepare_trades(make_trades())

    market_data = {
        "RELIANCE": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-08-03 10:05:00+05:30",
                        "2026-08-03 10:10:00+05:30",
                        "2026-08-03 10:15:00+05:30",
                    ]
                ),
                "open": [100.0, 100.5, 101.0],
                "high": [101.0, 102.0, 102.5],
                "low": [99.5, 100.0, 101.0],
                "close": [100.5, 101.5, 102.0],
                "volume": [100, 100, 100],
            }
        )
    }

    result = _calculate_mae_mfe(
        trades.iloc[[0]],
        market_data,
    )

    assert result.iloc[0]["mae"] == -0.5
    assert result.iloc[0]["mfe"] == 2.5


def test_diagnostic_config_defaults():
    config = DiagnosticConfig()

    assert isinstance(config.data_root, Path)
    assert isinstance(config.reports_root, Path)
    assert config.trades_file == "portfolio_trades.csv"
    assert config.rejections_file == (
        "portfolio_rejections.csv"
    )