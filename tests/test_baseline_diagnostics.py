import pandas as pd

from src.baseline_diagnostics import (
    analyze_by_day,
    analyze_by_entry_time,
    analyze_by_symbol,
    analyze_exit_reasons,
    analyze_stop_distance,
    calculate_overall_metrics,
)


def make_trades() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "RELIANCE",
                "signal_time": "2026-08-03 10:00:00+05:30",
                "entry_time": "2026-08-03 10:05:00+05:30",
                "entry": 100.0,
                "stop": 99.0,
                "target": 102.0,
                "quantity": 10,
                "exit_time": "2026-08-03 10:15:00+05:30",
                "exit": 102.0,
                "gross_pnl": 20.0,
                "total_cost": 2.0,
                "net_pnl": 18.0,
                "reason": "TARGET",
            },
            {
                "symbol": "HDFCBANK",
                "signal_time": "2026-08-03 11:00:00+05:30",
                "entry_time": "2026-08-03 11:05:00+05:30",
                "entry": 200.0,
                "stop": 199.0,
                "target": 202.0,
                "quantity": 10,
                "exit_time": "2026-08-03 11:15:00+05:30",
                "exit": 199.0,
                "gross_pnl": -10.0,
                "total_cost": 2.0,
                "net_pnl": -12.0,
                "reason": "STOP",
            },
            {
                "symbol": "RELIANCE",
                "signal_time": "2026-08-04 12:00:00+05:30",
                "entry_time": "2026-08-04 12:05:00+05:30",
                "entry": 100.0,
                "stop": 99.0,
                "target": 102.0,
                "quantity": 10,
                "exit_time": "2026-08-04 12:20:00+05:30",
                "exit": 100.5,
                "gross_pnl": 5.0,
                "total_cost": 2.0,
                "net_pnl": 3.0,
                "reason": "SESSION_CLOSE",
            },
        ]
    )


def test_overall_metrics():
    metrics = calculate_overall_metrics(make_trades())

    values = dict(zip(metrics["metric"], metrics["value"]))

    assert values["total_trades"] == 3
    assert values["winning_trades"] == 2
    assert values["losing_trades"] == 1
    assert values["gross_pnl"] == 15.0
    assert values["total_cost"] == 6.0
    assert values["net_pnl"] == 9.0
    assert values["expectancy_per_trade"] == 3.0


def test_exit_reason_analysis():
    result = analyze_exit_reasons(make_trades())

    assert set(result["reason"]) == {
        "TARGET",
        "STOP",
        "SESSION_CLOSE",
    }

    target = result[result["reason"] == "TARGET"].iloc[0]

    assert target["trades"] == 1
    assert target["wins"] == 1
    assert target["net_pnl"] == 18.0


def test_symbol_analysis():
    result = analyze_by_symbol(make_trades())

    assert set(result["symbol"]) == {
        "RELIANCE",
        "HDFCBANK",
    }

    reliance = result[result["symbol"] == "RELIANCE"].iloc[0]

    assert reliance["trades"] == 2
    assert reliance["net_pnl"] == 21.0


def test_day_analysis():
    result = analyze_by_day(make_trades())

    assert len(result) == 2

    first_day = result.iloc[0]

    assert first_day["trades"] == 2
    assert first_day["net_pnl"] == 6.0


def test_entry_time_analysis():
    result = analyze_by_entry_time(make_trades())

    assert len(result) == 3
    assert set(result["entry_clock"]) == {
        "10:05",
        "11:05",
        "12:05",
    }


def test_stop_distance_analysis():
    result = analyze_stop_distance(make_trades())

    assert len(result) == 3
    assert "risk_pct_of_entry" in result.columns
    assert "r_multiple" in result.columns

    assert result["risk_per_share"].eq(1.0).all()