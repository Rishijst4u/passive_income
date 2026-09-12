import pandas as pd

from src.multi_backtest import (
    calculate_symbol_metrics,
    find_data_file,
    rejection_to_dict,
    trade_to_dict,
)


def test_find_data_file_for_reliance():
    path = find_data_file("RELIANCE")

    assert path.exists()
    assert path.suffix == ".parquet"
    assert path.parent.name == "5m"
    assert path.parent.parent.name == "RELIANCE"


def test_find_data_file_rejects_unknown_symbol():
    try:
        find_data_file("NOT_A_REAL_SYMBOL")
    except FileNotFoundError as exc:
        assert "NOT_A_REAL_SYMBOL" in str(exc)
    else:
        raise AssertionError(
            "Expected FileNotFoundError"
        )


def test_calculate_symbol_metrics_with_no_trades():
    metrics = calculate_symbol_metrics(
        symbol="TEST",
        trades=[],
        starting_capital=50_000.0,
        rejected_signals=3,
    )

    assert metrics["symbol"] == "TEST"
    assert metrics["trades"] == 0
    assert metrics["wins"] == 0
    assert metrics["losses"] == 0
    assert metrics["net_pnl"] == 0.0
    assert metrics["rejected_signals"] == 3


class FakeTrade:
    signal_time = pd.Timestamp(
        "2026-08-03 10:00",
        tz="Asia/Kolkata",
    )
    entry_time = pd.Timestamp(
        "2026-08-03 10:05",
        tz="Asia/Kolkata",
    )
    entry = 100.0
    stop = 99.0
    target = 102.0
    quantity = 10
    exit_time = pd.Timestamp(
        "2026-08-03 10:30",
        tz="Asia/Kolkata",
    )
    exit = 102.0
    gross_pnl = 20.0
    total_cost = 2.0
    net_pnl = 18.0
    brokerage = 0.5
    stt = 0.5
    exchange_transaction_charges = 0.1
    sebi_charges = 0.01
    stamp_duty = 0.03
    gst = 0.1
    ipft = 0.001
    slippage = 0.2
    reason = "TARGET"


class FakeRejection:
    timestamp = pd.Timestamp(
        "2026-08-03 10:00",
        tz="Asia/Kolkata",
    )
    reason = "OUTSIDE_ENTRY_WINDOW"


def test_trade_to_dict():
    result = trade_to_dict(
        "RELIANCE",
        FakeTrade(),
    )

    assert result["symbol"] == "RELIANCE"
    assert result["entry"] == 100.0
    assert result["stop"] == 99.0
    assert result["target"] == 102.0
    assert result["quantity"] == 10
    assert result["net_pnl"] == 18.0
    assert result["reason"] == "TARGET"


def test_rejection_to_dict():
    result = rejection_to_dict(
        "RELIANCE",
        FakeRejection(),
    )

    assert result["symbol"] == "RELIANCE"
    assert (
        result["reason"]
        == "OUTSIDE_ENTRY_WINDOW"
    )


def test_calculate_symbol_metrics():
    trade_one = FakeTrade()

    trade_two = FakeTrade()
    trade_two.net_pnl = -10.0
    trade_two.gross_pnl = -8.0

    metrics = calculate_symbol_metrics(
        symbol="RELIANCE",
        trades=[
            trade_one,
            trade_two,
        ],
        starting_capital=50_000.0,
        rejected_signals=5,
    )

    assert metrics["trades"] == 2
    assert metrics["wins"] == 1
    assert metrics["losses"] == 1
    assert metrics["win_rate_pct"] == 50.0
    assert metrics["net_pnl"] == 8.0
    assert metrics["gross_pnl"] == 12.0
    assert metrics["rejected_signals"] == 5