import pandas as pd

from src.config import load_settings
from src.portfolio_backtest import (
    PortfolioBacktestResult,
    _build_timeline,
    _signal_score,
    run_portfolio_backtest,
)


def make_settings():
    return load_settings()


def make_frame(
    symbol: str,
    timestamps: list[str],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps).tz_localize(
                "Asia/Kolkata"
            ),
            "symbol": symbol,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000] * len(timestamps),
        }
    )


def test_empty_portfolio_returns_empty_result():
    result = run_portfolio_backtest(
        data_by_symbol={},
        settings=make_settings(),
    )

    assert isinstance(result, PortfolioBacktestResult)
    assert result.trades == []
    assert result.rejected_signals == []


def test_signal_score_supports_score_attribute():
    class SignalObject:
        score = 0.75

    assert _signal_score(SignalObject()) == 0.75


def test_signal_score_supports_strategy_score_attribute():
    class SignalObject:
        strategy_score = 0.81

    assert _signal_score(SignalObject()) == 0.81


def test_signal_score_defaults_to_zero():
    class SignalObject:
        pass

    assert _signal_score(SignalObject()) == 0.0


def test_build_timeline_combines_symbols():
    timestamps = [
        "2026-08-03 09:15",
        "2026-08-03 09:20",
    ]

    data = {
        "AAA": make_frame(
            "AAA",
            timestamps,
            [100, 101],
            [101, 102],
            [99, 100],
            [100.5, 101.5],
        ),
        "BBB": make_frame(
            "BBB",
            timestamps,
            [200, 201],
            [201, 202],
            [199, 200],
            [200.5, 201.5],
        ),
    }

    timeline = _build_timeline(data)

    assert len(timeline) == 2

    first_timestamp = next(iter(timeline))

    assert timeline[first_timestamp] == [
        ("AAA", 0),
        ("BBB", 0),
    ]


def test_portfolio_does_not_open_position_without_signal():
    timestamps = [
        "2026-08-03 09:15",
        "2026-08-03 09:20",
        "2026-08-03 09:25",
    ]

    data = {
        "AAA": make_frame(
            "AAA",
            timestamps,
            [100, 101, 102],
            [101, 102, 103],
            [99, 100, 101],
            [100.5, 101.5, 102.5],
        )
    }

    def no_signal(data, symbol, settings):
        return None

    result = run_portfolio_backtest(
        data_by_symbol=data,
        settings=make_settings(),
        signal_generator=no_signal,
    )

    assert result.trades == []
    assert result.rejected_signals == []


def test_multiple_symbols_share_one_portfolio():
    timestamps = [
        "2026-08-03 09:15",
        "2026-08-03 09:20",
        "2026-08-03 09:25",
        "2026-08-03 09:30",
        "2026-08-03 09:35",
    ]

    data = {
        "AAA": make_frame(
            "AAA",
            timestamps,
            [100, 100, 100, 100, 100],
            [101, 101, 101, 101, 101],
            [99, 99, 99, 99, 99],
            [100, 100, 100, 100, 100],
        ),
        "BBB": make_frame(
            "BBB",
            timestamps,
            [200, 200, 200, 200, 200],
            [201, 201, 201, 201, 201],
            [199, 199, 199, 199, 199],
            [200, 200, 200, 200, 200],
        ),
    }

    class FakeSignal:
        def __init__(self, symbol, timestamp, stop, score):
            self.symbol = symbol
            self.timestamp = timestamp
            self.stop = stop
            self.score = score

    signal_counts = {
        "AAA": 0,
        "BBB": 0,
    }

    def signal_generator(frame, symbol, settings):
        signal_counts[symbol] += 1

        if len(frame) != 1:
            return None

        timestamp = frame.iloc[-1]["timestamp"]

        return FakeSignal(
            symbol=symbol,
            timestamp=timestamp,
            stop=(
                99.0
                if symbol == "AAA"
                else 199.0
            ),
            score=(
                0.80
                if symbol == "AAA"
                else 0.60
            ),
        )

    result = run_portfolio_backtest(
        data_by_symbol=data,
        settings=make_settings(),
        signal_generator=signal_generator,
    )

    assert len(result.trades) <= 1


def test_portfolio_result_preserves_trade_objects():
    result = run_portfolio_backtest(
        data_by_symbol={},
        settings=make_settings(),
    )

    assert isinstance(result.trades, list)
    assert isinstance(result.rejected_signals, list)