import pandas as pd
import pytest

from src.market_data import (
    ASIA_KOLKATA,
    DataValidationError,
    YFinanceProvider,
    build_quality_report,
    canonical_symbol,
    normalize_ohlcv,
    validate_ohlcv,
)
from src.trading_calendar import NseTradingCalendar


def make_session_frame(
    trading_date="2026-08-03",
    missing_times=None,
):
    """
    Create a complete NSE 5-minute session.

    NSE regular equity session:
        09:15 through 15:30

    For 5-minute OHLCV candles, timestamps represent the
    beginning of each candle, so the final timestamp is 15:25.

    Therefore a complete session contains 75 bars.
    """
    missing_times = set(missing_times or [])

    timestamps = pd.date_range(
        f"{trading_date} 09:15",
        f"{trading_date} 15:25",
        freq="5min",
        tz=ASIA_KOLKATA,
    )

    timestamps = [
        timestamp
        for timestamp in timestamps
        if timestamp.strftime("%H:%M") not in missing_times
    ]

    rows = []

    for timestamp in timestamps:
        rows.append(
            {
                "timestamp": timestamp,
                "symbol": "TEST",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1000,
            }
        )

    return pd.DataFrame(rows)


def test_canonical_symbol():
    assert canonical_symbol("reliance") == "RELIANCE"
    assert canonical_symbol("RELIANCE.NS") == "RELIANCE"


def test_yfinance_provider_name():
    assert YFinanceProvider.name == "yfinance"


def test_normalize_indexed_ohlcv():
    index = pd.date_range(
        "2026-08-03 09:15",
        periods=3,
        freq="5min",
        tz=ASIA_KOLKATA,
        name="Datetime",
    )

    frame = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1000, 1100, 1200],
        },
        index=index,
    )

    normalized = normalize_ohlcv(
        frame,
        "TEST",
    )

    assert list(normalized.columns) == [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    assert len(normalized) == 3
    assert normalized["symbol"].unique().tolist() == ["TEST"]
    assert str(normalized["timestamp"].dt.tz) == ASIA_KOLKATA


def test_normalize_naive_timestamp_localizes_to_ist():
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2026-08-03 09:15:00",
            ],
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100.5],
            "volume": [1000],
        }
    )

    normalized = normalize_ohlcv(
        frame,
        "TEST",
    )

    assert str(normalized["timestamp"].dt.tz) == ASIA_KOLKATA


def test_normalize_utc_timestamp_converts_to_ist():
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp(
                    "2026-08-03 03:45:00",
                    tz="UTC",
                ),
            ],
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100.5],
            "volume": [1000],
        }
    )

    normalized = normalize_ohlcv(
        frame,
        "TEST",
    )

    assert (
        normalized.loc[0, "timestamp"]
        == pd.Timestamp(
            "2026-08-03 09:15:00",
            tz=ASIA_KOLKATA,
        )
    )


def test_normalize_rejects_multiindex_columns():
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "TEST"),
            ("High", "TEST"),
            ("Low", "TEST"),
            ("Close", "TEST"),
            ("Volume", "TEST"),
        ]
    )

    frame = pd.DataFrame(
        [[100, 101, 99, 100.5, 1000]],
        columns=columns,
    )

    with pytest.raises(
        ValueError,
        match="Multi-symbol",
    ):
        normalize_ohlcv(
            frame,
            "TEST",
        )


def test_normalize_rejects_missing_required_columns():
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2026-08-03 09:15:00",
            ],
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100.5],
        }
    )

    with pytest.raises(
        ValueError,
        match="volume",
    ):
        normalize_ohlcv(
            frame,
            "TEST",
        )


def test_complete_75_bar_session_is_valid():
    frame = make_session_frame()

    report = build_quality_report(frame)

    assert report.rows == 75
    assert report.out_of_session_rows == 0
    assert report.unexpected_trading_dates == 0
    assert report.detected_gaps == 0
    assert report.is_valid


def test_0915_is_valid_session_start():
    frame = make_session_frame()

    report = build_quality_report(frame)

    assert report.first_timestamp == pd.Timestamp(
        "2026-08-03 09:15:00",
        tz=ASIA_KOLKATA,
    )

    assert report.out_of_session_rows == 0


def test_1525_is_valid_final_five_minute_bar():
    frame = make_session_frame()

    report = build_quality_report(frame)

    assert report.last_timestamp == pd.Timestamp(
        "2026-08-03 15:25:00",
        tz=ASIA_KOLKATA,
    )

    assert report.out_of_session_rows == 0


def test_1530_timestamp_is_out_of_session_for_5m_candle():
    frame = make_session_frame()

    frame.loc[len(frame)] = {
        "timestamp": pd.Timestamp(
            "2026-08-03 15:30:00",
            tz=ASIA_KOLKATA,
        ),
        "symbol": "TEST",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000,
    }

    report = build_quality_report(frame)

    assert report.out_of_session_rows == 1


def test_missing_single_middle_bar_is_detected():
    frame = make_session_frame(
        missing_times={"11:30"},
    )

    report = build_quality_report(frame)

    assert report.rows == 74
    assert report.detected_gaps == 1
    assert not report.is_valid


def test_missing_final_two_bars_are_detected():
    frame = make_session_frame(
        missing_times={
            "15:20",
            "15:25",
        },
    )

    report = build_quality_report(frame)

    assert report.rows == 73
    assert report.detected_gaps == 2
    assert not report.is_valid


def test_weekend_date_is_unexpected():
    frame = make_session_frame(
        trading_date="2026-08-08",
    )

    report = build_quality_report(frame)

    assert report.unexpected_trading_dates == 1
    assert not report.is_valid


def test_nse_holiday_is_unexpected():
    # 26-Aug-2026 is an NSE holiday.
    frame = make_session_frame(
        trading_date="2026-08-26",
    )

    report = build_quality_report(frame)

    assert report.unexpected_trading_dates == 1
    assert not report.is_valid


def test_invalid_ohlc_relationship_is_detected():
    frame = make_session_frame()

    frame.loc[0, "high"] = 98.0

    report = build_quality_report(frame)

    assert report.invalid_ohlc_relationships == 1
    assert not report.is_valid


def test_negative_volume_is_detected():
    frame = make_session_frame()

    frame.loc[0, "volume"] = -1

    report = build_quality_report(frame)

    assert report.negative_volume == 1
    assert not report.is_valid


def test_missing_values_are_detected():
    frame = make_session_frame()

    frame.loc[0, "close"] = None

    report = build_quality_report(frame)

    assert report.missing_values["close"] == 1
    assert not report.is_valid


def test_duplicate_timestamp_is_detected():
    frame = make_session_frame()

    frame.loc[1, "timestamp"] = frame.loc[
        0,
        "timestamp",
    ]

    report = build_quality_report(frame)

    assert report.duplicate_timestamps == 1
    assert not report.is_valid


def test_timestamp_order_is_detected():
    frame = make_session_frame()

    frame = pd.concat(
        [
            frame.iloc[:10],
            frame.iloc[[20]],
            frame.iloc[10:20],
            frame.iloc[21:],
        ],
        ignore_index=True,
    )

    report = build_quality_report(frame)

    assert not report.timestamp_ordered
    assert not report.is_valid


def test_validate_ohlcv_returns_report_for_valid_data():
    frame = make_session_frame()

    report = validate_ohlcv(frame)

    assert report.is_valid


def test_validate_ohlcv_raises_for_invalid_data():
    frame = make_session_frame(
        missing_times={"11:30"},
    )

    with pytest.raises(
        DataValidationError,
    ) as exc_info:
        validate_ohlcv(frame)

    assert exc_info.value.report.detected_gaps == 1


def test_only_5m_validation_is_supported():
    frame = make_session_frame()

    with pytest.raises(
        ValueError,
        match="Only 5m",
    ):
        build_quality_report(
            frame,
            interval="1m",
        )


def test_expected_session_contains_75_bars():
    frame = make_session_frame()

    assert len(frame) == 75


def test_august_2026_calendar_contains_holiday():
    calendar = NseTradingCalendar()

    assert not calendar.is_trading_day(
        pd.Timestamp("2026-08-26").date()
    )