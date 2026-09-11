import pandas as pd
import pytest

from src.market_data import (
    ASIA_KOLKATA,
    DataValidationError,
    build_quality_report,
    normalize_ohlcv,
    validate_ohlcv,
)
from src.trading_calendar import (
    DEFAULT_NSE_TRADING_CALENDAR,
    TradingDayType,
)


def raw_frame():
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.DatetimeIndex(
            ["2026-01-05 09:15", "2026-01-05 09:20", "2026-01-05 09:25"],
            name="Datetime",
        ),
    )


def test_normalizes_schema_and_naive_timestamps_to_kolkata():
    normalized = normalize_ohlcv(raw_frame(), "reliance.ns")

    assert list(normalized.columns) == [
        "timestamp", "symbol", "open", "high", "low", "close", "volume",
    ]
    assert normalized["symbol"].unique().tolist() == ["RELIANCE"]
    assert str(normalized["timestamp"].dt.tz) == ASIA_KOLKATA


def test_normalizes_utc_timestamps_to_kolkata():
    frame = raw_frame()
    frame.index = frame.index.tz_localize("UTC")

    normalized = normalize_ohlcv(frame, "TCS")

    assert normalized.iloc[0]["timestamp"] == pd.Timestamp("2026-01-05 14:45", tz=ASIA_KOLKATA)


def test_calendar_identifies_weekends():
    assert DEFAULT_NSE_TRADING_CALENDAR.day_type(pd.Timestamp("2026-01-24").date()) is TradingDayType.WEEKEND


def test_calendar_identifies_known_nse_holidays():
    assert DEFAULT_NSE_TRADING_CALENDAR.day_type(pd.Timestamp("2026-01-26").date()) is TradingDayType.MARKET_HOLIDAY


def test_calendar_identifies_normal_trading_days():
    assert DEFAULT_NSE_TRADING_CALENDAR.day_type(pd.Timestamp("2026-01-27").date()) is TradingDayType.TRADING_DAY


def test_missing_required_column_is_rejected():
    frame = raw_frame().drop(columns="Volume")

    with pytest.raises(ValueError, match="missing required columns"):
        normalize_ohlcv(frame, "TCS")


def test_duplicate_timestamps_are_rejected():
    frame = raw_frame()
    frame.index = pd.DatetimeIndex(["2026-01-05 09:15", "2026-01-05 09:15", "2026-01-05 09:25"])

    with pytest.raises(DataValidationError) as error:
        validate_ohlcv(normalize_ohlcv(frame, "TCS"))

    assert error.value.report.duplicate_timestamps == 1


def test_invalid_ohlc_relationships_are_rejected():
    frame = raw_frame()
    frame.iloc[0, frame.columns.get_loc("High")] = 98.0

    with pytest.raises(DataValidationError) as error:
        validate_ohlcv(normalize_ohlcv(frame, "TCS"))

    assert error.value.report.invalid_ohlc_relationships == 1


def test_negative_volume_is_rejected():
    frame = raw_frame()
    frame.loc[1, "Volume"] = -1

    with pytest.raises(DataValidationError) as error:
        validate_ohlcv(normalize_ohlcv(frame, "TCS"))

    assert error.value.report.negative_volume == 1


def test_holiday_between_sessions_does_not_create_a_gap():
    frame = raw_frame().iloc[:2].copy()
    frame.index = pd.DatetimeIndex(["2026-01-23 15:30", "2026-01-27 09:15"])

    report = validate_ohlcv(normalize_ohlcv(frame, "TCS"))

    assert report.detected_gaps == 0


def test_unexpected_intraday_gap_is_reported():
    frame = raw_frame().iloc[:2].copy()
    frame.index = pd.DatetimeIndex(["2026-01-05 09:15", "2026-01-05 09:25"])

    with pytest.raises(DataValidationError) as error:
        validate_ohlcv(normalize_ohlcv(frame, "TCS"))

    assert error.value.report.detected_gaps == 1


def test_non_numeric_ohlcv_value_is_reported_as_missing():
    frame = raw_frame()
    frame["Close"] = frame["Close"].astype(object)
    frame.iloc[0, frame.columns.get_loc("Close")] = "not-a-price"

    with pytest.raises(DataValidationError) as error:
        validate_ohlcv(normalize_ohlcv(frame, "TCS"))

    assert error.value.report.missing_values["close"] == 1


def test_valid_data_is_accepted_with_deterministic_quality_report():
    normalized = normalize_ohlcv(raw_frame(), "TCS")

    report = validate_ohlcv(normalized)

    assert report.rows == 3
    assert report.first_timestamp == pd.Timestamp("2026-01-05 09:15", tz=ASIA_KOLKATA)
    assert report.last_timestamp == pd.Timestamp("2026-01-05 09:25", tz=ASIA_KOLKATA)
    assert report.detected_gaps == 0
    assert report == build_quality_report(normalized)
