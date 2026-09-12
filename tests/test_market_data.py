import pandas as pd

from src.market_data import build_quality_report


def test_partial_session_is_development_usable():
    timestamps = pd.date_range(
        "2026-08-31 09:15",
        "2026-08-31 15:10",
        freq="5min",
        tz="Asia/Kolkata",
    )

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "RELIANCE",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
        }
    )

    report = build_quality_report(
        frame,
        interval="5m",
    )

    assert report.session_status == "PARTIAL_SESSION"
    assert report.missing_session_bars == 3
    assert report.trailing_missing_session_bars == 3
    assert report.is_development_usable is True
    assert report.is_valid is False
    assert report.production_qualified is False


def test_internal_session_gap_is_not_development_usable():
    timestamps = pd.date_range(
        "2026-08-31 09:15",
        "2026-08-31 15:10",
        freq="5min",
        tz="Asia/Kolkata",
    )

    timestamps = timestamps.delete(
        timestamps.get_loc(
            pd.Timestamp(
                "2026-08-31 11:45",
                tz="Asia/Kolkata",
            )
        )
    )

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "RELIANCE",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
        }
    )

    report = build_quality_report(
        frame,
        interval="5m",
    )

    assert report.session_status == "INVALID"
    assert report.missing_session_bars == 4
    assert report.trailing_missing_session_bars == 3
    assert report.is_development_usable is False
    assert report.is_valid is False


def test_full_session_is_production_qualified():
    timestamps = pd.date_range(
        "2026-08-31 09:15",
        "2026-08-31 15:25",
        freq="5min",
        tz="Asia/Kolkata",
    )

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "RELIANCE",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
        }
    )

    report = build_quality_report(
        frame,
        interval="5m",
    )

    assert report.session_status == "FULL_SESSION"
    assert report.missing_session_bars == 0
    assert report.trailing_missing_session_bars == 0
    assert report.is_development_usable is True
    assert report.is_valid is True
    assert report.production_qualified is True