"""Regression tests for strategy signal timestamps."""

from __future__ import annotations

import pandas as pd

import src.strategy as strategy
from src.config import load_settings


def test_signal_timestamp_uses_timestamp_column(monkeypatch):
    settings = load_settings()

    signal_timestamp = pd.Timestamp(
        "2026-08-03 10:25:00",
        tz="Asia/Kolkata",
    )

    indicator_frame = pd.DataFrame(
        [
            {
                "timestamp": signal_timestamp,
                "close": 101.0,
                "low": 100.0,
                "vwap": 100.0,
                "ema9": 101.0,
                "ema21": 100.0,
                "rsi": 60.0,
                "vol_ratio": 2.0,
                "prior_high": 100.0,
            }
        ],
        index=[123],
    )

    monkeypatch.setattr(
        strategy,
        "add_indicators",
        lambda _, __: indicator_frame,
    )

    signal = strategy.generate_signal(
        pd.DataFrame(),
        symbol="TEST",
        settings=settings,
    )

    assert signal is not None
    assert pd.Timestamp(signal.timestamp) == signal_timestamp


def test_signal_timestamp_falls_back_to_datetime_index(monkeypatch):
    settings = load_settings()

    signal_timestamp = pd.Timestamp(
        "2026-08-03 10:30:00",
        tz="Asia/Kolkata",
    )

    indicator_frame = pd.DataFrame(
        [
            {
                "close": 101.0,
                "low": 100.0,
                "vwap": 100.0,
                "ema9": 101.0,
                "ema21": 100.0,
                "rsi": 60.0,
                "vol_ratio": 2.0,
                "prior_high": 100.0,
            }
        ],
        index=pd.DatetimeIndex([signal_timestamp]),
    )

    monkeypatch.setattr(
        strategy,
        "add_indicators",
        lambda _, __: indicator_frame,
    )

    signal = strategy.generate_signal(
        pd.DataFrame(),
        symbol="TEST",
        settings=settings,
    )

    assert signal is not None
    assert pd.Timestamp(signal.timestamp) == signal_timestamp