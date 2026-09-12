import pandas as pd
import pytest

from src import strategy
from src.config import ConfigurationError, Settings, StrategySettings, load_settings


def test_loads_repository_settings():
    settings = load_settings()

    assert isinstance(settings, Settings)
    assert settings.capital == 50000.0
    assert settings.max_risk_per_trade_pct == 0.5
    assert settings.max_position_value == 10000.0
    assert settings.max_daily_loss_pct == 1.0
    assert settings.max_trades_per_day == 3
    assert settings.risk_reward_min == 2.0
    assert settings.timeframe == "5m"
    assert settings.entry_start == "09:30"
    assert settings.entry_end == "14:30"
    assert settings.strategy == StrategySettings(
        9,
        21,
        55.0,
        70.0,
        1.5,
        3,
    )


def test_missing_required_value_raises_configuration_error(monkeypatch):
    monkeypatch.setattr(
        "src.config.yaml.safe_load",
        lambda _: {"strategy": {}},
    )

    with pytest.raises(ConfigurationError, match="Invalid settings file"):
        load_settings()


def test_strategy_uses_supplied_configuration(monkeypatch):
    settings = Settings(
        capital=50000,
        max_risk_per_trade_pct=0.5,
        max_position_value=10000,
        max_daily_loss_pct=1.0,
        max_trades_per_day=3,
        risk_reward_min=3.0,
        timeframe="5m",
        entry_start="09:30",
        entry_end="14:30",
        strategy=StrategySettings(
            5,
            10,
            50,
            75,
            2.0,
            4,
        ),
    )

    captured = {}

    def indicator_frame(_, strategy_settings):
        captured["strategy_settings"] = strategy_settings

        return pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp(
                        "2026-01-01 09:30",
                        tz="Asia/Kolkata",
                    ),
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.5,
                    "close": 101.0,
                    "volume": 200.0,
                    "ema9": 101.0,
                    "ema21": 100.0,
                    "ema50": 98.0,
                    "rsi": 60.0,
                    "vwap": 100.0,
                    "vol_ratio": 2.1,
                    "body_ratio": 0.5,
                    "prior_high": 100.0,
                    "prior_low": 98.0,
                    "prev_close": 100.0,
                    "prev_low": 99.5,
                    "prev_high": 101.0,
                    "prev_ema21": 100.0,
                    "prev_vwap": 100.0,
                },
                {
                    "timestamp": pd.Timestamp(
                        "2026-01-01 09:35",
                        tz="Asia/Kolkata",
                    ),
                    "open": 100.5,
                    "high": 103.0,
                    "low": 100.5,
                    "close": 102.0,
                    "volume": 250.0,
                    "ema9": 101.5,
                    "ema21": 100.5,
                    "ema50": 98.5,
                    "rsi": 60.0,
                    "vwap": 100.5,
                    "vol_ratio": 2.1,
                    "body_ratio": 0.6,
                    "prior_high": 102.0,
                    "prior_low": 99.0,
                    "prev_close": 101.0,
                    "prev_low": 100.5,
                    "prev_high": 102.0,
                    "prev_ema21": 100.5,
                    "prev_vwap": 100.5,
                },
            ]
        )

    monkeypatch.setattr(
        strategy,
        "add_indicators",
        indicator_frame,
    )

    signal = strategy.generate_signal(
        pd.DataFrame(),
        symbol="TEST",
        settings=settings,
    )

    assert captured["strategy_settings"] == settings.strategy
    assert signal is not None

    # Entry = 102
    # Stop = 100.5
    # Risk = 1.5
    # Configured R:R = 3
    # Target = 102 + (1.5 * 3) = 106.5
    assert signal.target == 106.5