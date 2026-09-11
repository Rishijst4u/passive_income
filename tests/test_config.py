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
    assert settings.strategy == StrategySettings(9, 21, 55.0, 70.0, 1.5, 3)


def test_missing_required_value_raises_configuration_error(monkeypatch):
    monkeypatch.setattr("src.config.yaml.safe_load", lambda _: {"strategy": {}})

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
        strategy=StrategySettings(5, 10, 50, 75, 2.0, 4),
    )
    captured = {}

    def indicator_frame(_, strategy_settings):
        captured["strategy_settings"] = strategy_settings
        return pd.DataFrame(
            [{
                "close": 101.0,
                "low": 100.0,
                "vwap": 100.0,
                "ema9": 101.0,
                "ema21": 100.0,
                "rsi": 60.0,
                "vol_ratio": 2.1,
                "prior_high": 100.0,
            }],
            index=pd.DatetimeIndex(["2026-01-01 09:30"]),
        )

    monkeypatch.setattr(strategy, "add_indicators", indicator_frame)

    signal = strategy.generate_signal(pd.DataFrame(), settings=settings)

    assert captured["strategy_settings"] == settings.strategy
    assert signal is not None
    assert signal.target == 104.0
