from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class StrategySettings:
    ema_fast: int
    ema_slow: int
    rsi_min: float
    rsi_max: float
    volume_ratio_min: float
    breakout_lookback: int


@dataclass(frozen=True)
class Settings:
    capital: float
    max_risk_per_trade_pct: float
    max_position_value: float
    max_daily_loss_pct: float
    max_trades_per_day: int
    risk_reward_min: float
    timeframe: str
    entry_start: str
    entry_end: str
    strategy: StrategySettings


class ConfigurationError(ValueError):
    """Raised when a settings file is missing required configuration values."""


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def load_settings(path: str | Path | None = None) -> Settings:
    """Load local YAML settings into typed, immutable configuration objects."""
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    with settings_path.open(encoding="utf-8") as settings_file:
        raw: Mapping[str, Any] = yaml.safe_load(settings_file) or {}

    try:
        strategy_raw = raw["strategy"]
        if not isinstance(strategy_raw, Mapping):
            raise TypeError("strategy must be a mapping")
        return Settings(
            capital=float(raw["capital"]),
            max_risk_per_trade_pct=float(raw["max_risk_per_trade_pct"]),
            max_position_value=float(raw["max_position_value"]),
            max_daily_loss_pct=float(raw["max_daily_loss_pct"]),
            max_trades_per_day=int(raw["max_trades_per_day"]),
            risk_reward_min=float(raw["risk_reward_min"]),
            timeframe=str(raw["timeframe"]),
            entry_start=str(raw["entry_start"]),
            entry_end=str(raw["entry_end"]),
            strategy=StrategySettings(
                ema_fast=int(strategy_raw["ema_fast"]),
                ema_slow=int(strategy_raw["ema_slow"]),
                rsi_min=float(strategy_raw["rsi_min"]),
                rsi_max=float(strategy_raw["rsi_max"]),
                volume_ratio_min=float(strategy_raw["volume_ratio_min"]),
                breakout_lookback=int(strategy_raw["breakout_lookback"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigurationError(
            f"Invalid settings file: {settings_path}"
        ) from error
