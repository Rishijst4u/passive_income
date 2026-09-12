from dataclasses import dataclass

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volume import VolumeWeightedAveragePrice

from .config import Settings, StrategySettings, load_settings


@dataclass
class Signal:
    symbol: str
    timestamp: object
    action: str
    price: float
    stop: float
    target: float
    score: int


def add_indicators(
    df: pd.DataFrame, strategy_settings: StrategySettings | None = None
) -> pd.DataFrame:
    strategy_settings = strategy_settings or load_settings().strategy

    out = df.copy()

    out["ema9"] = EMAIndicator(
        out["close"], window=strategy_settings.ema_fast
    ).ema_indicator()

    out["ema21"] = EMAIndicator(
        out["close"], window=strategy_settings.ema_slow
    ).ema_indicator()

    out["rsi"] = RSIIndicator(out["close"], window=14).rsi()

    vwap = VolumeWeightedAveragePrice(
        out["high"],
        out["low"],
        out["close"],
        out["volume"],
        window=14,
    )

    out["vwap"] = vwap.volume_weighted_average_price()
    out["vol_avg"] = out["volume"].rolling(20).mean()
    out["vol_ratio"] = out["volume"] / out["vol_avg"]

    out["prior_high"] = (
        out["high"]
        .rolling(strategy_settings.breakout_lookback)
        .max()
        .shift(1)
    )

    return out


def generate_signal(
    df: pd.DataFrame, symbol: str = "TEST", settings: Settings | None = None
) -> Signal | None:
    settings = settings or load_settings()
    strategy_settings = settings.strategy

    x = add_indicators(df, strategy_settings).dropna()

    if x.empty:
        return None

    r = x.iloc[-1]

    conditions = [
        r.close > r.vwap,
        r.ema9 > r.ema21,
        strategy_settings.rsi_min <= r.rsi <= strategy_settings.rsi_max,
        r.vol_ratio > strategy_settings.volume_ratio_min,
        r.close > r.prior_high,
    ]

    if not all(conditions):
        return None

    entry = float(r.close)

    # Initial research stop: below the latest bar low.
    stop = float(r.low)

    risk = entry - stop

    if risk <= 0:
        return None

    target = entry + settings.risk_reward_min * risk

    score = int(sum(conditions) / len(conditions) * 100)

    # The signal timestamp must represent the actual market bar.
    #
    # Production market-data frames contain a timestamp column.
    # Existing tests/mocked indicator frames may instead use a
    # DatetimeIndex, so support both without changing strategy logic.
    if "timestamp" in x.columns:
        signal_timestamp = r["timestamp"]
    else:
        signal_timestamp = x.index[-1]

    return Signal(
        symbol,
        signal_timestamp,
        "BUY",
        entry,
        stop,
        target,
        score,
    )