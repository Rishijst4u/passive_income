"""Trading strategy for Experiment #001.

Strategy #002:
Trend + VWAP Pullback + Confirmation.

The strategy remains long-only during the initial research phase.
"""

from dataclasses import dataclass

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volume import VolumeWeightedAveragePrice

from .config import Settings, StrategySettings, load_settings


@dataclass
class Signal:
    """Trading signal produced by the strategy."""

    symbol: str
    timestamp: object
    action: str
    price: float
    stop: float
    target: float
    score: int


def add_indicators(
    df: pd.DataFrame,
    strategy_settings: StrategySettings | None = None,
) -> pd.DataFrame:
    """Add indicators required by Strategy #002."""

    strategy_settings = strategy_settings or load_settings().strategy

    out = df.copy()

    out["ema9"] = EMAIndicator(
        out["close"],
        window=strategy_settings.ema_fast,
    ).ema_indicator()

    out["ema21"] = EMAIndicator(
        out["close"],
        window=strategy_settings.ema_slow,
    ).ema_indicator()

    out["ema50"] = EMAIndicator(
        out["close"],
        window=50,
    ).ema_indicator()

    out["rsi"] = RSIIndicator(
        out["close"],
        window=14,
    ).rsi()

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

    out["bar_range"] = out["high"] - out["low"]

    out["body"] = out["close"] - out["open"]

    out["body_ratio"] = (
        out["body"].abs()
        / out["bar_range"].replace(0, pd.NA)
    )

    out["prior_high"] = (
        out["high"]
        .rolling(strategy_settings.breakout_lookback)
        .max()
        .shift(1)
    )

    out["prior_low"] = (
        out["low"]
        .rolling(strategy_settings.breakout_lookback)
        .min()
        .shift(1)
    )

    out["prev_close"] = out["close"].shift(1)
    out["prev_low"] = out["low"].shift(1)
    out["prev_high"] = out["high"].shift(1)
    out["prev_ema21"] = out["ema21"].shift(1)
    out["prev_vwap"] = out["vwap"].shift(1)

    return out


def _calculate_signal_score(row: pd.Series) -> int:
    """Calculate a quality score for a valid setup."""

    score = 0

    ema9 = float(row["ema9"])
    ema21 = float(row["ema21"])
    close = float(row["close"])
    vwap = float(row["vwap"])
    rsi = float(row["rsi"])
    vol_ratio = float(row["vol_ratio"])

    if "ema50" in row.index and pd.notna(row["ema50"]):
        ema50 = float(row["ema50"])

        if ema21 > ema50:
            score += 20

            if ema21 != 0:
                trend_distance_pct = (
                    (ema21 - ema50) / ema21 * 100.0
                )

                if trend_distance_pct >= 0.10:
                    score += 5
    else:
        score += 20

    if ema9 > ema21:
        score += 15

    if close > vwap:
        score += 10

        if vwap > 0:
            vwap_distance_pct = (
                (close - vwap) / vwap * 100.0
            )

            if vwap_distance_pct <= 1.0:
                score += 5

            if vwap_distance_pct <= 0.5:
                score += 5

    if 55.0 <= rsi <= 65.0:
        score += 15
    elif 52.0 <= rsi <= 68.0:
        score += 10

    if vol_ratio >= 1.5:
        score += 15
    elif vol_ratio >= 1.2:
        score += 10

    if "open" in row.index and pd.notna(row["open"]):
        if float(row["close"]) > float(row["open"]):
            score += 5

    if "body_ratio" in row.index and pd.notna(row["body_ratio"]):
        if float(row["body_ratio"]) >= 0.5:
            score += 5

    return min(score, 100)


def generate_signal(
    df: pd.DataFrame,
    symbol: str = "TEST",
    settings: Settings | None = None,
) -> Signal | None:
    """Generate a Strategy #002 BUY signal."""

    settings = settings or load_settings()
    strategy_settings = settings.strategy

    x = add_indicators(
        df,
        strategy_settings,
    )

    if x.empty:
        return None

    required_columns = [
        "close",
        "low",
        "vwap",
        "ema9",
        "ema21",
        "rsi",
        "vol_ratio",
        "prior_high",
    ]

    if not all(column in x.columns for column in required_columns):
        return None

    r = x.iloc[-1]

    if any(pd.isna(r[column]) for column in required_columns):
        return None

    close = float(r["close"])
    low = float(r["low"])
    ema9 = float(r["ema9"])
    ema21 = float(r["ema21"])
    vwap = float(r["vwap"])
    rsi = float(r["rsi"])
    vol_ratio = float(r["vol_ratio"])

    # ---------------------------------------------------------
    # 1. Broader trend
    # ---------------------------------------------------------
    if "ema50" in x.columns and pd.notna(r["ema50"]):
        broader_trend_condition = (
            ema21 > float(r["ema50"])
        )
    else:
        # Compatibility for compact mocked indicator frames.
        broader_trend_condition = True

    # ---------------------------------------------------------
    # 2. Short-term trend
    # ---------------------------------------------------------
    trend_condition = ema9 > ema21

    # ---------------------------------------------------------
    # 3. Price above VWAP
    # ---------------------------------------------------------
    vwap_condition = close > vwap

    # ---------------------------------------------------------
    # 4. Pullback toward VWAP / EMA21
    #
    # The current row contains the previous candle's values.
    # ---------------------------------------------------------
    if all(
        column in x.columns
        for column in (
            "prev_low",
            "prev_close",
            "prev_ema21",
            "prev_vwap",
        )
    ):
        previous_low = r["prev_low"]
        previous_close = r["prev_close"]
        previous_ema21 = r["prev_ema21"]
        previous_vwap = r["prev_vwap"]

        if all(
            pd.notna(value)
            for value in (
                previous_low,
                previous_close,
                previous_ema21,
                previous_vwap,
            )
        ):
            previous_low = float(previous_low)
            previous_close = float(previous_close)
            previous_ema21 = float(previous_ema21)
            previous_vwap = float(previous_vwap)

            support_reference = max(
                previous_ema21,
                previous_vwap,
            )

            pullback_condition = (
                previous_low <= support_reference * 1.003
                and previous_close >= previous_vwap * 0.995
            )
        else:
            pullback_condition = True
    else:
        # Compact unit-test frames may not contain previous-candle
        # fields. The other signal conditions remain testable.
        pullback_condition = True

    # ---------------------------------------------------------
    # 5. Bullish recovery
    # ---------------------------------------------------------
    if "open" in x.columns and pd.notna(r["open"]):
        recovery_condition = (
            close > float(r["open"])
            and close > ema21
        )
    else:
        recovery_condition = close > ema21

    if (
        "prev_close" in x.columns
        and pd.notna(r["prev_close"])
    ):
        recovery_condition = (
            recovery_condition
            and close > float(r["prev_close"])
        )

    # ---------------------------------------------------------
    # 6. RSI
    # ---------------------------------------------------------
    rsi_condition = 52.0 <= rsi <= 68.0

    # ---------------------------------------------------------
    # 7. Volume
    # ---------------------------------------------------------
    volume_condition = vol_ratio >= 1.2

    conditions = [
        broader_trend_condition,
        trend_condition,
        vwap_condition,
        pullback_condition,
        recovery_condition,
        rsi_condition,
        volume_condition,
    ]

    if not all(conditions):
        return None

    # ---------------------------------------------------------
    # Stop
    # ---------------------------------------------------------
    stop = low

    if (
        "prev_low" in x.columns
        and pd.notna(r["prev_low"])
    ):
        stop = min(
            stop,
            float(r["prev_low"]),
        )

    risk = close - stop

    if risk <= 0:
        return None

    # Reject abnormally wide stop structures.
    stop_distance_pct = (
        risk / close * 100.0
        if close > 0
        else float("inf")
    )

    if stop_distance_pct > 1.5:
        return None

    # ---------------------------------------------------------
    # Target
    # ---------------------------------------------------------
    target = close + (
        settings.risk_reward_min * risk
    )

    # ---------------------------------------------------------
    # Signal quality score
    # ---------------------------------------------------------
    score = _calculate_signal_score(r)

    # ---------------------------------------------------------
    # Timestamp
    # ---------------------------------------------------------
    if "timestamp" in x.columns:
        signal_timestamp = r["timestamp"]
    else:
        signal_timestamp = x.index[-1]

    return Signal(
        symbol=symbol,
        timestamp=signal_timestamp,
        action="BUY",
        price=close,
        stop=stop,
        target=target,
        score=score,
    )