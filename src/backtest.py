from dataclasses import dataclass
import pandas as pd
from .config import load_settings
from .strategy import generate_signal

@dataclass
class Trade:
    symbol: str
    entry_time: object
    entry: float
    stop: float
    target: float
    exit_time: object
    exit: float
    pnl: float
    reason: str

def run_simple_backtest(df: pd.DataFrame, symbol="TEST") -> list[Trade]:
    # Educational baseline: evaluates one signal at a time and exits at stop/target.
    # Realistic slippage/cost model will be added before the 100-trade experiment.
    trades = []
    settings = load_settings()
    i = 30
    while i < len(df) - 2:
        window = df.iloc[:i+1]
        sig = generate_signal(window, symbol, settings)
        if sig:
            for j in range(i + 1, len(df)):
                bar = df.iloc[j]
                if bar.low <= sig.stop:
                    exit_price, reason = sig.stop, "STOP"
                elif bar.high >= sig.target:
                    exit_price, reason = sig.target, "TARGET"
                else:
                    continue
                trades.append(Trade(
                    symbol, sig.timestamp, sig.price, sig.stop, sig.target,
                    df.index[j], float(exit_price),
                    float(exit_price - sig.price), reason
                ))
                i = j
                break
        i += 1
    return trades
