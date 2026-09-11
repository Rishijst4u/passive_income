import numpy as np
import pandas as pd

def make_sample_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01 09:15", periods=n, freq="5min")
    returns = rng.normal(0.00005, 0.002, n)
    close = 100 * np.exp(np.cumsum(returns))
    spread = rng.uniform(0.05, 0.5, n)
    high = close + spread
    low = close - spread
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.integers(1000, 10000, n).astype(float)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume
    }, index=idx)
