from src.sample_data import make_sample_data
from src.strategy import add_indicators

def test_indicators_exist():
    df = add_indicators(make_sample_data(100))
    for col in ["ema9", "ema21", "rsi", "vwap", "vol_ratio", "prior_high"]:
        assert col in df.columns
