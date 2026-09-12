import pandas as pd
import pytest

from src.dhan_instruments import (
    DhanInstrumentError,
    DhanInstrumentMaster,
    normalize_instrument_master,
)


def make_master():
    return pd.DataFrame(
        {
            "SEM_SMST_SECURITY_ID": ["1333", "9999", "8888"],
            "SEM_EXM_EXCH_ID": ["NSE", "NSE", "BSE"],
            "SEM_SEGMENT": ["E", "D", "E"],
            "SEM_INSTRUMENT_NAME": ["EQUITY", "FUTIDX", "EQUITY"],
            "SM_SYMBOL_NAME": ["RELIANCE", "NIFTY", "RELIANCE"],
            "SEM_TRADING_SYMBOL": ["RELIANCE", "NIFTY-I", "RELIANCE"],
        }
    )


def test_normalize_compact_master():
    result = normalize_instrument_master(make_master())

    assert list(result.columns) == [
        "security_id",
        "exchange",
        "segment",
        "instrument",
        "symbol_name",
        "trading_symbol",
        "custom_symbol",
        "isin",
    ]
    assert result.loc[0, "security_id"] == "1333"


def test_resolve_nse_equity():
    master = DhanInstrumentMaster(make_master())

    instrument = master.resolve("RELIANCE.NS")

    assert instrument.symbol == "RELIANCE"
    assert instrument.security_id == "1333"
    assert instrument.exchange_segment == "NSE_EQ"
    assert instrument.instrument == "EQUITY"


def test_non_equity_and_other_exchange_are_ignored():
    master = DhanInstrumentMaster(make_master())

    with pytest.raises(DhanInstrumentError):
        master.resolve("NIFTY")


def test_missing_symbol_raises():
    master = DhanInstrumentMaster(make_master())

    with pytest.raises(DhanInstrumentError, match="not found"):
        master.resolve("INFY")


def test_required_columns_are_checked():
    frame = pd.DataFrame({"SEM_EXM_EXCH_ID": ["NSE"]})

    with pytest.raises(DhanInstrumentError, match="missing required columns"):
        normalize_instrument_master(frame)
