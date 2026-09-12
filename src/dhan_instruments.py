"""DhanHQ instrument-master loading and NSE equity resolution."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

DHAN_INSTRUMENT_MASTER_URL = (
    "https://images.dhan.co/api-data/api-scrip-master.csv"
)


class DhanInstrumentError(RuntimeError):
    """Raised when Dhan instrument-master data cannot be loaded or resolved."""


@dataclass(frozen=True)
class DhanInstrument:
    """Canonical instrument identity required by Dhan chart APIs."""

    symbol: str
    security_id: str
    exchange_segment: str = "NSE_EQ"
    instrument: str = "EQUITY"


def _canonical_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().removesuffix(".NS")


def _column_by_alias(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    normalized = {
        str(column).strip().upper(): column
        for column in frame.columns
    }
    for alias in aliases:
        if alias.upper() in normalized:
            return normalized[alias.upper()]
    return None


def normalize_instrument_master(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize Dhan compact/detailed instrument-master columns.

    The returned frame contains only fields needed by this project.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Instrument master must be a pandas DataFrame.")
    if frame.empty:
        raise DhanInstrumentError("Dhan instrument master is empty.")

    security_col = _column_by_alias(
        frame,
        (
            "SEM_SMST_SECURITY_ID",
            "SECURITY_ID",
            "SMST_SECURITY_ID",
        ),
    )
    exchange_col = _column_by_alias(
        frame,
        ("SEM_EXM_EXCH_ID", "EXCH_ID", "EXCHANGE"),
    )
    segment_col = _column_by_alias(
        frame,
        ("SEM_SEGMENT", "SEGMENT", "EXCHANGE_SEGMENT"),
    )
    instrument_col = _column_by_alias(
        frame,
        ("SEM_INSTRUMENT_NAME", "INSTRUMENT", "INSTRUMENT_TYPE"),
    )
    symbol_col = _column_by_alias(
        frame,
        ("SM_SYMBOL_NAME", "SYMBOL_NAME", "SYMBOL"),
    )
    trading_symbol_col = _column_by_alias(
        frame,
        ("SEM_TRADING_SYMBOL", "TRADING_SYMBOL"),
    )
    custom_symbol_col = _column_by_alias(
        frame,
        ("SEM_CUSTOM_SYMBOL", "CUSTOM_SYMBOL", "DISPLAY_NAME"),
    )
    isin_col = _column_by_alias(frame, ("ISIN",))

    missing = [
        name
        for name, column in (
            ("security_id", security_col),
            ("exchange", exchange_col),
            ("segment", segment_col),
            ("instrument", instrument_col),
        )
        if column is None
    ]
    if missing:
        raise DhanInstrumentError(
            "Dhan instrument master is missing required columns: "
            + ", ".join(missing)
        )

    out = pd.DataFrame(
        {
            "security_id": frame[security_col].astype("string").str.strip(),
            "exchange": frame[exchange_col].astype("string").str.strip().str.upper(),
            "segment": frame[segment_col].astype("string").str.strip().str.upper(),
            "instrument": frame[instrument_col].astype("string").str.strip().str.upper(),
        }
    )

    out["symbol_name"] = (
        frame[symbol_col].astype("string").str.strip().str.upper()
        if symbol_col is not None
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    out["trading_symbol"] = (
        frame[trading_symbol_col].astype("string").str.strip().str.upper()
        if trading_symbol_col is not None
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    out["custom_symbol"] = (
        frame[custom_symbol_col].astype("string").str.strip().str.upper()
        if custom_symbol_col is not None
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )
    out["isin"] = (
        frame[isin_col].astype("string").str.strip().str.upper()
        if isin_col is not None
        else pd.Series(pd.NA, index=frame.index, dtype="string")
    )

    return out.reset_index(drop=True)


class DhanInstrumentMaster:
    """Resolver for Dhan Security IDs from an instrument-master DataFrame."""

    def __init__(self, frame: pd.DataFrame):
        self._frame = normalize_instrument_master(frame)

    @classmethod
    def from_csv(cls, path: str | Path) -> "DhanInstrumentMaster":
        return cls(pd.read_csv(path, low_memory=False))

    @classmethod
    def from_url(
        cls,
        url: str = DHAN_INSTRUMENT_MASTER_URL,
        timeout: float = 30.0,
        opener: Callable[..., object] | None = None,
    ) -> "DhanInstrumentMaster":
        request = Request(
            url,
            headers={"User-Agent": "passive-income-trading-research/1.0"},
        )
        open_fn = opener or urlopen
        try:
            with open_fn(request, timeout=timeout) as response:
                payload = response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise DhanInstrumentError(
                f"Unable to download Dhan instrument master: {exc}"
            ) from exc

        try:
            frame = pd.read_csv(BytesIO(payload), low_memory=False)
        except Exception as exc:
            raise DhanInstrumentError(
                f"Unable to parse Dhan instrument master CSV: {exc}"
            ) from exc
        return cls(frame)

    def resolve(self, symbol: str) -> DhanInstrument:
        canonical = _canonical_symbol(symbol)
        if not canonical:
            raise DhanInstrumentError("Symbol must not be empty.")

        frame = self._frame
        eligible = frame[
            frame["exchange"].eq("NSE")
            & frame["segment"].isin(["E", "EQUITY"])
            & frame["instrument"].isin(["EQUITY", "EQUITY SHARES"])
        ].copy()

        symbol_columns = ("symbol_name", "trading_symbol", "custom_symbol")
        matches = pd.Series(False, index=eligible.index)
        for column in symbol_columns:
            values = eligible[column].fillna("").astype(str)
            values = values.str.removesuffix(".NS")
            matches |= values.eq(canonical)

        candidates = eligible.loc[matches]
        if candidates.empty:
            raise DhanInstrumentError(
                f"NSE equity instrument not found for symbol '{canonical}'."
            )

        if len(candidates) > 1:
            # Prefer an exact trading-symbol match when multiple rows exist.
            trading_matches = candidates[
                candidates["trading_symbol"]
                .fillna("")
                .astype(str)
                .str.removesuffix(".NS")
                .eq(canonical)
            ]
            if len(trading_matches) == 1:
                candidates = trading_matches
            else:
                raise DhanInstrumentError(
                    f"Multiple NSE equity instruments found for symbol '{canonical}'."
                )

        row = candidates.iloc[0]
        return DhanInstrument(
            symbol=canonical,
            security_id=str(row["security_id"]),
        )
