"""Provider-neutral NSE intraday market-data ingestion and validation."""

from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Protocol

import pandas as pd


ASIA_KOLKATA = "Asia/Kolkata"
NSE_SESSION_START = time(9, 15)
NSE_SESSION_END = time(15, 30)
FIVE_MINUTES_NS = 5 * 60 * 1_000_000_000
OHLCV_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


class MarketDataProvider(Protocol):
    name: str

    def download(
        self, symbol: str, start: str | datetime | date, end: str | datetime | date,
        interval: str,
    ) -> pd.DataFrame: ...


class ProviderError(RuntimeError):
    """Raised when a provider cannot return requested market data."""


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    duplicate_timestamps: int
    missing_values: dict[str, int]
    invalid_ohlc_relationships: int
    negative_volume: int
    out_of_session_rows: int
    timestamp_ordered: bool
    timezone: str | None
    detected_gaps: int
    first_timestamp: pd.Timestamp | None
    last_timestamp: pd.Timestamp | None

    @property
    def is_valid(self) -> bool:
        return (
            self.rows > 0
            and self.duplicate_timestamps == 0
            and not any(self.missing_values.values())
            and self.invalid_ohlc_relationships == 0
            and self.negative_volume == 0
            and self.out_of_session_rows == 0
            and self.timestamp_ordered
            and self.timezone == ASIA_KOLKATA
            and self.detected_gaps == 0
        )


class DataValidationError(ValueError):
    """Raised when normalized OHLCV data fails validation without alteration."""

    def __init__(self, report: DataQualityReport):
        self.report = report
        super().__init__(f"Market-data validation failed: {report}")


class YFinanceProvider:
    """Prototype-only Yahoo Finance adapter for a small, recent NSE data sample."""

    name = "yfinance"

    def download(self, symbol, start, end, interval) -> pd.DataFrame:
        import yfinance as yf

        ticker = symbol.upper()
        if not ticker.endswith(".NS"):
            ticker = f"{ticker}.NS"
        frame = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
            multi_level_index=False,
        )
        if frame.empty:
            raise ProviderError(
                f"{self.name} returned no data for {ticker} ({start} to {end}, {interval})"
            )
        return frame


def canonical_symbol(symbol: str) -> str:
    return symbol.upper().removesuffix(".NS")


def normalize_ohlcv(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return provider output in the project OHLCV schema without repairing data."""
    if isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("Multi-symbol provider output is not supported for symbol ingestion")

    out = frame.copy()
    lower_columns = {str(column).lower(): column for column in out.columns}
    if "timestamp" in lower_columns:
        timestamp = out.pop(lower_columns["timestamp"])
    else:
        timestamp = pd.Series(out.index, index=out.index)

    required = ("open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in lower_columns]
    if missing:
        raise ValueError(f"Provider output is missing required columns: {', '.join(missing)}")

    normalized = pd.DataFrame(
        {"timestamp": pd.to_datetime(timestamp, errors="coerce").to_numpy()}
    )
    for column in required:
        normalized[column] = pd.to_numeric(out[lower_columns[column]], errors="coerce").to_numpy()

    timestamps = normalized["timestamp"]
    if timestamps.dt.tz is None:
        normalized["timestamp"] = timestamps.dt.tz_localize(ASIA_KOLKATA)
    else:
        normalized["timestamp"] = timestamps.dt.tz_convert(ASIA_KOLKATA)
    normalized.insert(1, "symbol", canonical_symbol(symbol))
    return normalized.loc[:, OHLCV_COLUMNS]


def build_quality_report(frame: pd.DataFrame, interval: str = "5m") -> DataQualityReport:
    if interval != "5m":
        raise ValueError("Only 5m OHLCV validation is supported in this milestone")
    missing_columns = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Normalized data is missing columns: {', '.join(missing_columns)}")

    timestamps = frame["timestamp"]
    timezone = str(timestamps.dt.tz) if isinstance(timestamps.dtype, pd.DatetimeTZDtype) else None
    missing_values = {column: int(frame[column].isna().sum()) for column in OHLCV_COLUMNS}
    duplicate_timestamps = int(timestamps.duplicated().sum())
    timestamp_ordered = bool(timestamps.is_monotonic_increasing)
    invalid_ohlc = (
        (frame["high"] < frame[["open", "low", "close"]].max(axis=1))
        | (frame["low"] > frame[["open", "high", "close"]].min(axis=1))
    )
    negative_volume = int((frame["volume"] < 0).sum())

    local_time = timestamps.dt.time
    out_of_session = (
        (timestamps.dt.dayofweek >= 5)
        | (local_time < NSE_SESSION_START)
        | (local_time > NSE_SESSION_END)
    )
    detected_gaps = sum(
        current.date() == previous.date()
        and pd.Timestamp(current).value - pd.Timestamp(previous).value > FIVE_MINUTES_NS
        for previous, current in zip(timestamps.iloc[:-1], timestamps.iloc[1:])
        if not pd.isna(previous) and not pd.isna(current)
    )

    return DataQualityReport(
        rows=len(frame),
        duplicate_timestamps=duplicate_timestamps,
        missing_values=missing_values,
        invalid_ohlc_relationships=int(invalid_ohlc.sum()),
        negative_volume=negative_volume,
        out_of_session_rows=int(out_of_session.sum()),
        timestamp_ordered=timestamp_ordered,
        timezone=timezone,
        detected_gaps=detected_gaps,
        first_timestamp=timestamps.iloc[0] if len(frame) else None,
        last_timestamp=timestamps.iloc[-1] if len(frame) else None,
    )


def validate_ohlcv(frame: pd.DataFrame, interval: str = "5m") -> DataQualityReport:
    report = build_quality_report(frame, interval)
    if not report.is_valid:
        raise DataValidationError(report)
    return report


@dataclass(frozen=True)
class IngestionResult:
    raw_path: Path
    processed_path: Path
    report: DataQualityReport


def ingest_market_data(
    provider: MarketDataProvider,
    symbol: str,
    start: str | datetime | date,
    end: str | datetime | date,
    interval: str = "5m",
    data_root: Path = DATA_ROOT,
) -> IngestionResult:
    """Download, retain raw data, validate normalized data, then write Parquet."""
    if interval != "5m":
        raise ValueError("Only 5m ingestion is supported in this milestone")

    source_symbol = canonical_symbol(symbol)
    file_name = f"{source_symbol}_{pd.Timestamp(start).date()}_{pd.Timestamp(end).date()}_{interval}.parquet"
    raw_path = data_root / "raw" / provider.name / source_symbol / interval / file_name
    processed_path = data_root / "processed" / source_symbol / interval / file_name

    raw = provider.download(symbol, start, end, interval)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(raw_path, engine="pyarrow", index=True)

    normalized = normalize_ohlcv(raw, source_symbol)
    report = validate_ohlcv(normalized, interval)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(processed_path, engine="pyarrow", index=False)
    return IngestionResult(raw_path, processed_path, report)
