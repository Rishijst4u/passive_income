"""Provider-neutral NSE intraday market-data ingestion and validation."""

from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Protocol

import pandas as pd

from .trading_calendar import DEFAULT_NSE_TRADING_CALENDAR, TradingCalendar


ASIA_KOLKATA = "Asia/Kolkata"

# NSE regular equity trading session.
NSE_SESSION_START = time(9, 15)
NSE_SESSION_END = time(15, 30)

OHLCV_COLUMNS = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


class MarketDataProvider(Protocol):
    name: str

    def download(
        self,
        symbol: str,
        start: str | datetime | date,
        end: str | datetime | date,
        interval: str,
    ) -> pd.DataFrame:
        ...


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
    unexpected_trading_dates: int
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
            and self.unexpected_trading_dates == 0
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
                f"{self.name} returned no data for "
                f"{ticker} ({start} to {end}, {interval})"
            )

        return frame


def canonical_symbol(symbol: str) -> str:
    """Return canonical NSE symbol without the Yahoo .NS suffix."""
    return symbol.upper().removesuffix(".NS")


def normalize_ohlcv(
    frame: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """Return provider output in the project OHLCV schema without repairing data."""

    if isinstance(frame.columns, pd.MultiIndex):
        raise ValueError(
            "Multi-symbol provider output is not supported for symbol ingestion"
        )

    out = frame.copy()

    lower_columns = {
        str(column).lower(): column
        for column in out.columns
    }

    if "timestamp" in lower_columns:
        timestamp = out.pop(lower_columns["timestamp"])
    else:
        timestamp = pd.Series(out.index, index=out.index)

    required = (
        "open",
        "high",
        "low",
        "close",
        "volume",
    )

    missing = [
        column
        for column in required
        if column not in lower_columns
    ]

    if missing:
        raise ValueError(
            "Provider output is missing required columns: "
            + ", ".join(missing)
        )

    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                timestamp,
                errors="coerce",
            ).to_numpy()
        }
    )

    for column in required:
        normalized[column] = pd.to_numeric(
            out[lower_columns[column]],
            errors="coerce",
        ).to_numpy()

    timestamps = normalized["timestamp"]

    if timestamps.dt.tz is None:
        normalized["timestamp"] = timestamps.dt.tz_localize(
            ASIA_KOLKATA
        )
    else:
        normalized["timestamp"] = timestamps.dt.tz_convert(
            ASIA_KOLKATA
        )

    normalized.insert(
        1,
        "symbol",
        canonical_symbol(symbol),
    )

    return normalized.loc[:, OHLCV_COLUMNS]


def _expected_session_timestamps(
    trading_date: date,
) -> pd.DatetimeIndex:
    """
    Return expected 5-minute candle timestamps for an NSE session.

    NSE regular trading is 09:15-15:30. For OHLCV candles, timestamps
    represent the beginning of each five-minute interval, so the final
    expected candle timestamp is 15:25.
    """

    session_start = pd.Timestamp.combine(
        trading_date,
        NSE_SESSION_START,
    ).tz_localize(ASIA_KOLKATA)

    final_bar_timestamp = pd.Timestamp.combine(
        trading_date,
        NSE_SESSION_END,
    ).tz_localize(ASIA_KOLKATA) - pd.Timedelta(minutes=5)

    return pd.date_range(
        session_start,
        final_bar_timestamp,
        freq="5min",
    )


def _count_missing_session_bars(
    timestamps: pd.Series,
    calendar: TradingCalendar,
) -> int:
    """Count missing 5-minute bars on dates that are trading days."""

    valid_timestamps = timestamps.dropna()

    if valid_timestamps.empty:
        return 0

    observed = set(valid_timestamps)

    missing_bars = 0

    first_date = valid_timestamps.min().date()
    last_date = valid_timestamps.max().date()

    for trading_date in pd.date_range(
        first_date,
        last_date,
        freq="D",
    ).date:

        if not calendar.is_trading_day(trading_date):
            continue

        expected_bars = _expected_session_timestamps(
            trading_date
        )

        missing_bars += sum(
            expected not in observed
            for expected in expected_bars
        )

    return missing_bars


def _count_unexpected_trading_dates(
    timestamps: pd.Series,
    calendar: TradingCalendar,
) -> int:
    """
    Count distinct dates in the data that are not NSE trading days.

    A date is counted once regardless of how many bars it contains.
    """

    valid_timestamps = timestamps.dropna()

    if valid_timestamps.empty:
        return 0

    unique_dates = set(valid_timestamps.dt.date)

    return sum(
        not calendar.is_trading_day(trading_date)
        for trading_date in unique_dates
    )


def build_quality_report(
    frame: pd.DataFrame,
    interval: str = "5m",
    calendar: TradingCalendar = DEFAULT_NSE_TRADING_CALENDAR,
) -> DataQualityReport:

    if interval != "5m":
        raise ValueError(
            "Only 5m OHLCV validation is supported in this milestone"
        )

    missing_columns = [
        column
        for column in OHLCV_COLUMNS
        if column not in frame.columns
    ]

    if missing_columns:
        raise ValueError(
            "Normalized data is missing columns: "
            + ", ".join(missing_columns)
        )

    timestamps = frame["timestamp"]

    timezone = (
        str(timestamps.dt.tz)
        if isinstance(
            timestamps.dtype,
            pd.DatetimeTZDtype,
        )
        else None
    )

    missing_values = {
        column: int(frame[column].isna().sum())
        for column in OHLCV_COLUMNS
    }

    duplicate_timestamps = int(
        timestamps.duplicated().sum()
    )

    timestamp_ordered = bool(
        timestamps.is_monotonic_increasing
    )

    invalid_ohlc = (
        (
            frame["high"]
            < frame[
                ["open", "low", "close"]
            ].max(axis=1)
        )
        |
        (
            frame["low"]
            > frame[
                ["open", "high", "close"]
            ].min(axis=1)
        )
    )

    negative_volume = int(
        (frame["volume"] < 0).sum()
    )

    valid_timestamp_mask = timestamps.notna()

    local_time = timestamps.dt.time

    trading_day_mask = (
        timestamps.dt.date.map(
            lambda trading_date: (
                calendar.is_trading_day(trading_date)
                if pd.notna(trading_date)
                else False
            )
        )
    )

    out_of_session = (
        valid_timestamp_mask
        &
        (
            (local_time < NSE_SESSION_START)
            |
            (local_time >= NSE_SESSION_END)
        )
    )

    # Rows on exchange holidays/weekends are tracked separately.
    # They are not classified as a clock-time/session violation.
    unexpected_trading_dates = _count_unexpected_trading_dates(
        timestamps,
        calendar,
    )

    # A timestamp is considered out-of-session if it is outside
    # the regular NSE equity session clock.
    #
    # A timestamp on a holiday is reported separately through
    # unexpected_trading_dates.
    _ = trading_day_mask

    detected_gaps = _count_missing_session_bars(
        timestamps,
        calendar,
    )

    return DataQualityReport(
        rows=len(frame),
        duplicate_timestamps=duplicate_timestamps,
        missing_values=missing_values,
        invalid_ohlc_relationships=int(
            invalid_ohlc.sum()
        ),
        negative_volume=negative_volume,
        out_of_session_rows=int(
            out_of_session.sum()
        ),
        unexpected_trading_dates=unexpected_trading_dates,
        timestamp_ordered=timestamp_ordered,
        timezone=timezone,
        detected_gaps=detected_gaps,
        first_timestamp=(
            timestamps.iloc[0]
            if len(frame)
            else None
        ),
        last_timestamp=(
            timestamps.iloc[-1]
            if len(frame)
            else None
        ),
    )


def validate_ohlcv(
    frame: pd.DataFrame,
    interval: str = "5m",
    calendar: TradingCalendar = DEFAULT_NSE_TRADING_CALENDAR,
) -> DataQualityReport:

    report = build_quality_report(
        frame,
        interval,
        calendar,
    )

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
    calendar: TradingCalendar = DEFAULT_NSE_TRADING_CALENDAR,
) -> IngestionResult:
    """Download, retain raw data, validate normalized data, then write Parquet."""

    if interval != "5m":
        raise ValueError(
            "Only 5m ingestion is supported in this milestone"
        )

    source_symbol = canonical_symbol(symbol)

    file_name = (
        f"{source_symbol}_"
        f"{pd.Timestamp(start).date()}_"
        f"{pd.Timestamp(end).date()}_"
        f"{interval}.parquet"
    )

    raw_path = (
        data_root
        / "raw"
        / provider.name
        / source_symbol
        / interval
        / file_name
    )

    processed_path = (
        data_root
        / "processed"
        / source_symbol
        / interval
        / file_name
    )

    raw = provider.download(
        symbol,
        start,
        end,
        interval,
    )

    raw_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw.to_parquet(
        raw_path,
        engine="pyarrow",
        index=True,
    )

    normalized = normalize_ohlcv(
        raw,
        source_symbol,
    )

    report = validate_ohlcv(
        normalized,
        interval,
        calendar,
    )

    processed_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized.to_parquet(
        processed_path,
        engine="pyarrow",
        index=False,
    )

    return IngestionResult(
        raw_path,
        processed_path,
        report,
    )