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

    # Session completeness diagnostics.
    missing_session_bars: int = 0
    trailing_missing_session_bars: int = 0
    session_status: str = "INVALID"

    @property
    def is_valid(self) -> bool:
        """Strict production-quality validation."""

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
            and self.session_status == "FULL_SESSION"
        )

    @property
    def is_development_usable(self) -> bool:
        """
        Whether the dataset is safe to use for development/backtesting.

        A partial session is acceptable only when all missing session
        candles are trailing candles at the end of the NSE session.
        Internal gaps are never accepted.
        """

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
            and self.missing_session_bars
            == self.trailing_missing_session_bars
            and self.session_status
            in {"FULL_SESSION", "PARTIAL_SESSION"}
        )

    @property
    def production_qualified(self) -> bool:
        """Whether the dataset is suitable for production-grade research."""

        return self.is_valid


class DataValidationError(ValueError):
    """Raised when normalized OHLCV data fails validation."""

    def __init__(self, report: DataQualityReport):
        self.report = report
        super().__init__(
            f"Market-data validation failed: {report}"
        )


class YFinanceProvider:
    """Prototype-only Yahoo Finance adapter."""

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
    """Return provider output in the project OHLCV schema."""

    if isinstance(frame.columns, pd.MultiIndex):
        raise ValueError(
            "Multi-symbol provider output is not supported "
            "for symbol ingestion"
        )

    out = frame.copy()

    lower_columns = {
        str(column).lower(): column
        for column in out.columns
    }

    if "timestamp" in lower_columns:
        timestamp = out.pop(
            lower_columns["timestamp"]
        )
    else:
        timestamp = pd.Series(
            out.index,
            index=out.index,
        )

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
        normalized["timestamp"] = (
            timestamps.dt.tz_localize(ASIA_KOLKATA)
        )
    else:
        normalized["timestamp"] = (
            timestamps.dt.tz_convert(ASIA_KOLKATA)
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

    final_bar_timestamp = (
        pd.Timestamp.combine(
            trading_date,
            NSE_SESSION_END,
        ).tz_localize(ASIA_KOLKATA)
        - pd.Timedelta(minutes=5)
    )

    return pd.date_range(
        session_start,
        final_bar_timestamp,
        freq="5min",
    )


def _session_gap_details(
    timestamps: pd.Series,
    calendar: TradingCalendar,
) -> tuple[int, int]:
    """
    Return:

    (total missing session bars, trailing missing session bars)

    Trailing bars are missing only after the final observed candle
    for that trading date. Any internal missing candle is therefore
    counted in total missing bars but not trailing missing bars.
    """

    valid_timestamps = timestamps.dropna()

    if valid_timestamps.empty:
        return 0, 0

    missing_bars = 0
    trailing_missing_bars = 0

    first_date = valid_timestamps.min().date()
    last_date = valid_timestamps.max().date()

    for trading_date in pd.date_range(
        first_date,
        last_date,
        freq="D",
    ).date:

        if not calendar.is_trading_day(trading_date):
            continue

        expected = _expected_session_timestamps(
            trading_date
        )

        observed = set(
            valid_timestamps[
                valid_timestamps.dt.date == trading_date
            ]
        )

        missing = [
            timestamp
            for timestamp in expected
            if timestamp not in observed
        ]

        missing_bars += len(missing)

        if not observed:
            continue

        last_observed = max(observed)

        trailing_missing_bars += sum(
            timestamp > last_observed
            for timestamp in missing
        )

    return (
        missing_bars,
        trailing_missing_bars,
    )


def _count_missing_session_bars(
    timestamps: pd.Series,
    calendar: TradingCalendar,
) -> int:
    """Count missing 5-minute bars on NSE trading days."""

    missing_bars, _ = _session_gap_details(
        timestamps,
        calendar,
    )

    return missing_bars


def _count_unexpected_trading_dates(
    timestamps: pd.Series,
    calendar: TradingCalendar,
) -> int:
    """Count distinct dates that are not NSE trading days."""

    valid_timestamps = timestamps.dropna()

    if valid_timestamps.empty:
        return 0

    unique_dates = set(
        valid_timestamps.dt.date
    )

    return sum(
        not calendar.is_trading_day(trading_date)
        for trading_date in unique_dates
    )


def _session_status(
    missing_session_bars: int,
    trailing_missing_session_bars: int,
) -> str:
    """Classify session completeness."""

    if missing_session_bars == 0:
        return "FULL_SESSION"

    if (
        missing_session_bars
        == trailing_missing_session_bars
    ):
        return "PARTIAL_SESSION"

    return "INVALID"


def build_quality_report(
    frame: pd.DataFrame,
    interval: str = "5m",
    calendar: TradingCalendar = DEFAULT_NSE_TRADING_CALENDAR,
) -> DataQualityReport:

    if interval != "5m":
        raise ValueError(
            "Only 5m OHLCV validation is supported "
            "in this milestone"
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

    out_of_session = (
        valid_timestamp_mask
        & (
            (local_time < NSE_SESSION_START)
            |
            (local_time >= NSE_SESSION_END)
        )
    )

    unexpected_trading_dates = (
        _count_unexpected_trading_dates(
            timestamps,
            calendar,
        )
    )

    (
        missing_session_bars,
        trailing_missing_session_bars,
    ) = _session_gap_details(
        timestamps,
        calendar,
    )

    session_status = _session_status(
        missing_session_bars,
        trailing_missing_session_bars,
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
        unexpected_trading_dates=(
            unexpected_trading_dates
        ),
        timestamp_ordered=timestamp_ordered,
        timezone=timezone,
        detected_gaps=missing_session_bars,
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
        missing_session_bars=missing_session_bars,
        trailing_missing_session_bars=(
            trailing_missing_session_bars
        ),
        session_status=session_status,
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


def _remove_non_trading_dates(
    frame: pd.DataFrame,
    calendar: TradingCalendar,
) -> pd.DataFrame:
    """
    Remove rows that fall on dates when NSE was closed.

    Raw provider data is never modified. This only cleans the
    normalized processed dataset.
    """

    trading_date_mask = frame["timestamp"].dt.date.map(
        calendar.is_trading_day
    )

    return (
        frame.loc[trading_date_mask]
        .reset_index(drop=True)
    )


def ingest_market_data(
    provider: MarketDataProvider,
    symbol: str,
    start: str | datetime | date,
    end: str | datetime | date,
    interval: str = "5m",
    data_root: Path = DATA_ROOT,
    calendar: TradingCalendar = DEFAULT_NSE_TRADING_CALENDAR,
) -> IngestionResult:
    """
    Download, retain raw data, clean non-trading dates,
    validate normalized data, then write Parquet.

    Development datasets may be PARTIAL_SESSION when the only
    missing candles are trailing candles at the end of the session.
    """

    if interval != "5m":
        raise ValueError(
            "Only 5m ingestion is supported "
            "in this milestone"
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

    # Dhan has returned candles on an NSE holiday in our
    # qualification test. Preserve those rows in raw data,
    # but exclude them from the processed research dataset.
    normalized = _remove_non_trading_dates(
        normalized,
        calendar,
    )

    report = build_quality_report(
        normalized,
        interval,
        calendar,
    )

    if not report.is_development_usable:
        raise DataValidationError(report)

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