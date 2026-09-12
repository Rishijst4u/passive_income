"""DhanHQ historical intraday market-data provider."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .dhan_instruments import DhanInstrumentMaster
from .market_data import ASIA_KOLKATA, MarketDataProvider

DHAN_INTRADAY_URL = "https://api.dhan.co/v2/charts/intraday"
SUPPORTED_INTERVALS = frozenset({"1", "5", "15", "25", "60"})
MAX_INTRADAY_REQUEST_DAYS = 90


class DhanProviderError(RuntimeError):
    """Raised when Dhan historical-data retrieval fails."""


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"Invalid date: {value!r}")
    return parsed.date()


def _request_payload(
    security_id: str,
    exchange_segment: str,
    instrument: str,
    interval: str,
    start: date,
    end: date,
) -> dict[str, Any]:
    return {
        "securityId": security_id,
        "exchangeSegment": exchange_segment,
        "instrument": instrument,
        "interval": interval,
        "oi": False,
        "fromDate": f"{start.isoformat()} 00:00:00",
        "toDate": f"{end.isoformat()} 00:00:00",
    }


def _response_to_dataframe(
    response: dict[str, Any],
    symbol: str,
) -> pd.DataFrame:
    required = ("open", "high", "low", "close", "volume", "timestamp")
    missing = [key for key in required if key not in response]
    if missing:
        raise DhanProviderError(
            "Dhan intraday response is missing fields: " + ", ".join(missing)
        )

    lengths = {key: len(response[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise DhanProviderError(
            f"Dhan intraday response arrays have inconsistent lengths: {lengths}"
        )

    if not lengths["timestamp"]:
        raise DhanProviderError("Dhan returned no historical candles.")

    try:
        timestamps = pd.to_datetime(
            pd.Series(response["timestamp"]),
            unit="s",
            utc=True,
            errors="raise",
        ).dt.tz_convert(ASIA_KOLKATA)
    except Exception as exc:
        raise DhanProviderError(
            "Unable to parse Dhan epoch timestamps."
        ) from exc

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": str(symbol).upper().removesuffix(".NS"),
            "open": pd.to_numeric(response["open"], errors="coerce"),
            "high": pd.to_numeric(response["high"], errors="coerce"),
            "low": pd.to_numeric(response["low"], errors="coerce"),
            "close": pd.to_numeric(response["close"], errors="coerce"),
            "volume": pd.to_numeric(response["volume"], errors="coerce"),
        }
    )

    return frame[
        ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    ]


class DhanProvider(MarketDataProvider):
    """Provider-neutral adapter for Dhan intraday historical candles."""

    name = "dhan"

    def __init__(
        self,
        access_token: str,
        instrument_master: DhanInstrumentMaster,
        *,
        endpoint: str = DHAN_INTRADAY_URL,
        timeout: float = 30.0,
        opener: Callable[..., object] | None = None,
    ):
        if not access_token or not access_token.strip():
            raise ValueError("Dhan access token must not be empty.")
        self.access_token = access_token.strip()
        self.instrument_master = instrument_master
        self.endpoint = endpoint
        self.timeout = timeout
        self._opener = opener or urlopen

    def _fetch_chunk(
        self,
        security_id: str,
        exchange_segment: str,
        instrument: str,
        interval: str,
        start: date,
        end: date,
        symbol: str,
    ) -> pd.DataFrame:
        payload = _request_payload(
            security_id,
            exchange_segment,
            instrument,
            interval,
            start,
            end,
        )
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "access-token": self.access_token,
                "User-Agent": "passive-income-trading-research/1.0",
            },
            method="POST",
        )

        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise DhanProviderError(
                f"Dhan historical-data request failed with HTTP {exc.code}: {body}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise DhanProviderError(
                f"Dhan historical-data request failed: {exc}"
            ) from exc

        try:
            response_json = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DhanProviderError(
                "Dhan returned a non-JSON historical-data response."
            ) from exc

        if not isinstance(response_json, dict):
            raise DhanProviderError("Dhan historical response must be a JSON object.")

        if response_json.get("errorCode") or response_json.get("errorMessage"):
            raise DhanProviderError(
                f"Dhan API error: {response_json.get('errorCode', '')} "
                f"{response_json.get('errorMessage', '')}".strip()
            )

        return _response_to_dataframe(response_json, symbol)

    def download(
        self,
        symbol: str,
        start: str | datetime | date,
        end: str | datetime | date,
        interval: str,
    ) -> pd.DataFrame:
        interval_value = str(interval).lower().removesuffix("m")
        if interval_value not in SUPPORTED_INTERVALS:
            raise ValueError(
                f"Dhan intraday interval must be one of "
                f"{sorted(SUPPORTED_INTERVALS)} minutes."
            )

        start_date = _as_date(start)
        end_date = _as_date(end)
        if end_date <= start_date:
            raise ValueError("Dhan download end date must be after start date.")

        instrument = self.instrument_master.resolve(symbol)

        chunks: list[pd.DataFrame] = []
        chunk_start = start_date
        while chunk_start < end_date:
            chunk_end = min(
                chunk_start + timedelta(days=MAX_INTRADAY_REQUEST_DAYS),
                end_date,
            )
            chunks.append(
                self._fetch_chunk(
                    instrument.security_id,
                    instrument.exchange_segment,
                    instrument.instrument,
                    interval_value,
                    chunk_start,
                    chunk_end,
                    instrument.symbol,
                )
            )
            chunk_start = chunk_end

        if not chunks:
            raise DhanProviderError("No Dhan historical-data chunks were requested.")

        frame = pd.concat(chunks, ignore_index=True)
        if frame.empty:
            raise DhanProviderError("Dhan returned no historical candles.")

        # Chunk boundaries are intentionally de-duplicated because provider
        # implementations can differ on whether toDate is inclusive.
        frame = (
            frame.drop_duplicates(subset=["timestamp"], keep="first")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return frame
