"""Diagnostic tool for investigating Dhan intraday historical candles."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from dotenv import load_dotenv

from src.dhan_instruments import DhanInstrumentMaster


DHAN_INTRADAY_URL = "https://api.dhan.co/v2/charts/intraday"
IST = "Asia/Kolkata"


def fetch_dhan_data(
    *,
    security_id: str,
    interval: str,
    from_date: str,
    to_date: str,
) -> dict:
    """Fetch raw Dhan historical intraday response."""

    load_dotenv()

    access_token = os.getenv(
        "DHAN_ACCESS_TOKEN",
        "",
    ).strip()

    if not access_token:
        raise RuntimeError(
            "DHAN_ACCESS_TOKEN is missing from .env/environment."
        )

    payload = {
        "securityId": str(security_id),
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": str(interval),
        "oi": False,
        "fromDate": from_date,
        "toDate": to_date,
    }

    print("Request payload:")
    print(json.dumps(payload, indent=2))
    print()

    request = Request(
        DHAN_INTRADAY_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": access_token,
            "User-Agent": (
                "ai-trading-platform-experiment-001/1.0"
            ),
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)

    except HTTPError as exc:
        error_body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        print(f"HTTP status: {exc.code}")
        print("Dhan error response:")
        print(error_body)
        print()

        raise

    except URLError as exc:
        raise RuntimeError(
            f"Unable to reach Dhan API: {exc}"
        ) from exc


def response_to_dataframe(payload: dict) -> pd.DataFrame:
    """Convert Dhan OHLCV response to a dataframe."""

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timestamp",
    ]

    missing = [
        key for key in required
        if key not in payload
    ]

    if missing:
        raise RuntimeError(
            "Dhan response is missing fields: "
            f"{missing}\nResponse keys: "
            f"{list(payload.keys())}"
        )

    lengths = {
        key: len(payload[key])
        for key in required
    }

    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            f"Dhan response arrays have different lengths: "
            f"{lengths}"
        )

    frame = pd.DataFrame(
        {
            "open": payload["open"],
            "high": payload["high"],
            "low": payload["low"],
            "close": payload["close"],
            "volume": payload["volume"],
            "timestamp": payload["timestamp"],
        }
    )

    # Dhan v2 returns Unix epoch seconds.
    frame["timestamp"] = (
        pd.to_datetime(
            frame["timestamp"],
            unit="s",
            utc=True,
        )
        .dt.tz_convert(IST)
    )

    return frame.sort_values(
        "timestamp"
    ).reset_index(drop=True)


def main() -> None:
    master_path = Path(
        "data/raw/dhan/instrument_master/"
        "api-scrip-master.csv"
    )

    master = DhanInstrumentMaster.from_csv(
        master_path
    )

    symbol = "HDFCBANK"
    instrument = master.resolve(symbol)

    print(
        f"Symbol: {symbol}"
    )
    print(
        f"Security ID: {instrument.security_id}"
    )
    print()

    # Current diagnostic:
    # deliberately request only the final 30 minutes.
    from_date = "2026-08-31 15:00:00"
    to_date = "2026-08-31 15:30:00"
    interval = "1"

    payload = fetch_dhan_data(
        security_id=instrument.security_id,
        interval=interval,
        from_date=from_date,
        to_date=to_date,
    )

    frame = response_to_dataframe(payload)

    print(
        f"Rows: {len(frame)}"
    )

    if frame.empty:
        print("No candles returned.")
        return

    print(
        "First timestamp:",
        frame["timestamp"].iloc[0],
    )
    print(
        "Last timestamp:",
        frame["timestamp"].iloc[-1],
    )

    print()
    print("Timestamps:")
    for timestamp in frame["timestamp"]:
        print(timestamp)


if __name__ == "__main__":
    main()