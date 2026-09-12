"""Diagnostic tool for investigating Dhan intraday historical candles.

This is a manual diagnostic script, not a pytest test module.
"""

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


def fetch(
    *,
    security_id: str,
    interval: str,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    """Fetch one Dhan historical-data window."""

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
            "access-token": os.environ["DHAN_ACCESS_TOKEN"],
            "User-Agent": (
                "ai-trading-platform-experiment-001/1.0"
            ),
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            payload = json.loads(body)

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

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timestamp",
    ]

    missing = [
        key
        for key in required
        if key not in payload
    ]

    if missing:
        raise RuntimeError(
            "Dhan response is missing fields: "
            f"{missing}\n"
            f"Response keys: {list(payload.keys())}"
        )

    lengths = {
        key: len(payload[key])
        for key in required
    }

    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            "Dhan response arrays have different "
            f"lengths: {lengths}"
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

    frame["timestamp"] = (
        pd.to_datetime(
            frame["timestamp"],
            unit="s",
            utc=True,
        )
        .dt.tz_convert(IST)
    )

    return (
        frame
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def inspect_window(
    *,
    security_id: str,
    interval: str,
    start: str,
    end: str,
) -> None:
    """Fetch and display one diagnostic window."""

    print("=" * 80)
    print(f"Interval : {interval} minute")
    print(f"Request  : {start} -> {end}")

    frame = fetch(
        security_id=security_id,
        interval=interval,
        from_date=start,
        to_date=end,
    )

    print(f"Rows     : {len(frame)}")

    if frame.empty:
        print("First    : NONE")
        print("Last     : NONE")
        return

    print(
        f"First    : {frame['timestamp'].iloc[0]}"
    )
    print(
        f"Last     : {frame['timestamp'].iloc[-1]}"
    )

    print("\nLast 10 timestamps:")

    for timestamp in frame["timestamp"].tail(10):
        print(f"  {timestamp}")


def main() -> None:
    """Run the Dhan historical-data time-window diagnostic."""

    load_dotenv()

    access_token = os.getenv(
        "DHAN_ACCESS_TOKEN",
        "",
    ).strip()

    if not access_token:
        raise RuntimeError(
            "DHAN_ACCESS_TOKEN is missing from "
            ".env/environment."
        )

    master_path = Path(
        "data/raw/dhan/instrument_master/"
        "api-scrip-master.csv"
    )

    if not master_path.exists():
        raise FileNotFoundError(
            f"Dhan instrument master not found: "
            f"{master_path}"
        )

    master = DhanInstrumentMaster.from_csv(
        master_path
    )

    symbol = "HDFCBANK"

    instrument = master.resolve(symbol)

    print(
        f"HDFCBANK Security ID: "
        f"{instrument.security_id}"
    )
    print()

    # --------------------------------------------------------------
    # 1-minute tests
    # --------------------------------------------------------------

    print("\n1-MINUTE TESTS")

    inspect_window(
        security_id=instrument.security_id,
        interval="1",
        start="2026-08-31 09:15:00",
        end="2026-08-31 12:00:00",
    )

    inspect_window(
        security_id=instrument.security_id,
        interval="1",
        start="2026-08-31 12:00:00",
        end="2026-08-31 14:00:00",
    )

    inspect_window(
        security_id=instrument.security_id,
        interval="1",
        start="2026-08-31 14:00:00",
        end="2026-08-31 15:30:00",
    )

    # --------------------------------------------------------------
    # 5-minute tests
    # --------------------------------------------------------------

    print("\n\n5-MINUTE TESTS")

    inspect_window(
        security_id=instrument.security_id,
        interval="5",
        start="2026-08-31 09:15:00",
        end="2026-08-31 12:00:00",
    )

    inspect_window(
        security_id=instrument.security_id,
        interval="5",
        start="2026-08-31 12:00:00",
        end="2026-08-31 14:00:00",
    )

    inspect_window(
        security_id=instrument.security_id,
        interval="5",
        start="2026-08-31 14:00:00",
        end="2026-08-31 15:30:00",
    )


if __name__ == "__main__":
    main()