"""Market-data provider construction for the CLI."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from .dhan_instruments import (
    DHAN_INSTRUMENT_MASTER_URL,
    DhanInstrumentMaster,
)
from .dhan_market_data import DhanProvider
from .market_data import MarketDataProvider, ProviderError, YFinanceProvider


DEFAULT_DHAN_MASTER_CACHE = (
    Path("data")
    / "raw"
    / "dhan"
    / "instrument_master"
    / "api-scrip-master.csv"
)


def _download_instrument_master(
    url: str = DHAN_INSTRUMENT_MASTER_URL,
) -> bytes:
    """Download the public Dhan instrument master CSV."""

    request = Request(
        url,
        headers={
            "User-Agent": "ai-trading-platform-experiment-001/1.0"
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    except Exception as exc:
        raise ProviderError(
            f"Unable to download Dhan instrument master from "
            f"{url}: {exc}"
        ) from exc


def _load_or_download_dhan_master(
    cache_path: Path,
) -> DhanInstrumentMaster:
    """Load a cached Dhan master, downloading it when absent."""

    cache_path = Path(cache_path)

    if not cache_path.exists():
        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cache_path.write_bytes(
            _download_instrument_master()
        )

    try:
        return DhanInstrumentMaster.from_csv(
            cache_path
        )
    except Exception as exc:
        raise ProviderError(
            f"Unable to load Dhan instrument master: "
            f"{cache_path}: {exc}"
        ) from exc


def build_market_data_provider(
    name: str,
    *,
    env_file: str | Path | None = None,
    instrument_master_path: str | Path | None = None,
    dhan_master_cache: str | Path = DEFAULT_DHAN_MASTER_CACHE,
) -> MarketDataProvider:
    """Build a configured market-data provider.

    Dhan credentials are intentionally read only from
    environment/.env.

    They are never accepted as CLI arguments.
    """

    provider_name = name.strip().lower()

    if provider_name == "yfinance":
        return YFinanceProvider()

    if provider_name != "dhan":
        raise ProviderError(
            f"Unsupported market-data provider: {name!r}. "
            "Expected 'yfinance' or 'dhan'."
        )

    if env_file is not None:
        load_dotenv(
            dotenv_path=env_file,
            override=False,
        )
    else:
        load_dotenv(
            override=False
        )

    access_token = os.getenv(
        "DHAN_ACCESS_TOKEN",
        "",
    ).strip()

    if not access_token:
        raise ProviderError(
            "Dhan provider requires DHAN_ACCESS_TOKEN "
            "in .env or the environment."
        )

    master_path = (
        Path(instrument_master_path)
        if instrument_master_path is not None
        else Path(dhan_master_cache)
    )

    master = _load_or_download_dhan_master(
        master_path
    )

    return DhanProvider(
        access_token=access_token,
        instrument_master=master,
    )