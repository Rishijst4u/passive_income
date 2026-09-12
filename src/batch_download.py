"""Batch historical market-data downloader for Experiment #001."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .market_data import DataValidationError, ProviderError, ingest_market_data
from .provider_factory import build_market_data_provider
from .research_universe import get_research_universe


@dataclass(frozen=True)
class DownloadResult:
    """Result for one symbol download."""

    symbol: str
    success: bool
    message: str


def download_symbol(
    *,
    symbol: str,
    provider_name: str,
    start: str | date,
    end: str | date,
    interval: str,
    instrument_master: str | Path | None = None,
) -> DownloadResult:
    """Download and validate historical data for one symbol."""

    try:
        provider = build_market_data_provider(
            provider_name,
            instrument_master_path=instrument_master,
        )

        ingest_market_data(
            provider=provider,
            symbol=symbol,
            start=start,
            end=end,
            interval=interval,
        )

        return DownloadResult(
            symbol=symbol,
            success=True,
            message="downloaded successfully",
        )

    except (
        DataValidationError,
        ProviderError,
        ValueError,
        RuntimeError,
    ) as exc:
        return DownloadResult(
            symbol=symbol,
            success=False,
            message=str(exc),
        )


def download_universe(
    *,
    provider_name: str,
    start: str | date,
    end: str | date,
    interval: str,
    instrument_master: str | Path | None = None,
    symbols: tuple[str, ...] | None = None,
) -> list[DownloadResult]:
    """Download historical data for the complete research universe."""

    universe = (
        symbols
        if symbols is not None
        else get_research_universe()
    )

    results: list[DownloadResult] = []

    for symbol in universe:
        print(f"\nDownloading {symbol}...")

        result = download_symbol(
            symbol=symbol,
            provider_name=provider_name,
            start=start,
            end=end,
            interval=interval,
            instrument_master=instrument_master,
        )

        results.append(result)

        if result.success:
            print(f"  SUCCESS: {symbol}")
        else:
            print(f"  FAILED: {symbol}")
            print(f"  Reason: {result.message}")

    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Download historical market data for "
            "Experiment #001 research universe."
        )
    )

    parser.add_argument(
        "--provider",
        default="dhan",
        choices=("dhan", "yfinance"),
        help="Market-data provider.",
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Start date, e.g. 2026-08-01.",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="End date, e.g. 2026-09-01.",
    )

    parser.add_argument(
        "--interval",
        default="5m",
        choices=("5m",),
        help="Historical candle interval.",
    )

    parser.add_argument(
        "--instrument-master",
        default=None,
        help="Optional Dhan instrument-master CSV path.",
    )

    return parser


def main() -> None:
    """Run the batch downloader."""

    parser = build_parser()
    args = parser.parse_args()

    print("=" * 70)
    print("Experiment #001 - Batch Market Data Download")
    print("=" * 70)
    print(f"Provider : {args.provider}")
    print(f"Start    : {args.start}")
    print(f"End      : {args.end}")
    print(f"Interval : {args.interval}")
    print()

    universe = get_research_universe()

    print(f"Stocks   : {len(universe)}")
    print(", ".join(universe))
    print()

    results = download_universe(
        provider_name=args.provider,
        start=args.start,
        end=args.end,
        interval=args.interval,
        instrument_master=args.instrument_master,
    )

    successful = [
        result
        for result in results
        if result.success
    ]

    failed = [
        result
        for result in results
        if not result.success
    ]

    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Total    : {len(results)}")
    print(f"Success  : {len(successful)}")
    print(f"Failed   : {len(failed)}")

    if failed:
        print()
        print("Failed symbols:")

        for result in failed:
            print(f"- {result.symbol}: {result.message}")

    if not successful:
        raise SystemExit(
            "No symbols were downloaded successfully."
        )


if __name__ == "__main__":
    main()