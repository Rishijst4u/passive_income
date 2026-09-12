"""Command-line interface for Experiment #001."""

import argparse
from pathlib import Path

import pandas as pd

from .analytics import calculate_metrics
from .backtest import run_backtest
from .market_data import (
    ProviderError,
    ingest_market_data,
)
from .provider_factory import (
    build_market_data_provider,
)
from .report import (
    format_metrics_report,
    rejected_signals_to_dataframe,
    trades_to_dataframe,
)
from .sample_data import make_sample_data


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "AI Trading Platform - Experiment #001"
        )
    )

    parser.add_argument(
        "command",
        choices=[
            "backtest",
            "download",
        ],
    )

    parser.add_argument(
        "--symbol"
    )

    parser.add_argument(
        "--start"
    )

    parser.add_argument(
        "--end"
    )

    parser.add_argument(
        "--interval",
        default="5m",
    )

    parser.add_argument(
        "--provider",
        choices=[
            "yfinance",
            "dhan",
        ],
        default="yfinance",
        help=(
            "Market-data provider for the "
            "download command."
        ),
    )

    parser.add_argument(
        "--instrument-master",
        help=(
            "Optional local Dhan instrument-master CSV. "
            "If omitted, "
            "data/raw/dhan/instrument_master/"
            "api-scrip-master.csv is used as the cache."
        ),
    )

    parser.add_argument(
        "--data",
        help=(
            "Path to processed Parquet market data. "
            "If omitted, sample data is used."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "data/processed/backtest_results"
        ),
        help=(
            "Directory for backtest output files."
        ),
    )

    return parser


def run_backtest_command(args) -> None:
    """Run a backtest and print/save its results."""

    if args.data:
        data_path = Path(
            args.data
        )

        if not data_path.exists():
            raise SystemExit(
                "Backtest data file does not exist: "
                f"{data_path}"
            )

        print(
            f"Loading market data: {data_path}"
        )

        df = pd.read_parquet(
            data_path
        )

        if df.empty:
            raise SystemExit(
                "Backtest data file is empty: "
                f"{data_path}"
            )

        result = run_backtest(
            df=df,
            symbol=args.symbol or "UNKNOWN",
        )

        print(
            f"Rows loaded : {len(df)}"
        )

    else:
        print(
            "No --data supplied; using sample data."
        )

        print(
            "NOTE: sample data is for pipeline "
            "validation only; it is not evidence "
            "of strategy profitability."
        )

        df = make_sample_data()

        result = run_backtest(
            df=df,
            symbol=args.symbol or "TEST",
        )

    from .config import load_settings

    settings = load_settings()

    metrics = calculate_metrics(
        result.trades,
        starting_capital=settings.capital,
    )

    print(
        format_metrics_report(
            metrics,
            rejected_signals=(
                len(result.rejected_signals)
            ),
        )
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades_df = trades_to_dataframe(
        result
    )

    rejected_df = (
        rejected_signals_to_dataframe(
            result
        )
    )

    trades_path = (
        output_dir / "trades.csv"
    )

    rejected_path = (
        output_dir / "rejected_signals.csv"
    )

    trades_df.to_csv(
        trades_path,
        index=False,
    )

    rejected_df.to_csv(
        rejected_path,
        index=False,
    )

    print("")
    print(
        f"Trade log : {trades_path}"
    )

    print(
        "Rejected signals : "
        f"{rejected_path}"
    )


def run_download_command(
    args,
    parser: argparse.ArgumentParser,
) -> None:
    """Download and validate market data."""

    if not all(
        [
            args.symbol,
            args.start,
            args.end,
        ]
    ):
        parser.error(
            "download requires --symbol, "
            "--start, and --end"
        )

    try:
        provider = build_market_data_provider(
            args.provider,
            instrument_master_path=(
                args.instrument_master
            ),
        )

        result = ingest_market_data(
            provider,
            args.symbol,
            args.start,
            args.end,
            args.interval,
        )

    except ProviderError as exc:
        parser.error(
            str(exc)
        )

    print(
        f"Provider: {provider.name}"
    )

    print(
        f"Raw data: {result.raw_path}"
    )

    print(
        "Validated data: "
        f"{result.processed_path}"
    )

    print(
        result.report
    )


def main(argv=None):
    """CLI entry point."""

    parser = build_parser()

    args = parser.parse_args(
        argv
    )

    if args.command == "backtest":
        run_backtest_command(
            args
        )

    elif args.command == "download":
        run_download_command(
            args,
            parser,
        )


if __name__ == "__main__":
    main()