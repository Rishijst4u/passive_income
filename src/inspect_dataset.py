"""Inspect the downloaded research dataset for Experiment #001."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
)


def find_stock_files() -> list[Path]:
    """Find all processed 5-minute Parquet files."""

    return sorted(
        DATA_ROOT.glob("*/*/*.parquet")
    )


def inspect_file(path: Path) -> dict:
    """Inspect one stock dataset."""

    frame = pd.read_parquet(path)

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
    )

    trading_days = frame["timestamp"].dt.date.unique()

    daily_counts = (
        frame.groupby(
            frame["timestamp"].dt.date
        )
        .size()
    )

    return {
        "symbol": path.parts[-3],
        "rows": len(frame),
        "trading_days": len(trading_days),
        "min_timestamp": frame["timestamp"].min(),
        "max_timestamp": frame["timestamp"].max(),
        "min_daily_candles": int(daily_counts.min()),
        "max_daily_candles": int(daily_counts.max()),
        "total_daily_rows": int(daily_counts.sum()),
    }


def main() -> None:
    """Inspect all downloaded stock datasets."""

    files = find_stock_files()

    if not files:
        raise SystemExit(
            f"No processed Parquet files found under {DATA_ROOT}"
        )

    results = []

    for path in files:
        try:
            result = inspect_file(path)
            results.append(result)
        except Exception as exc:
            results.append(
                {
                    "symbol": path.parts[-3],
                    "rows": 0,
                    "trading_days": 0,
                    "min_timestamp": None,
                    "max_timestamp": None,
                    "min_daily_candles": 0,
                    "max_daily_candles": 0,
                    "total_daily_rows": 0,
                    "error": str(exc),
                }
            )

    report = pd.DataFrame(results)

    print("=" * 100)
    print("EXPERIMENT #001 - DATASET INSPECTION")
    print("=" * 100)
    print()
    print(f"Data directory : {DATA_ROOT}")
    print(f"Files found    : {len(files)}")
    print()

    display_columns = [
        "symbol",
        "rows",
        "trading_days",
        "min_timestamp",
        "max_timestamp",
        "min_daily_candles",
        "max_daily_candles",
    ]

    print(
        report[
            [
                column
                for column in display_columns
                if column in report.columns
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        f"Stocks inspected : {len(report)}"
    )

    print(
        f"Stocks with data : "
        f"{(report['rows'] > 0).sum()}"
    )

    print(
        f"Total candles    : "
        f"{report['rows'].sum():,}"
    )

    print()

    if "error" in report.columns:
        errors = report[
            report["error"].notna()
        ]

        if not errors.empty:
            print("ERRORS:")
            for _, row in errors.iterrows():
                print(
                    f"- {row['symbol']}: "
                    f"{row['error']}"
                )

    print()
    print("Expected development-session characteristics:")
    print("- First candle approximately 09:15")
    print("- Last candle approximately 15:10")
    print("- 72 candles for a complete Dhan development day")
    print("- No internal missing candles")
    print("- Known NSE holidays should not appear")


if __name__ == "__main__":
    main()