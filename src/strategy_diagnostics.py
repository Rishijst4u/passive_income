"""Trade-level diagnostics for Strategy #002."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_settings
from .research_universe import get_research_universe
from .strategy import add_indicators


DEFAULT_DATA_ROOT = Path("data") / "processed"
DEFAULT_REPORT_ROOT = Path("data") / "reports"
DEFAULT_TRADES_FILE = DEFAULT_REPORT_ROOT / "portfolio_trades.csv"
DEFAULT_OUTPUT_FILE = DEFAULT_REPORT_ROOT / "strategy_002_trade_diagnostics.csv"

ASIA_KOLKATA = "Asia/Kolkata"


def _local_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        return timestamp.tz_localize(ASIA_KOLKATA)

    return timestamp.tz_convert(ASIA_KOLKATA)


def load_processed_data(
    symbols: tuple[str, ...],
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    interval: str = "5m",
) -> dict[str, pd.DataFrame]:
    """Load all processed data for the research universe."""

    data: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        symbol_dir = Path(data_root) / symbol / interval
        files = sorted(symbol_dir.glob("*.parquet"))

        if not files:
            raise FileNotFoundError(
                f"No processed Parquet files found for {symbol} "
                f"under {symbol_dir}"
            )

        frame = pd.concat(
            [pd.read_parquet(path) for path in files],
            ignore_index=True,
        )

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            errors="raise",
        ).map(_local_timestamp)

        frame = (
            frame.sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )

        data[symbol] = frame

    return data


def _safe_float(row: pd.Series, column: str) -> float | None:
    """Return a finite float or None."""

    if column not in row.index:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not pd.notna(result):
        return None

    return result


def _calculate_trade_diagnostics(
    trade: pd.Series,
    frame: pd.DataFrame,
) -> dict[str, object]:
    """Calculate setup and post-entry diagnostics for one trade."""

    signal_time = _local_timestamp(trade["signal_time"])
    entry_time = _local_timestamp(trade["entry_time"])
    exit_time = _local_timestamp(trade["exit_time"])

    history = frame[
        frame["timestamp"] <= signal_time
    ].copy()

    if history.empty:
        raise ValueError(
            f"No historical data available for signal time {signal_time}"
        )

    indicators = add_indicators(
        history,
        load_settings().strategy,
    )

    signal_row = indicators.iloc[-1]

    entry_price = float(trade["entry"])
    stop_price = float(trade["stop"])
    exit_price = float(trade["exit"])

    risk_per_share = entry_price - stop_price

    if risk_per_share > 0:
        entry_stop_distance_pct = (
            risk_per_share / entry_price * 100.0
        )
    else:
        entry_stop_distance_pct = None

    future = frame[
        (frame["timestamp"] >= entry_time)
        & (frame["timestamp"] <= exit_time)
    ].copy()

    if future.empty:
        future = frame[
            frame["timestamp"] >= entry_time
        ].head(1)

    if future.empty:
        mae = None
        mfe = None
    else:
        lowest_low = float(future["low"].min())
        highest_high = float(future["high"].max())

        mae = lowest_low - entry_price
        mfe = highest_high - entry_price

    if risk_per_share > 0 and mfe is not None:
        mfe_r = mfe / risk_per_share
    else:
        mfe_r = None

    if risk_per_share > 0 and mae is not None:
        mae_r = mae / risk_per_share
    else:
        mae_r = None

    if risk_per_share > 0:
        net_r = float(trade["net_pnl"]) / (
            risk_per_share * float(trade["quantity"])
        )
    else:
        net_r = None

    ema21 = _safe_float(signal_row, "ema21")
    ema50 = _safe_float(signal_row, "ema50")
    vwap = _safe_float(signal_row, "vwap")
    close = _safe_float(signal_row, "close")
    prev_low = _safe_float(signal_row, "prev_low")
    prev_close = _safe_float(signal_row, "prev_close")
    prev_ema21 = _safe_float(signal_row, "prev_ema21")
    prev_vwap = _safe_float(signal_row, "prev_vwap")

    if (
        ema21 is not None
        and ema50 is not None
        and ema21 != 0
    ):
        ema21_ema50_distance_pct = (
            (ema21 - ema50) / ema21 * 100.0
        )
    else:
        ema21_ema50_distance_pct = None

    if (
        close is not None
        and vwap is not None
        and vwap != 0
    ):
        close_vwap_distance_pct = (
            (close - vwap) / vwap * 100.0
        )
    else:
        close_vwap_distance_pct = None

    if (
        close is not None
        and ema21 is not None
        and ema21 != 0
    ):
        close_ema21_distance_pct = (
            (close - ema21) / ema21 * 100.0
        )
    else:
        close_ema21_distance_pct = None

    if (
        prev_low is not None
        and prev_ema21 is not None
        and prev_vwap is not None
    ):
        support_reference = max(
            prev_ema21,
            prev_vwap,
        )

        if support_reference != 0:
            pullback_distance_pct = (
                (support_reference - prev_low)
                / support_reference
                * 100.0
            )
        else:
            pullback_distance_pct = None
    else:
        pullback_distance_pct = None

    if (
        prev_close is not None
        and prev_vwap is not None
        and prev_vwap != 0
    ):
        previous_close_vwap_distance_pct = (
            (prev_close - prev_vwap)
            / prev_vwap
            * 100.0
        )
    else:
        previous_close_vwap_distance_pct = None

    signal_timestamp = signal_time

    return {
        "symbol": trade["symbol"],
        "signal_time": signal_timestamp,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry": entry_price,
        "stop": stop_price,
        "target": float(trade["target"]),
        "exit": exit_price,
        "quantity": int(trade["quantity"]),
        "signal_score": int(getattr(trade, "score", 0))
        if hasattr(trade, "score")
        else None,
        "rsi": _safe_float(signal_row, "rsi"),
        "vol_ratio": _safe_float(signal_row, "vol_ratio"),
        "ema9": _safe_float(signal_row, "ema9"),
        "ema21": ema21,
        "ema50": ema50,
        "vwap": vwap,
        "close_vwap_distance_pct": close_vwap_distance_pct,
        "close_ema21_distance_pct": close_ema21_distance_pct,
        "ema21_ema50_distance_pct": ema21_ema50_distance_pct,
        "previous_close_vwap_distance_pct": (
            previous_close_vwap_distance_pct
        ),
        "pullback_distance_pct": pullback_distance_pct,
        "body_ratio": _safe_float(signal_row, "body_ratio"),
        "risk_per_share": risk_per_share,
        "entry_stop_distance_pct": entry_stop_distance_pct,
        "mae": mae,
        "mfe": mfe,
        "mae_r": mae_r,
        "mfe_r": mfe_r,
        "net_r": net_r,
        "gross_pnl": float(trade["gross_pnl"]),
        "total_cost": float(trade["total_cost"]),
        "net_pnl": float(trade["net_pnl"]),
        "reason": trade["reason"],
        "signal_hour": signal_timestamp.hour,
        "signal_minute": signal_timestamp.minute,
    }


def build_diagnostics(
    trades_path: Path = DEFAULT_TRADES_FILE,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    interval: str = "5m",
) -> pd.DataFrame:
    """Build trade-level Strategy #002 diagnostics."""

    trades = pd.read_csv(trades_path)

    if trades.empty:
        return pd.DataFrame()

    required_columns = {
        "symbol",
        "signal_time",
        "entry_time",
        "exit_time",
        "entry",
        "stop",
        "target",
        "exit",
        "quantity",
        "gross_pnl",
        "total_cost",
        "net_pnl",
        "reason",
    }

    missing = required_columns.difference(trades.columns)

    if missing:
        raise ValueError(
            "Trade report is missing required columns: "
            f"{sorted(missing)}"
        )

    symbols = tuple(
        sorted(
            set(trades["symbol"].astype(str).str.upper())
        )
    )

    data = load_processed_data(
        symbols,
        data_root=data_root,
        interval=interval,
    )

    rows: list[dict[str, object]] = []

    for _, trade in trades.iterrows():
        symbol = str(trade["symbol"]).upper()

        diagnostic = _calculate_trade_diagnostics(
            trade,
            data[symbol],
        )

        rows.append(diagnostic)

    return pd.DataFrame(rows)


def print_summary(diagnostics: pd.DataFrame) -> None:
    """Print useful Strategy #002 diagnostic summaries."""

    if diagnostics.empty:
        print("No trades available for diagnostics.")
        return

    print()
    print("=" * 60)
    print("STRATEGY #002 — TRADE DIAGNOSTICS")
    print("=" * 60)

    print(f"Trades analyzed       : {len(diagnostics)}")

    stop_trades = diagnostics[
        diagnostics["reason"].astype(str).str.contains("STOP")
    ]

    target_trades = diagnostics[
        diagnostics["reason"].astype(str).eq("TARGET")
    ]

    print(f"Stop trades           : {len(stop_trades)}")
    print(f"Target trades         : {len(target_trades)}")

    if not stop_trades.empty:
        print()
        print("STOP TRADE CHARACTERISTICS")
        print("-" * 60)

        for column in (
            "rsi",
            "vol_ratio",
            "close_vwap_distance_pct",
            "close_ema21_distance_pct",
            "ema21_ema50_distance_pct",
            "pullback_distance_pct",
            "body_ratio",
            "entry_stop_distance_pct",
            "mae_r",
            "mfe_r",
            "net_r",
        ):
            if column in stop_trades.columns:
                value = stop_trades[column].mean()

                if pd.notna(value):
                    print(f"{column:35s}: {value: .4f}")

    if not target_trades.empty:
        print()
        print("TARGET TRADE CHARACTERISTICS")
        print("-" * 60)

        for column in (
            "rsi",
            "vol_ratio",
            "close_vwap_distance_pct",
            "close_ema21_distance_pct",
            "ema21_ema50_distance_pct",
            "pullback_distance_pct",
            "body_ratio",
            "entry_stop_distance_pct",
            "mae_r",
            "mfe_r",
            "net_r",
        ):
            if column in target_trades.columns:
                value = target_trades[column].mean()

                if pd.notna(value):
                    print(f"{column:35s}: {value: .4f}")

    print()
    print("STOP PERFORMANCE BY HOUR")
    print("-" * 60)

    stop_by_hour = (
        stop_trades.groupby("signal_hour")
        .agg(
            trades=("symbol", "size"),
            net_pnl=("net_pnl", "sum"),
            avg_mfe_r=("mfe_r", "mean"),
        )
        .reset_index()
    )

    for _, row in stop_by_hour.iterrows():
        print(
            f"{int(row['signal_hour']):02d}:00"
            f" | {int(row['trades']):2d} trades"
            f" | net ₹{row['net_pnl']:8.2f}"
            f" | avg MFE {row['avg_mfe_r']:.2f}R"
        )

    print()
    print("PER-SYMBOL PERFORMANCE")
    print("-" * 60)

    symbol_summary = (
        diagnostics.groupby("symbol")
        .agg(
            trades=("symbol", "size"),
            net_pnl=("net_pnl", "sum"),
            avg_mfe_r=("mfe_r", "mean"),
            avg_mae_r=("mae_r", "mean"),
        )
        .sort_values("net_pnl")
    )

    for symbol, row in symbol_summary.iterrows():
        print(
            f"{symbol:15s}"
            f" | {int(row['trades']):2d} trades"
            f" | net ₹{row['net_pnl']:8.2f}"
            f" | MFE {row['avg_mfe_r']:.2f}R"
            f" | MAE {row['avg_mae_r']:.2f}R"
        )

    print("=" * 60)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnose Strategy #002 trades."
    )

    parser.add_argument(
        "--trades",
        type=Path,
        default=DEFAULT_TRADES_FILE,
        help="Portfolio trades CSV.",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Processed market-data root.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output diagnostics CSV.",
    )

    parser.add_argument(
        "--interval",
        default="5m",
        choices=["5m"],
        help="Market-data interval.",
    )

    args = parser.parse_args()

    diagnostics = build_diagnostics(
        trades_path=args.trades,
        data_root=args.data_root,
        interval=args.interval,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics.to_csv(
        args.output,
        index=False,
    )

    print_summary(diagnostics)

    print()
    print(f"Diagnostics CSV      : {args.output}")


if __name__ == "__main__":
    main()