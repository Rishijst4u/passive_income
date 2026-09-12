"""Multi-stock baseline backtest for Experiment #001."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .config import load_settings
from .costs import CostSettings
from .research_universe import get_research_universe


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "reports"
)

TRADES_OUTPUT = OUTPUT_ROOT / "baseline_trades.csv"
SUMMARY_OUTPUT = OUTPUT_ROOT / "baseline_summary.csv"
REJECTIONS_OUTPUT = OUTPUT_ROOT / "baseline_rejections.csv"


def find_data_file(symbol: str) -> Path:
    """Find the processed 5-minute Parquet file for a symbol."""

    symbol_root = (
        PROCESSED_DATA_ROOT
        / symbol
        / "5m"
    )

    files = sorted(symbol_root.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No processed 5-minute data found for {symbol} "
            f"under {symbol_root}"
        )

    if len(files) > 1:
        raise RuntimeError(
            f"Multiple processed files found for {symbol}: "
            f"{files}"
        )

    return files[0]


def calculate_symbol_metrics(
    *,
    symbol: str,
    trades: list,
    starting_capital: float,
    rejected_signals: int,
) -> dict:
    """Calculate baseline metrics for one stock."""

    total_trades = len(trades)

    if total_trades == 0:
        return {
            "symbol": symbol,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "avg_net_pnl": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "rejected_signals": rejected_signals,
            "return_pct": 0.0,
        }

    pnl = pd.Series(
        [
            float(trade.net_pnl)
            for trade in trades
        ],
        dtype="float64",
    )

    gross_pnl = float(
        sum(float(trade.gross_pnl) for trade in trades)
    )

    net_pnl = float(pnl.sum())

    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())

    win_rate = (
        wins / total_trades * 100.0
    )

    winning_pnl = float(
        pnl[pnl > 0].sum()
    )

    losing_pnl = float(
        -pnl[pnl < 0].sum()
    )

    if losing_pnl > 0:
        profit_factor = (
            winning_pnl / losing_pnl
        )
    elif winning_pnl > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    equity = (
        starting_capital
        + pnl.cumsum()
    )

    running_peak = equity.cummax()

    drawdown = equity - running_peak

    max_drawdown = float(
        drawdown.min()
    )

    if starting_capital > 0:
        max_drawdown_pct = (
            max_drawdown
            / starting_capital
            * 100.0
        )
    else:
        max_drawdown_pct = 0.0

    return_pct = (
        net_pnl
        / starting_capital
        * 100.0
    )

    return {
        "symbol": symbol,
        "trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "avg_net_pnl": net_pnl / total_trades,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "rejected_signals": rejected_signals,
        "return_pct": return_pct,
    }


def trade_to_dict(
    symbol: str,
    trade,
) -> dict:
    """Convert a Trade object into a report row."""

    return {
        "symbol": symbol,
        "signal_time": trade.signal_time,
        "entry_time": trade.entry_time,
        "entry": trade.entry,
        "stop": trade.stop,
        "target": trade.target,
        "quantity": trade.quantity,
        "exit_time": trade.exit_time,
        "exit": trade.exit,
        "gross_pnl": trade.gross_pnl,
        "total_cost": trade.total_cost,
        "net_pnl": trade.net_pnl,
        "brokerage": trade.brokerage,
        "stt": trade.stt,
        "exchange_transaction_charges": (
            trade.exchange_transaction_charges
        ),
        "sebi_charges": trade.sebi_charges,
        "stamp_duty": trade.stamp_duty,
        "gst": trade.gst,
        "ipft": trade.ipft,
        "slippage": trade.slippage,
        "reason": trade.reason,
    }


def rejection_to_dict(
    symbol: str,
    rejection,
) -> dict:
    """Convert a rejected signal into a report row."""

    return {
        "symbol": symbol,
        "timestamp": rejection.timestamp,
        "reason": rejection.reason,
    }


def run_multi_stock_backtest(
    *,
    symbols: tuple[str, ...] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run the existing strategy across all research stocks."""

    settings = load_settings()

    cost_settings = CostSettings()

    universe = (
        symbols
        if symbols is not None
        else get_research_universe()
    )

    all_trades: list[dict] = []
    all_rejections: list[dict] = []
    summaries: list[dict] = []

    for symbol in universe:
        print()
        print("-" * 80)
        print(f"BACKTESTING {symbol}")
        print("-" * 80)

        data_file = find_data_file(symbol)

        frame = pd.read_parquet(data_file)

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
        )

        print(
            f"Data rows : {len(frame):,}"
        )
        print(
            f"Period    : "
            f"{frame['timestamp'].min()} -> "
            f"{frame['timestamp'].max()}"
        )

        result = run_backtest(
            frame,
            symbol=symbol,
            settings=settings,
            cost_settings=cost_settings,
        )

        for trade in result.trades:
            all_trades.append(
                trade_to_dict(
                    symbol,
                    trade,
                )
            )

        for rejection in result.rejected_signals:
            all_rejections.append(
                rejection_to_dict(
                    symbol,
                    rejection,
                )
            )

        metrics = calculate_symbol_metrics(
            symbol=symbol,
            trades=result.trades,
            starting_capital=settings.capital,
            rejected_signals=len(
                result.rejected_signals
            ),
        )

        summaries.append(metrics)

        print(
            f"Trades    : {metrics['trades']}"
        )
        print(
            f"Net P&L   : ₹{metrics['net_pnl']:,.2f}"
        )
        print(
            f"Win rate  : "
            f"{metrics['win_rate_pct']:.2f}%"
        )
        print(
            f"PF        : "
            f"{metrics['profit_factor']:.2f}"
        )

    trades_df = pd.DataFrame(
        all_trades
    )

    rejections_df = pd.DataFrame(
        all_rejections,
        columns=[
            "symbol",
            "timestamp",
            "reason",
        ],
    )

    summary_df = pd.DataFrame(
        summaries
    )

    return (
        trades_df,
        summary_df,
        rejections_df,
    )


def save_reports(
    *,
    trades_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    rejections_df: pd.DataFrame,
) -> None:
    """Save consolidated backtest reports."""

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades_df.to_csv(
        TRADES_OUTPUT,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    rejections_df.to_csv(
        REJECTIONS_OUTPUT,
        index=False,
    )


def print_final_summary(
    *,
    trades_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    rejections_df: pd.DataFrame,
) -> None:
    """Print consolidated baseline results."""

    total_trades = len(trades_df)

    if total_trades:
        total_net_pnl = float(
            trades_df["net_pnl"].sum()
        )

        total_gross_pnl = float(
            trades_df["gross_pnl"].sum()
        )

        wins = int(
            (trades_df["net_pnl"] > 0).sum()
        )

        losses = int(
            (trades_df["net_pnl"] < 0).sum()
        )

        win_rate = (
            wins
            / total_trades
            * 100.0
        )

        winning_pnl = float(
            trades_df.loc[
                trades_df["net_pnl"] > 0,
                "net_pnl",
            ].sum()
        )

        losing_pnl = float(
            -trades_df.loc[
                trades_df["net_pnl"] < 0,
                "net_pnl",
            ].sum()
        )

        if losing_pnl > 0:
            profit_factor = (
                winning_pnl
                / losing_pnl
            )
        elif winning_pnl > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        equity = (
            50_000.0
            + trades_df["net_pnl"].cumsum()
        )

        running_peak = equity.cummax()

        max_drawdown = float(
            (equity - running_peak).min()
        )

        return_pct = (
            total_net_pnl
            / 50_000.0
            * 100.0
        )

    else:
        total_net_pnl = 0.0
        total_gross_pnl = 0.0
        wins = 0
        losses = 0
        win_rate = 0.0
        profit_factor = 0.0
        max_drawdown = 0.0
        return_pct = 0.0

    print()
    print("=" * 80)
    print("EXPERIMENT #001 - BASELINE BACKTEST")
    print("=" * 80)
    print()
    print("Strategy : VWAP + Momentum + Volume Breakout v1")
    print("Mode     : Historical backtest")
    print("Capital  : ₹50,000")
    print("Stocks   : 20")
    print()
    print(f"Total trades       : {total_trades}")
    print(f"Wins               : {wins}")
    print(f"Losses             : {losses}")
    print(f"Win rate           : {win_rate:.2f}%")
    print(f"Gross P&L          : ₹{total_gross_pnl:,.2f}")
    print(f"Net P&L            : ₹{total_net_pnl:,.2f}")
    print(f"Return             : {return_pct:.2f}%")
    print(f"Profit factor      : {profit_factor:.2f}")
    print(f"Max drawdown       : ₹{max_drawdown:,.2f}")
    print(
        f"Rejected signals   : "
        f"{len(rejections_df)}"
    )
    print()
    print("Reports:")
    print(f"- {TRADES_OUTPUT}")
    print(f"- {SUMMARY_OUTPUT}")
    print(f"- {REJECTIONS_OUTPUT}")
    print()

    if not summary_df.empty:
        print("STOCK-BY-STOCK RESULTS")
        print("-" * 80)

        display_columns = [
            "symbol",
            "trades",
            "win_rate_pct",
            "net_pnl",
            "profit_factor",
            "max_drawdown",
        ]

        display_df = summary_df[
            display_columns
        ].copy()

        print(
            display_df.to_string(
                index=False,
                formatters={
                    "win_rate_pct": (
                        lambda value:
                        f"{value:.2f}%"
                    ),
                    "net_pnl": (
                        lambda value:
                        f"₹{value:,.2f}"
                    ),
                    "profit_factor": (
                        lambda value:
                        (
                            "inf"
                            if value == float("inf")
                            else f"{value:.2f}"
                        )
                    ),
                    "max_drawdown": (
                        lambda value:
                        f"₹{value:,.2f}"
                    ),
                },
            )
        )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Experiment #001 baseline "
            "backtest across the 20-stock universe."
        )
    )

    return parser


def main() -> None:
    """CLI entry point."""

    parser = build_parser()
    parser.parse_args()

    trades_df, summary_df, rejections_df = (
        run_multi_stock_backtest()
    )

    save_reports(
        trades_df=trades_df,
        summary_df=summary_df,
        rejections_df=rejections_df,
    )

    print_final_summary(
        trades_df=trades_df,
        summary_df=summary_df,
        rejections_df=rejections_df,
    )


if __name__ == "__main__":
    main()