"""Run the shared-account portfolio backtest for Experiment #001."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analytics import calculate_max_drawdown
from .config import load_settings
from .portfolio_backtest import run_portfolio_backtest
from .research_universe import get_research_universe
from .report import rejected_signals_to_dataframe, trades_to_dataframe


DEFAULT_DATA_ROOT = Path("data") / "processed"
DEFAULT_OUTPUT_DIR = Path("data") / "reports"
DEFAULT_INTERVAL = "5m"


def load_processed_data(
    symbols: tuple[str, ...],
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    interval: str = DEFAULT_INTERVAL,
) -> dict[str, pd.DataFrame]:
    """Load processed Parquet data for every research-universe symbol."""

    data: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        symbol_dir = Path(data_root) / symbol / interval
        files = sorted(symbol_dir.glob("*.parquet"))

        if not files:
            raise FileNotFoundError(
                f"No processed Parquet files found for {symbol} "
                f"under {symbol_dir}"
            )

        frames = [pd.read_parquet(path) for path in files]

        frame = pd.concat(frames, ignore_index=True)

        if "timestamp" not in frame.columns:
            raise ValueError(
                f"Processed data for {symbol} does not contain "
                "'timestamp' column."
            )

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            errors="raise",
        )

        frame = (
            frame.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )

        data[symbol] = frame

    return data


def build_summary(
    trades_df: pd.DataFrame,
    rejected_count: int,
    starting_capital: float,
) -> pd.DataFrame:
    """Build a compact portfolio-backtest summary."""

    total_trades = len(trades_df)

    if total_trades == 0:
        gross_pnl = 0.0
        total_cost = 0.0
        net_pnl = 0.0
        wins = 0
        losses = 0
        win_rate = 0.0
        average_winner = 0.0
        average_loser = 0.0
        expectancy = 0.0
        profit_factor = 0.0
        payoff_ratio = 0.0
        max_drawdown = 0.0
        max_drawdown_pct = 0.0
    else:
        pnl = trades_df["net_pnl"].astype(float)

        gross_pnl = float(trades_df["gross_pnl"].sum())
        total_cost = float(trades_df["total_cost"].sum())
        net_pnl = float(pnl.sum())

        wins = int((pnl > 0).sum())
        losses = int((pnl < 0).sum())

        win_rate = wins / total_trades * 100.0

        winning_pnl = pnl[pnl > 0]
        losing_pnl = pnl[pnl < 0]

        average_winner = (
            float(winning_pnl.mean())
            if not winning_pnl.empty
            else 0.0
        )

        average_loser = (
            float(losing_pnl.mean())
            if not losing_pnl.empty
            else 0.0
        )

        expectancy = float(pnl.mean())

        gross_profit = float(winning_pnl.sum())
        gross_loss = float(-losing_pnl.sum())

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = 0.0

        if average_loser != 0:
            payoff_ratio = average_winner / abs(average_loser)
        else:
            payoff_ratio = 0.0

        equity_pnl = pnl.reset_index(drop=True)

        max_drawdown, max_drawdown_pct = calculate_max_drawdown(
            equity_pnl,
            starting_capital,
        )

    ending_capital = starting_capital + net_pnl
    return_pct = (
        net_pnl / starting_capital * 100.0
        if starting_capital > 0
        else 0.0
    )

    return pd.DataFrame(
        [
            {
                "strategy": "VWAP + Momentum + Volume Breakout v1",
                "mode": "Portfolio historical backtest",
                "starting_capital": starting_capital,
                "ending_capital": ending_capital,
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "win_rate_pct": win_rate,
                "gross_pnl": gross_pnl,
                "total_cost": total_cost,
                "net_pnl": net_pnl,
                "return_pct": return_pct,
                "expectancy_per_trade": expectancy,
                "average_winner": average_winner,
                "average_loser": average_loser,
                "payoff_ratio": payoff_ratio,
                "profit_factor": profit_factor,
                "max_drawdown": max_drawdown,
                "max_drawdown_pct": max_drawdown_pct,
                "rejected_signals": rejected_count,
            }
        ]
    )


def run(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """Run the portfolio backtest and write report files."""

    settings = load_settings()
    symbols = get_research_universe()

    data = load_processed_data(
        symbols,
        data_root=data_root,
        interval=interval,
    )

    result = run_portfolio_backtest(
        data,
        settings=settings,
    )

    trades_df = trades_to_dataframe(result)
    rejected_df = rejected_signals_to_dataframe(result)

    output_dir.mkdir(parents=True, exist_ok=True)

    trades_path = output_dir / "portfolio_trades.csv"
    rejected_path = output_dir / "portfolio_rejections.csv"
    summary_path = output_dir / "portfolio_summary.csv"

    trades_df.to_csv(trades_path, index=False)
    rejected_df.to_csv(rejected_path, index=False)

    summary_df = build_summary(
        trades_df,
        rejected_count=len(result.rejected_signals),
        starting_capital=settings.capital,
    )

    summary_df.to_csv(summary_path, index=False)

    print()
    print("=" * 60)
    print("EXPERIMENT #001 — PORTFOLIO BACKTEST")
    print("=" * 60)
    print(f"Strategy          : VWAP + Momentum + Volume Breakout v1")
    print(f"Mode              : Shared-account historical backtest")
    print(f"Capital           : ₹{settings.capital:,.2f}")
    print(f"Stocks            : {len(symbols)}")
    print(f"Interval          : {interval}")
    print(f"Total trades      : {len(result.trades)}")
    print(f"Rejected signals  : {len(result.rejected_signals)}")

    row = summary_df.iloc[0]

    print(f"Wins              : {int(row['wins'])}")
    print(f"Losses            : {int(row['losses'])}")
    print(f"Win rate          : {row['win_rate_pct']:.2f}%")
    print(f"Gross P&L         : ₹{row['gross_pnl']:,.2f}")
    print(f"Trading costs     : ₹{row['total_cost']:,.2f}")
    print(f"Net P&L           : ₹{row['net_pnl']:,.2f}")
    print(f"Return            : {row['return_pct']:.2f}%")
    print(f"Expectancy/trade  : ₹{row['expectancy_per_trade']:,.2f}")
    print(f"Profit factor     : {row['profit_factor']:.3f}")
    print(f"Payoff ratio      : {row['payoff_ratio']:.3f}")
    print(f"Max drawdown      : ₹{row['max_drawdown']:,.2f}")
    print(f"Max drawdown %    : {row['max_drawdown_pct']:.2f}%")
    print()
    print(f"Trades report     : {trades_path}")
    print(f"Rejections report : {rejected_path}")
    print(f"Summary report    : {summary_path}")
    print("=" * 60)

    return summary_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Experiment #001 portfolio backtest."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Processed market-data root.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for backtest reports.",
    )

    parser.add_argument(
        "--interval",
        default=DEFAULT_INTERVAL,
        choices=["5m"],
        help="Market-data interval.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run(
        data_root=args.data_root,
        output_dir=args.output_dir,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()