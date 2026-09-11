"""Reporting helpers for Experiment #001 backtests."""

from dataclasses import asdict, is_dataclass
import pandas as pd

from .analytics import BacktestMetrics
from .backtest import BacktestResult


def _object_to_dict(obj) -> dict:
    """Convert a dataclass or namespace-like object to a dictionary."""
    if is_dataclass(obj):
        return asdict(obj)

    if hasattr(obj, "__dict__"):
        return vars(obj).copy()

    raise TypeError(
        f"Expected a dataclass or object with __dict__, got {type(obj).__name__}"
    )


def trades_to_dataframe(result: BacktestResult) -> pd.DataFrame:
    if not result.trades:
        return pd.DataFrame()

    return pd.DataFrame(
        [_object_to_dict(trade) for trade in result.trades]
    )


def rejected_signals_to_dataframe(result: BacktestResult) -> pd.DataFrame:
    if not result.rejected_signals:
        return pd.DataFrame(
            columns=["symbol", "timestamp", "reason"]
        )

    return pd.DataFrame(
        [_object_to_dict(signal) for signal in result.rejected_signals]
    )


def format_metrics_report(
    metrics: BacktestMetrics,
    rejected_signals: int = 0,
) -> str:
    return "\n".join(
        [
            "",
            "=" * 60,
            "Experiment #001",
            "VWAP + Momentum + Volume Breakout v1",
            "=" * 60,
            "",
            f"Starting Capital : ₹{metrics.starting_capital:,.2f}",
            f"Total Trades     : {metrics.total_trades}",
            f"Winning Trades   : {metrics.winning_trades}",
            f"Losing Trades    : {metrics.losing_trades}",
            f"Breakeven Trades : {metrics.breakeven_trades}",
            f"Rejected Signals : {rejected_signals}",
            f"Win Rate         : {metrics.win_rate:.2f}%",
            "",
            f"Gross P&L        : ₹{metrics.gross_pnl:,.2f}",
            f"Trading Costs    : ₹{metrics.total_costs:,.2f}",
            f"Net P&L          : ₹{metrics.net_pnl:,.2f}",
            f"Return           : {metrics.return_pct:.2f}%",
            "",
            f"Average Winner   : ₹{metrics.average_winner:,.2f}",
            f"Average Loser    : ₹{metrics.average_loser:,.2f}",
            f"Expectancy       : ₹{metrics.expectancy_per_trade:,.2f}",
            f"Payoff Ratio     : {metrics.payoff_ratio:.2f}",
            f"Profit Factor    : {metrics.profit_factor:.2f}",
            "",
            f"Max Drawdown     : ₹{metrics.max_drawdown:,.2f}",
            f"Max Drawdown %   : {metrics.max_drawdown_pct:.2f}%",
            f"Max Loss Streak  : {metrics.max_consecutive_losses}",
            f"Avg Daily P&L    : ₹{metrics.average_daily_pnl:,.2f}",
            "",
            "=" * 60,
        ]
    )