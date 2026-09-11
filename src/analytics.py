"""Performance analytics for Experiment #001 backtests."""

from dataclasses import dataclass

import pandas as pd

from .backtest import Trade


@dataclass(frozen=True)
class BacktestMetrics:
    starting_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    gross_pnl: float
    total_costs: float
    net_pnl: float
    return_pct: float
    average_winner: float
    average_loser: float
    expectancy_per_trade: float
    payoff_ratio: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    max_consecutive_losses: int
    average_daily_pnl: float


def calculate_max_drawdown(
    pnl_series: pd.Series,
    starting_capital: float,
) -> tuple[float, float]:
    """
    Calculate maximum portfolio drawdown.

    Drawdown is calculated from the portfolio equity curve:

        equity = starting capital + cumulative net P&L

    Maximum drawdown is the largest peak-to-trough decline.

    Maximum drawdown percentage is measured relative to the
    portfolio equity at the preceding peak.
    """
    if starting_capital <= 0:
        raise ValueError("starting_capital must be greater than zero")

    if pnl_series.empty:
        return 0.0, 0.0

    pnl = pnl_series.astype(float)

    cumulative_pnl = pnl.cumsum()
    equity = starting_capital + cumulative_pnl

    running_peak = equity.cummax()

    drawdown = equity - running_peak
    drawdown_pct_series = (drawdown / running_peak) * 100.0

    max_drawdown = float(drawdown.min())
    max_drawdown_pct = float(drawdown_pct_series.min())

    return max_drawdown, max_drawdown_pct


def calculate_max_consecutive_losses(pnl_series: pd.Series) -> int:
    """Return the maximum number of consecutive losing trades."""
    if pnl_series.empty:
        return 0

    max_losses = 0
    current_losses = 0

    for pnl in pnl_series.astype(float):
        if pnl < 0:
            current_losses += 1
            max_losses = max(max_losses, current_losses)
        else:
            current_losses = 0

    return max_losses


def calculate_metrics(
    trades: list[Trade],
    starting_capital: float,
) -> BacktestMetrics:
    """Calculate performance metrics for a collection of backtest trades."""
    if starting_capital <= 0:
        raise ValueError("starting_capital must be greater than zero")

    if not trades:
        return BacktestMetrics(
            starting_capital=starting_capital,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate=0.0,
            gross_pnl=0.0,
            total_costs=0.0,
            net_pnl=0.0,
            return_pct=0.0,
            average_winner=0.0,
            average_loser=0.0,
            expectancy_per_trade=0.0,
            payoff_ratio=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            max_consecutive_losses=0,
            average_daily_pnl=0.0,
        )

    net_pnl_series = pd.Series(
        [float(trade.net_pnl) for trade in trades],
        dtype=float,
    )

    gross_pnl = sum(float(trade.gross_pnl) for trade in trades)
    total_costs = sum(float(trade.total_cost) for trade in trades)
    net_pnl = float(net_pnl_series.sum())

    winning_trades = int((net_pnl_series > 0).sum())
    losing_trades = int((net_pnl_series < 0).sum())
    breakeven_trades = int((net_pnl_series == 0).sum())

    total_trades = len(trades)

    win_rate = (
        winning_trades / total_trades * 100.0
        if total_trades
        else 0.0
    )

    winners = net_pnl_series[net_pnl_series > 0]
    losers = net_pnl_series[net_pnl_series < 0]

    average_winner = (
        float(winners.mean())
        if not winners.empty
        else 0.0
    )

    average_loser = (
        float(losers.mean())
        if not losers.empty
        else 0.0
    )

    expectancy_per_trade = net_pnl / total_trades

    payoff_ratio = (
        average_winner / abs(average_loser)
        if average_loser < 0
        else 0.0
    )

    gross_profit = float(winners.sum()) if not winners.empty else 0.0
    gross_loss = float(losers.sum()) if not losers.empty else 0.0

    profit_factor = (
        gross_profit / abs(gross_loss)
        if gross_loss < 0
        else 0.0
    )

    return_pct = net_pnl / starting_capital * 100.0

    max_drawdown, max_drawdown_pct = calculate_max_drawdown(
        net_pnl_series,
        starting_capital,
    )

    max_consecutive_losses = calculate_max_consecutive_losses(
        net_pnl_series
    )

    daily_pnl = pd.DataFrame(
    [
        (pd.Timestamp(trade.exit_time).date(), float(trade.net_pnl))
        for trade in trades
    ],
    columns=["date", "pnl"],
    )

    daily_totals = daily_pnl.groupby("date")["pnl"].sum()

    average_daily_pnl = (
        float(daily_totals.mean())
        if not daily_totals.empty
        else 0.0
    )

    return BacktestMetrics(
        starting_capital=starting_capital,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        breakeven_trades=breakeven_trades,
        win_rate=win_rate,
        gross_pnl=gross_pnl,
        total_costs=total_costs,
        net_pnl=net_pnl,
        return_pct=return_pct,
        average_winner=average_winner,
        average_loser=average_loser,
        expectancy_per_trade=expectancy_per_trade,
        payoff_ratio=payoff_ratio,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        max_consecutive_losses=max_consecutive_losses,
        average_daily_pnl=average_daily_pnl,
    )