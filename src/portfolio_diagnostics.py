"""Detailed diagnostics for Experiment #001 portfolio backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .market_data import ASIA_KOLKATA


@dataclass(frozen=True)
class DiagnosticConfig:
    """Configuration for portfolio diagnostic reports."""

    data_root: Path = Path("data/processed")
    reports_root: Path = Path("data/reports")
    trades_file: str = "portfolio_trades.csv"
    rejections_file: str = "portfolio_rejections.csv"


def _to_datetime_series(series: pd.Series) -> pd.Series:
    """Convert timestamps to timezone-aware Asia/Kolkata timestamps."""

    result = pd.to_datetime(series, errors="coerce", utc=True)
    return result.dt.tz_convert(ASIA_KOLKATA)


def _prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Prepare portfolio trades with derived diagnostic fields."""

    frame = trades.copy()

    if frame.empty:
        for column in (
            "risk_per_share",
            "planned_risk",
            "position_value",
            "holding_minutes",
            "gross_r_multiple",
            "net_r_multiple",
        ):
            frame[column] = pd.Series(dtype=float)
        return frame

    # CSV reports contain timestamps as strings. Normalize them before
    # performing datetime arithmetic.
    for column in ("signal_time", "entry_time", "exit_time"):
        if column in frame.columns:
            frame[column] = _to_datetime_series(frame[column])

    numeric_columns = (
        "entry",
        "stop",
        "target",
        "quantity",
        "exit",
        "gross_pnl",
        "total_cost",
        "net_pnl",
    )

    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    frame["risk_per_share"] = frame["entry"] - frame["stop"]
    frame["planned_risk"] = (
        frame["risk_per_share"] * frame["quantity"]
    )
    frame["position_value"] = (
        frame["entry"] * frame["quantity"]
    )

    frame["holding_minutes"] = (
        frame["exit_time"] - frame["entry_time"]
    ).dt.total_seconds() / 60.0

    frame["gross_r_multiple"] = (
        frame["gross_pnl"] / frame["planned_risk"]
    )

    frame["net_r_multiple"] = (
        frame["net_pnl"] / frame["planned_risk"]
    )

    return frame


def analyze_by_symbol(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by stock symbol."""

    if trades.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "trades",
                "wins",
                "losses",
                "win_rate",
                "gross_pnl",
                "net_pnl",
                "avg_net_pnl",
                "avg_r_multiple",
            ]
        )

    grouped = trades.groupby("symbol", sort=True)

    result = grouped.agg(
        trades=("symbol", "size"),
        wins=("net_pnl", lambda x: int((x > 0).sum())),
        losses=("net_pnl", lambda x: int((x <= 0).sum())),
        gross_pnl=("gross_pnl", "sum"),
        net_pnl=("net_pnl", "sum"),
        avg_net_pnl=("net_pnl", "mean"),
        avg_r_multiple=("net_r_multiple", "mean"),
    ).reset_index()

    result["win_rate"] = (
        result["wins"] / result["trades"] * 100.0
    )

    return result[
        [
            "symbol",
            "trades",
            "wins",
            "losses",
            "win_rate",
            "gross_pnl",
            "net_pnl",
            "avg_net_pnl",
            "avg_r_multiple",
        ]
    ]


def analyze_by_exit_reason(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by exit reason."""

    if trades.empty:
        return pd.DataFrame(
            columns=[
                "reason",
                "trades",
                "wins",
                "losses",
                "win_rate",
                "gross_pnl",
                "net_pnl",
                "avg_net_pnl",
            ]
        )

    grouped = trades.groupby("reason", sort=True)

    result = grouped.agg(
        trades=("reason", "size"),
        wins=("net_pnl", lambda x: int((x > 0).sum())),
        losses=("net_pnl", lambda x: int((x <= 0).sum())),
        gross_pnl=("gross_pnl", "sum"),
        net_pnl=("net_pnl", "sum"),
        avg_net_pnl=("net_pnl", "mean"),
    ).reset_index()

    result["win_rate"] = (
        result["wins"] / result["trades"] * 100.0
    )

    return result[
        [
            "reason",
            "trades",
            "wins",
            "losses",
            "win_rate",
            "gross_pnl",
            "net_pnl",
            "avg_net_pnl",
        ]
    ]


def analyze_by_hour(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by entry hour."""

    if trades.empty:
        return pd.DataFrame(
            columns=[
                "entry_hour",
                "trades",
                "wins",
                "losses",
                "win_rate",
                "net_pnl",
                "avg_net_pnl",
            ]
        )

    frame = trades.copy()
    frame["entry_hour"] = frame["entry_time"].dt.hour

    grouped = frame.groupby("entry_hour", sort=True)

    result = grouped.agg(
        trades=("entry_hour", "size"),
        wins=("net_pnl", lambda x: int((x > 0).sum())),
        losses=("net_pnl", lambda x: int((x <= 0).sum())),
        net_pnl=("net_pnl", "sum"),
        avg_net_pnl=("net_pnl", "mean"),
    ).reset_index()

    result["win_rate"] = (
        result["wins"] / result["trades"] * 100.0
    )

    return result[
        [
            "entry_hour",
            "trades",
            "wins",
            "losses",
            "win_rate",
            "net_pnl",
            "avg_net_pnl",
        ]
    ]


def analyze_by_day_of_week(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by weekday."""

    if trades.empty:
        return pd.DataFrame(
            columns=[
                "day_of_week",
                "trades",
                "wins",
                "losses",
                "win_rate",
                "net_pnl",
                "avg_net_pnl",
            ]
        )

    frame = trades.copy()
    frame["day_of_week"] = frame["entry_time"].dt.day_name()

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
    ]

    grouped = frame.groupby("day_of_week", sort=False)

    result = grouped.agg(
        trades=("day_of_week", "size"),
        wins=("net_pnl", lambda x: int((x > 0).sum())),
        losses=("net_pnl", lambda x: int((x <= 0).sum())),
        net_pnl=("net_pnl", "sum"),
        avg_net_pnl=("net_pnl", "mean"),
    ).reset_index()

    result["win_rate"] = (
        result["wins"] / result["trades"] * 100.0
    )

    result["day_of_week"] = pd.Categorical(
        result["day_of_week"],
        categories=day_order,
        ordered=True,
    )

    return result.sort_values("day_of_week").reset_index(drop=True)[
        [
            "day_of_week",
            "trades",
            "wins",
            "losses",
            "win_rate",
            "net_pnl",
            "avg_net_pnl",
        ]
    ]


def analyze_rejections(rejections: pd.DataFrame) -> pd.DataFrame:
    """Analyze rejected signal reasons."""

    if rejections.empty:
        return pd.DataFrame(
            columns=["reason", "count", "percentage"]
        )

    result = (
        rejections.groupby("reason", sort=True)
        .size()
        .reset_index(name="count")
    )

    total = int(result["count"].sum())

    if total > 0:
        result["percentage"] = (
            result["count"] / total * 100.0
        )
    else:
        result["percentage"] = 0.0

    return result


def calculate_consecutive_losses(trades: pd.DataFrame) -> int:
    """Return the longest consecutive losing-trade streak."""

    if trades.empty:
        return 0

    ordered = trades.sort_values(
        ["exit_time", "entry_time"]
    ).reset_index(drop=True)

    current = 0
    maximum = 0

    for pnl in ordered["net_pnl"]:
        if pnl <= 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0

    return maximum


def create_trade_sequence(trades: pd.DataFrame) -> pd.DataFrame:
    """Create chronological trade sequence diagnostics."""

    if trades.empty:
        return pd.DataFrame(
            columns=[
                "trade_number",
                "symbol",
                "entry_time",
                "exit_time",
                "net_pnl",
                "loss_streak",
                "cumulative_net_pnl",
            ]
        )

    result = (
        trades.sort_values(
            ["entry_time", "exit_time", "symbol"]
        )
        .reset_index(drop=True)
        .copy()
    )

    result.insert(
        0,
        "trade_number",
        range(1, len(result) + 1),
    )

    loss_streak: list[int] = []
    current_streak = 0

    for pnl in result["net_pnl"]:
        if pnl <= 0:
            current_streak += 1
        else:
            current_streak = 0

        loss_streak.append(current_streak)

    result["loss_streak"] = loss_streak
    result["cumulative_net_pnl"] = result["net_pnl"].cumsum()

    return result


def _calculate_mae_mfe(
    trades: pd.DataFrame,
    market_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Calculate maximum adverse/favorable excursion for long trades."""

    if trades.empty:
        result = trades.copy()
        result["mae"] = pd.Series(dtype=float)
        result["mfe"] = pd.Series(dtype=float)
        return result

    result_rows: list[dict] = []

    for _, trade in trades.iterrows():
        symbol = trade["symbol"]
        entry = float(trade["entry"])
        exit_time = trade["exit_time"]
        entry_time = trade["entry_time"]

        data = market_data.get(symbol)

        mae = float("nan")
        mfe = float("nan")

        if data is not None and not data.empty:
            frame = data.copy()

            if "timestamp" in frame.columns:
                frame["timestamp"] = _to_datetime_series(
                    frame["timestamp"]
                )

                window = frame[
                    (frame["timestamp"] >= entry_time)
                    & (frame["timestamp"] <= exit_time)
                ]

                if not window.empty:
                    mae = float(
                        (window["low"].astype(float) - entry).min()
                    )
                    mfe = float(
                        (window["high"].astype(float) - entry).max()
                    )

        row = trade.to_dict()
        row["mae"] = mae
        row["mfe"] = mfe
        result_rows.append(row)

    return pd.DataFrame(result_rows)


def _load_report(
    path: Path,
) -> pd.DataFrame:
    """Load a CSV report if it exists."""

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def load_portfolio_reports(
    config: DiagnosticConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load portfolio trades and rejection reports."""

    config = config or DiagnosticConfig()

    trades_path = config.reports_root / config.trades_file
    rejections_path = config.reports_root / config.rejections_file

    return (
        _load_report(trades_path),
        _load_report(rejections_path),
    )


def summary_statistics(trades: pd.DataFrame) -> dict[str, float]:
    """Calculate high-level diagnostic statistics."""

    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "gross_pnl": 0.0,
            "total_cost": 0.0,
            "net_pnl": 0.0,
            "expectancy": 0.0,
            "average_winner": 0.0,
            "average_loser": 0.0,
            "payoff_ratio": 0.0,
            "profit_factor": 0.0,
            "average_cost": 0.0,
            "average_holding_minutes": 0.0,
            "average_r_multiple": 0.0,
        }

    wins = trades.loc[trades["net_pnl"] > 0, "net_pnl"]
    losses = trades.loc[trades["net_pnl"] <= 0, "net_pnl"]

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf")
    )

    average_winner = (
        float(wins.mean())
        if not wins.empty
        else 0.0
    )

    average_loser = (
        float(losses.mean())
        if not losses.empty
        else 0.0
    )

    payoff_ratio = (
        average_winner / abs(average_loser)
        if average_loser != 0
        else float("inf")
    )

    return {
        "trades": int(len(trades)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate": float(len(wins) / len(trades) * 100.0),
        "gross_pnl": float(trades["gross_pnl"].sum()),
        "total_cost": float(trades["total_cost"].sum()),
        "net_pnl": float(trades["net_pnl"].sum()),
        "expectancy": float(trades["net_pnl"].mean()),
        "average_winner": average_winner,
        "average_loser": average_loser,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
        "average_cost": float(trades["total_cost"].mean()),
        "average_holding_minutes": float(
            trades["holding_minutes"].mean()
        ),
        "average_r_multiple": float(
            trades["net_r_multiple"].mean()
        ),
    }


def print_report(
    trades: pd.DataFrame,
    rejections: pd.DataFrame,
) -> None:
    """Print detailed diagnostics to the console."""

    metrics = summary_statistics(trades)

    print()
    print("=" * 60)
    print("EXPERIMENT #001 — PORTFOLIO DIAGNOSTICS")
    print("=" * 60)

    print(f"Trades                 : {metrics['trades']}")
    print(f"Win rate               : {metrics['win_rate']:.2f}%")
    print(f"Gross P&L              : ₹{metrics['gross_pnl']:,.2f}")
    print(f"Trading costs          : ₹{metrics['total_cost']:,.2f}")
    print(f"Net P&L                : ₹{metrics['net_pnl']:,.2f}")
    print(f"Expectancy/trade       : ₹{metrics['expectancy']:,.2f}")
    print(f"Average winner         : ₹{metrics['average_winner']:,.2f}")
    print(f"Average loser          : ₹{metrics['average_loser']:,.2f}")
    print(f"Payoff ratio           : {metrics['payoff_ratio']:.3f}")
    print(f"Profit factor          : {metrics['profit_factor']:.3f}")
    print(f"Average cost/trade     : ₹{metrics['average_cost']:,.2f}")
    print(
        "Average holding time   : "
        f"{metrics['average_holding_minutes']:.1f} min"
    )
    print(
        "Average R multiple     : "
        f"{metrics['average_r_multiple']:.3f}"
    )

    print()
    print("EXIT REASONS")
    print("-" * 60)

    exit_analysis = analyze_by_exit_reason(trades)

    if exit_analysis.empty:
        print("No trades.")
    else:
        for _, row in exit_analysis.iterrows():
            print(
                f"{str(row['reason']):35s}"
                f"{int(row['trades']):5d} trades | "
                f"win {row['win_rate']:6.2f}% | "
                f"net ₹{row['net_pnl']:10.2f}"
            )

    print()
    print("SYMBOL PERFORMANCE")
    print("-" * 60)

    symbol_analysis = analyze_by_symbol(trades)

    if symbol_analysis.empty:
        print("No trades.")
    else:
        for _, row in symbol_analysis.iterrows():
            print(
                f"{row['symbol']:15s}"
                f"{int(row['trades']):4d} trades | "
                f"win {row['win_rate']:6.2f}% | "
                f"net ₹{row['net_pnl']:10.2f}"
            )

    print()
    print("REJECTION REASONS")
    print("-" * 60)

    rejection_analysis = analyze_rejections(rejections)

    if rejection_analysis.empty:
        print("No rejected signals.")
    else:
        for _, row in rejection_analysis.iterrows():
            print(
                f"{str(row['reason']):35s}"
                f"{int(row['count']):5d} | "
                f"{row['percentage']:6.2f}%"
            )

    print()
    print(
        "Maximum consecutive losses : "
        f"{calculate_consecutive_losses(trades)}"
    )
    print("=" * 60)


def run_diagnostics(
    config: DiagnosticConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load reports, prepare trades, print diagnostics, and return data."""

    config = config or DiagnosticConfig()

    raw_trades, rejections = load_portfolio_reports(config)
    trades = _prepare_trades(raw_trades)

    print_report(trades, rejections)

    return trades, rejections


if __name__ == "__main__":
    run_diagnostics()