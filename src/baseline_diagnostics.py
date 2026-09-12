"""Diagnostic analysis for the Experiment #001 baseline backtest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPORT_ROOT = Path("data") / "reports"
TRADES_FILE = REPORT_ROOT / "baseline_trades.csv"


REQUIRED_COLUMNS = {
    "symbol",
    "signal_time",
    "entry_time",
    "entry",
    "stop",
    "target",
    "quantity",
    "exit_time",
    "exit",
    "gross_pnl",
    "total_cost",
    "net_pnl",
    "reason",
}


def _prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Validate and add all derived diagnostic columns.

    This function is intentionally safe to call on either:
    - the raw baseline trade DataFrame, or
    - a DataFrame already enriched by this module.
    """
    if trades.empty:
        return trades.copy()

    frame = trades.copy()

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            "Trade DataFrame is missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame["signal_time"] = pd.to_datetime(
        frame["signal_time"],
        errors="coerce",
    )
    frame["entry_time"] = pd.to_datetime(
        frame["entry_time"],
        errors="coerce",
    )
    frame["exit_time"] = pd.to_datetime(
        frame["exit_time"],
        errors="coerce",
    )

    frame["entry_date"] = frame["entry_time"].dt.date
    frame["entry_clock"] = frame["entry_time"].dt.strftime("%H:%M")

    frame["holding_minutes"] = (
        frame["exit_time"] - frame["entry_time"]
    ).dt.total_seconds() / 60.0

    frame["risk_per_share"] = (
        frame["entry"].astype(float)
        - frame["stop"].astype(float)
    )

    frame["planned_risk"] = (
        frame["risk_per_share"]
        * frame["quantity"].astype(float)
    )

    frame["r_multiple"] = (
        frame["gross_pnl"].astype(float)
        / frame["planned_risk"].replace(0, pd.NA)
    )

    frame["is_winner"] = frame["net_pnl"].astype(float) > 0
    frame["is_loser"] = frame["net_pnl"].astype(float) < 0

    return frame


def load_trades(path: Path = TRADES_FILE) -> pd.DataFrame:
    """Load baseline trades from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Trade report not found: {path}")

    trades = pd.read_csv(path)

    return _prepare_trades(trades)


def calculate_overall_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    """Return high-level diagnostic metrics."""
    frame = _prepare_trades(trades)

    if frame.empty:
        return pd.DataFrame(columns=["metric", "value"])

    total_trades = len(frame)

    winners = frame.loc[
        frame["net_pnl"] > 0,
        "net_pnl",
    ]

    losers = frame.loc[
        frame["net_pnl"] < 0,
        "net_pnl",
    ]

    gross_profit = float(frame["gross_pnl"].sum())
    total_cost = float(frame["total_cost"].sum())
    net_profit = float(frame["net_pnl"].sum())

    gross_wins = float(
        frame.loc[
            frame["gross_pnl"] > 0,
            "gross_pnl",
        ].sum()
    )

    gross_losses = float(
        frame.loc[
            frame["gross_pnl"] < 0,
            "gross_pnl",
        ].sum()
    )

    profit_factor = (
        gross_wins / abs(gross_losses)
        if gross_losses < 0
        else float("inf")
    )

    expectancy = (
        net_profit / total_trades
        if total_trades
        else 0.0
    )

    win_rate = (
        len(winners) / total_trades * 100.0
        if total_trades
        else 0.0
    )

    avg_winner = (
        float(winners.mean())
        if not winners.empty
        else 0.0
    )

    avg_loser = (
        float(losers.mean())
        if not losers.empty
        else 0.0
    )

    payoff_ratio = (
        avg_winner / abs(avg_loser)
        if avg_loser < 0
        else float("inf")
    )

    avg_cost = (
        total_cost / total_trades
        if total_trades
        else 0.0
    )

    absolute_gross_pnl = float(
        frame["gross_pnl"].abs().sum()
    )

    cost_pct_gross_pnl = (
        total_cost / absolute_gross_pnl * 100.0
        if absolute_gross_pnl > 0
        else 0.0
    )

    return pd.DataFrame(
        [
            ("total_trades", total_trades),
            ("winning_trades", len(winners)),
            ("losing_trades", len(losers)),
            ("win_rate_pct", win_rate),
            ("gross_pnl", gross_profit),
            ("total_cost", total_cost),
            ("net_pnl", net_profit),
            ("expectancy_per_trade", expectancy),
            ("average_winner", avg_winner),
            ("average_loser", avg_loser),
            ("payoff_ratio", payoff_ratio),
            ("gross_profit", gross_wins),
            ("gross_loss", gross_losses),
            ("profit_factor", profit_factor),
            ("average_cost_per_trade", avg_cost),
            (
                "cost_vs_absolute_gross_pnl_pct",
                cost_pct_gross_pnl,
            ),
            (
                "average_holding_minutes",
                float(frame["holding_minutes"].mean()),
            ),
            (
                "average_r_multiple",
                float(frame["r_multiple"].mean()),
            ),
        ],
        columns=["metric", "value"],
    )


def analyze_exit_reasons(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze trade outcomes by exit reason."""
    frame = _prepare_trades(trades)

    if frame.empty:
        return pd.DataFrame()

    grouped = (
        frame.groupby(
            "reason",
            dropna=False,
        )
        .agg(
            trades=("symbol", "size"),
            wins=("is_winner", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            total_cost=("total_cost", "sum"),
            net_pnl=("net_pnl", "sum"),
            avg_net_pnl=("net_pnl", "mean"),
            avg_holding_minutes=(
                "holding_minutes",
                "mean",
            ),
            avg_r_multiple=(
                "r_multiple",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["win_rate_pct"] = (
        grouped["wins"]
        / grouped["trades"]
        * 100.0
    )

    return grouped[
        [
            "reason",
            "trades",
            "wins",
            "win_rate_pct",
            "gross_pnl",
            "total_cost",
            "net_pnl",
            "avg_net_pnl",
            "avg_holding_minutes",
            "avg_r_multiple",
        ]
    ].sort_values("net_pnl")


def analyze_by_symbol(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by stock."""
    frame = _prepare_trades(trades)

    if frame.empty:
        return pd.DataFrame()

    grouped = (
        frame.groupby("symbol")
        .agg(
            trades=("symbol", "size"),
            wins=("is_winner", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            total_cost=("total_cost", "sum"),
            net_pnl=("net_pnl", "sum"),
            avg_net_pnl=("net_pnl", "mean"),
            avg_r_multiple=(
                "r_multiple",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["win_rate_pct"] = (
        grouped["wins"]
        / grouped["trades"]
        * 100.0
    )

    return grouped[
        [
            "symbol",
            "trades",
            "wins",
            "win_rate_pct",
            "gross_pnl",
            "total_cost",
            "net_pnl",
            "avg_net_pnl",
            "avg_r_multiple",
        ]
    ].sort_values("net_pnl")


def analyze_by_day(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by trading day."""
    frame = _prepare_trades(trades)

    if frame.empty:
        return pd.DataFrame()

    grouped = (
        frame.groupby("entry_date")
        .agg(
            trades=("symbol", "size"),
            wins=("is_winner", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            total_cost=("total_cost", "sum"),
            net_pnl=("net_pnl", "sum"),
        )
        .reset_index()
    )

    grouped["win_rate_pct"] = (
        grouped["wins"]
        / grouped["trades"]
        * 100.0
    )

    return grouped[
        [
            "entry_date",
            "trades",
            "wins",
            "win_rate_pct",
            "gross_pnl",
            "total_cost",
            "net_pnl",
        ]
    ].sort_values("entry_date")


def analyze_by_entry_time(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by entry clock time."""
    frame = _prepare_trades(trades)

    if frame.empty:
        return pd.DataFrame()

    grouped = (
        frame.groupby("entry_clock")
        .agg(
            trades=("symbol", "size"),
            wins=("is_winner", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            total_cost=("total_cost", "sum"),
            net_pnl=("net_pnl", "sum"),
            avg_holding_minutes=(
                "holding_minutes",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped["win_rate_pct"] = (
        grouped["wins"]
        / grouped["trades"]
        * 100.0
    )

    return grouped[
        [
            "entry_clock",
            "trades",
            "wins",
            "win_rate_pct",
            "gross_pnl",
            "total_cost",
            "net_pnl",
            "avg_holding_minutes",
        ]
    ].sort_values("entry_clock")


def analyze_stop_distance(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze trades by initial planned risk."""
    frame = _prepare_trades(trades)

    if frame.empty:
        return pd.DataFrame()

    frame["risk_pct_of_entry"] = (
        frame["risk_per_share"]
        / frame["entry"].astype(float)
        * 100.0
    )

    return frame[
        [
            "symbol",
            "entry_time",
            "entry",
            "stop",
            "target",
            "quantity",
            "risk_per_share",
            "risk_pct_of_entry",
            "planned_risk",
            "gross_pnl",
            "net_pnl",
            "r_multiple",
            "reason",
        ]
    ].sort_values("risk_pct_of_entry")


def write_reports(trades: pd.DataFrame) -> dict[str, Path]:
    """Generate diagnostic CSV reports."""
    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "overall": (
            REPORT_ROOT
            / "baseline_diagnostics_overall.csv"
        ),
        "exit_reason": (
            REPORT_ROOT
            / "baseline_diagnostics_exit_reason.csv"
        ),
        "symbol": (
            REPORT_ROOT
            / "baseline_diagnostics_symbol.csv"
        ),
        "day": (
            REPORT_ROOT
            / "baseline_diagnostics_day.csv"
        ),
        "entry_time": (
            REPORT_ROOT
            / "baseline_diagnostics_entry_time.csv"
        ),
        "risk": (
            REPORT_ROOT
            / "baseline_diagnostics_risk.csv"
        ),
    }

    calculate_overall_metrics(trades).to_csv(
        outputs["overall"],
        index=False,
    )

    analyze_exit_reasons(trades).to_csv(
        outputs["exit_reason"],
        index=False,
    )

    analyze_by_symbol(trades).to_csv(
        outputs["symbol"],
        index=False,
    )

    analyze_by_day(trades).to_csv(
        outputs["day"],
        index=False,
    )

    analyze_by_entry_time(trades).to_csv(
        outputs["entry_time"],
        index=False,
    )

    analyze_stop_distance(trades).to_csv(
        outputs["risk"],
        index=False,
    )

    return outputs


def print_summary(trades: pd.DataFrame) -> None:
    """Print concise diagnostic summary."""
    frame = _prepare_trades(trades)

    if frame.empty:
        print("No trades found.")
        return

    metrics = calculate_overall_metrics(frame)

    metric_map = dict(
        zip(
            metrics["metric"],
            metrics["value"],
        )
    )

    print()
    print("=" * 72)
    print("EXPERIMENT #001 — BASELINE DIAGNOSTICS")
    print("=" * 72)

    print(
        f"Trades                 : "
        f"{int(metric_map['total_trades'])}"
    )

    print(
        f"Win rate               : "
        f"{metric_map['win_rate_pct']:.2f}%"
    )

    print(
        f"Gross P&L              : "
        f"₹{metric_map['gross_pnl']:,.2f}"
    )

    print(
        f"Trading costs          : "
        f"₹{metric_map['total_cost']:,.2f}"
    )

    print(
        f"Net P&L                : "
        f"₹{metric_map['net_pnl']:,.2f}"
    )

    print(
        f"Expectancy/trade       : "
        f"₹{metric_map['expectancy_per_trade']:,.2f}"
    )

    print(
        f"Average winner         : "
        f"₹{metric_map['average_winner']:,.2f}"
    )

    print(
        f"Average loser          : "
        f"₹{metric_map['average_loser']:,.2f}"
    )

    print(
        f"Payoff ratio           : "
        f"{metric_map['payoff_ratio']:.3f}"
    )

    print(
        f"Profit factor          : "
        f"{metric_map['profit_factor']:.3f}"
    )

    print(
        f"Average cost/trade     : "
        f"₹{metric_map['average_cost_per_trade']:,.2f}"
    )

    print(
        f"Average holding time   : "
        f"{metric_map['average_holding_minutes']:.1f} min"
    )

    print(
        f"Average R multiple     : "
        f"{metric_map['average_r_multiple']:.3f}"
    )

    print()
    print("EXIT REASONS")
    print("-" * 72)

    exit_reason = analyze_exit_reasons(frame)

    for _, row in exit_reason.iterrows():
        print(
            f"{str(row['reason']):<32} "
            f"{int(row['trades']):>4} trades | "
            f"win {row['win_rate_pct']:>6.2f}% | "
            f"net ₹{row['net_pnl']:>9,.2f}"
        )

    print()
    print("WORST STOCKS")
    print("-" * 72)

    by_symbol = analyze_by_symbol(frame)

    for _, row in by_symbol.head(10).iterrows():
        print(
            f"{row['symbol']:<15} "
            f"{int(row['trades']):>3} trades | "
            f"win {row['win_rate_pct']:>6.2f}% | "
            f"net ₹{row['net_pnl']:>9,.2f}"
        )

    print()
    print("BEST STOCKS")
    print("-" * 72)

    for _, row in by_symbol.tail(10).sort_values(
        "net_pnl",
        ascending=False,
    ).iterrows():
        print(
            f"{row['symbol']:<15} "
            f"{int(row['trades']):>3} trades | "
            f"win {row['win_rate_pct']:>6.2f}% | "
            f"net ₹{row['net_pnl']:>9,.2f}"
        )

    print("=" * 72)


def main() -> None:
    """Run baseline diagnostics."""
    trades = load_trades()

    outputs = write_reports(trades)

    print_summary(trades)

    print()
    print("Diagnostic reports written:")

    for name, path in outputs.items():
        print(f"  {name:<12}: {path}")


if __name__ == "__main__":
    main()