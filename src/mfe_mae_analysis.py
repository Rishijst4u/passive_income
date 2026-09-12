"""MFE/MAE management analysis for Strategy #002.

This module analyzes completed portfolio trades to determine whether losing
trades were:

1. Immediate failures that never moved meaningfully in our favor, or
2. Trades that reached a meaningful favorable excursion before reversing.

The analysis is diagnostic only. It does not modify strategy parameters,
entry rules, stop rules, or target rules.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TRADES_FILE = Path("data") / "reports" / "portfolio_trades.csv"

STRATEGY_NAME = "Trend + VWAP Pullback + Confirmation v2"

MFE_BUCKETS = (
    ("< 0R", float("-inf"), 0.0),
    ("0–0.25R", 0.0, 0.25),
    ("0.25–0.50R", 0.25, 0.50),
    ("0.50–0.75R", 0.50, 0.75),
    ("0.75–1.00R", 0.75, 1.00),
    ("1.00–1.50R", 1.00, 1.50),
    ("1.50–2.00R", 1.50, 2.00),
    (">= 2R", 2.00, float("inf")),
)


def load_trades(path: Path = TRADES_FILE) -> pd.DataFrame:
    """Load the completed portfolio trades report."""

    if not path.exists():
        raise FileNotFoundError(
            f"Trade report not found: {path}. "
            "Run the portfolio backtest first."
        )

    trades = pd.read_csv(path)

    if trades.empty:
        raise ValueError(f"Trade report is empty: {path}")

    required_columns = {
        "symbol",
        "entry_time",
        "entry",
        "stop",
        "exit",
        "exit_time",
        "reason",
        "net_pnl",
    }

    missing = required_columns.difference(trades.columns)
    if missing:
        raise ValueError(
            "Trade report is missing required columns: "
            + ", ".join(sorted(missing))
        )

    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"],
        errors="coerce",
    )
    trades["exit_time"] = pd.to_datetime(
        trades["exit_time"],
        errors="coerce",
    )

    numeric_columns = [
        "entry",
        "stop",
        "exit",
        "net_pnl",
    ]

    for column in numeric_columns:
        trades[column] = pd.to_numeric(
            trades[column],
            errors="coerce",
        )

    if "quantity" in trades.columns:
        trades["quantity"] = pd.to_numeric(
            trades["quantity"],
            errors="coerce",
        )

    trades = trades.dropna(
        subset=[
            "entry",
            "stop",
            "exit",
            "entry_time",
            "exit_time",
        ]
    ).copy()

    return trades


def _calculate_risk_per_share(trades: pd.DataFrame) -> pd.Series:
    """Calculate initial risk per share."""

    return trades["entry"] - trades["stop"]


def _calculate_net_r_multiple(trades: pd.DataFrame) -> pd.Series:
    """Calculate net P&L in units of initial planned risk."""

    if "quantity" not in trades.columns:
        return pd.Series(
            [float("nan")] * len(trades),
            index=trades.index,
        )

    risk_per_share = _calculate_risk_per_share(trades)
    planned_risk = risk_per_share * trades["quantity"]

    return trades["net_pnl"] / planned_risk.replace(0, pd.NA)


def _calculate_mfe_r(trades: pd.DataFrame) -> pd.Series:
    """Return existing MFE in R when available.

    The portfolio diagnostics pipeline normally stores MFE information in
    its diagnostic output. The primary trade report may not contain it.

    If an MFE column is present, it is used directly. Otherwise this function
    returns NaN because MFE cannot be reconstructed from entry/exit alone.
    """

    for column in ("mfe_r", "MFE_R", "mfe"):
        if column in trades.columns:
            return pd.to_numeric(
                trades[column],
                errors="coerce",
            )

    return pd.Series(
        [float("nan")] * len(trades),
        index=trades.index,
    )


def _classify_mfe(value: float) -> str:
    """Assign an MFE value to a diagnostic bucket."""

    if pd.isna(value):
        return "UNKNOWN"

    for label, lower, upper in MFE_BUCKETS:
        if lower == float("-inf") and value < upper:
            return label

        if lower == float("inf"):
            return label

        if lower <= value < upper:
            return label

    return "UNKNOWN"


def _prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Prepare trade-level calculations for analysis."""

    result = trades.copy()

    result["risk_per_share"] = _calculate_risk_per_share(result)

    if "quantity" in result.columns:
        result["planned_risk"] = (
            result["risk_per_share"] * result["quantity"]
        )
    else:
        result["planned_risk"] = float("nan")

    result["net_r_multiple"] = _calculate_net_r_multiple(result)

    result["mfe_r"] = _calculate_mfe_r(result)

    result["mfe_bucket"] = result["mfe_r"].apply(_classify_mfe)

    result["holding_minutes"] = (
        result["exit_time"] - result["entry_time"]
    ).dt.total_seconds() / 60.0

    return result


def _print_header(title: str) -> None:
    """Print a consistent section header."""

    print()
    print(title)
    print("-" * len(title))


def _print_mfe_distribution(
    trades: pd.DataFrame,
    title: str,
) -> None:
    """Print MFE distribution for a trade subset."""

    _print_header(title)

    total = len(trades)

    print(f"Trades: {total}")

    if total == 0:
        return

    known_mfe = trades["mfe_r"].dropna()

    if known_mfe.empty:
        print("MFE data: unavailable in portfolio_trades.csv")
        return

    print(f"Trades with MFE: {len(known_mfe)}")

    for label, _, _ in MFE_BUCKETS:
        count = int(
            (trades["mfe_bucket"] == label).sum()
        )

        percentage = (
            count / total * 100.0
            if total
            else 0.0
        )

        print(
            f"{label:<12} : "
            f"{count:>3} trades "
            f"({percentage:>6.2f}%)"
        )


def _print_threshold_analysis(
    losing_trades: pd.DataFrame,
) -> None:
    """Analyze how many losing trades reached important MFE thresholds."""

    _print_header("LOSING TRADES — MFE THRESHOLD ANALYSIS")

    total = len(losing_trades)

    if total == 0:
        print("No losing trades.")
        return

    known = losing_trades["mfe_r"].dropna()

    if known.empty:
        print("MFE data unavailable in portfolio_trades.csv.")
        print(
            "Run the portfolio diagnostics that generates MFE/MAE "
            "data before using threshold analysis."
        )
        return

    thresholds = (
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        2.00,
    )

    for threshold in thresholds:
        count = int(
            (known >= threshold).sum()
        )

        percentage = count / len(known) * 100.0

        print(
            f"Reached +{threshold:.2f}R"
            f"{' ' * max(1, 8 - len(f'{threshold:.2f}'))}"
            f": {count:>3} / {len(known)} "
            f"({percentage:>6.2f}%)"
        )


def _print_failure_classification(
    losing_trades: pd.DataFrame,
) -> None:
    """Classify losing trades into immediate failures and reversal candidates."""

    _print_header("LOSING TRADE CLASSIFICATION")

    total = len(losing_trades)

    if total == 0:
        print("No losing trades.")
        return

    known = losing_trades["mfe_r"].dropna()

    if known.empty:
        print("MFE data unavailable.")
        return

    categories = {
        "Immediate failure (< +0.50R)": (
            known < 0.50
        ),
        "Meaningful move (>= +0.50R)": (
            known >= 0.50
        ),
        "Strong move (>= +0.75R)": (
            known >= 0.75
        ),
        "Near-target reversal (>= +1.00R)": (
            known >= 1.00
        ),
        "Target-zone reversal (>= +1.50R)": (
            known >= 1.50
        ),
    }

    for label, mask in categories.items():
        count = int(mask.sum())
        percentage = count / len(known) * 100.0

        print(
            f"{label:<38} : "
            f"{count:>3} / {len(known)} "
            f"({percentage:>6.2f}%)"
        )


def _print_stop_vs_target_summary(
    trades: pd.DataFrame,
) -> None:
    """Compare stop and target trades using available MFE information."""

    _print_header("STOP vs TARGET — MFE SUMMARY")

    grouped = []

    for reason in ("STOP", "TARGET"):
        subset = trades[
            trades["reason"].astype(str).str.upper() == reason
        ]

        if subset.empty:
            continue

        known = subset["mfe_r"].dropna()

        grouped.append(
            {
                "reason": reason,
                "trades": len(subset),
                "mfe_available": len(known),
                "avg_mfe": (
                    float(known.mean())
                    if not known.empty
                    else float("nan")
                ),
                "median_mfe": (
                    float(known.median())
                    if not known.empty
                    else float("nan")
                ),
                "avg_net_r": (
                    float(subset["net_r_multiple"].mean())
                    if subset["net_r_multiple"].notna().any()
                    else float("nan")
                ),
            }
        )

    if not grouped:
        print("No STOP/TARGET trades found.")
        return

    for row in grouped:
        print(
            f"{row['reason']:<8} | "
            f"Trades {row['trades']:>3} | "
            f"MFE {row['mfe_available']:>3} | "
            f"Avg MFE "
            f"{row['avg_mfe']:.2f}R "
            f"| Median MFE "
            f"{row['median_mfe']:.2f}R "
            f"| Avg Net R "
            f"{row['avg_net_r']:.2f}R"
        )


def _print_high_mfe_losers(
    losing_trades: pd.DataFrame,
) -> None:
    """Print losing trades that moved substantially in our favor."""

    _print_header("LOSING TRADES THAT REACHED +0.75R OR MORE")

    subset = losing_trades[
        losing_trades["mfe_r"].notna()
        & (losing_trades["mfe_r"] >= 0.75)
    ].copy()

    if subset.empty:
        print("None.")
        return

    columns = [
        "symbol",
        "entry_time",
        "exit_time",
        "mfe_r",
        "net_r_multiple",
        "net_pnl",
        "reason",
    ]

    available = [
        column
        for column in columns
        if column in subset.columns
    ]

    display = subset[available].sort_values(
        "mfe_r",
        ascending=False,
    )

    for _, row in display.iterrows():
        symbol = str(row["symbol"])

        entry_time = str(row["entry_time"])

        mfe = row["mfe_r"]

        net_r = row["net_r_multiple"]

        net_pnl = row["net_pnl"]

        print(
            f"{symbol:<12} | "
            f"Entry {entry_time:<20} | "
            f"MFE {mfe:>5.2f}R | "
            f"Net R {net_r:>6.2f}R | "
            f"Net ₹ {net_pnl:>8.2f}"
        )


def _print_conclusion(
    losing_trades: pd.DataFrame,
) -> None:
    """Print a diagnostic interpretation without changing the strategy."""

    _print_header("DIAGNOSTIC INTERPRETATION")

    known = losing_trades["mfe_r"].dropna()

    if known.empty:
        print(
            "MFE data is not present in portfolio_trades.csv, so "
            "entry-vs-management classification cannot yet be completed."
        )
        return

    total = len(known)

    immediate = int((known < 0.50).sum())
    meaningful = int((known >= 0.50).sum())
    strong = int((known >= 0.75).sum())
    near_target = int((known >= 1.00).sum())

    immediate_pct = immediate / total * 100.0
    meaningful_pct = meaningful / total * 100.0
    strong_pct = strong / total * 100.0
    near_target_pct = near_target / total * 100.0

    print(
        f"Losing trades analyzed with MFE : {total}"
    )

    print(
        f"Immediate failures (<0.50R)     : "
        f"{immediate} ({immediate_pct:.2f}%)"
    )

    print(
        f"Reached >=0.50R before losing    : "
        f"{meaningful} ({meaningful_pct:.2f}%)"
    )

    print(
        f"Reached >=0.75R before losing    : "
        f"{strong} ({strong_pct:.2f}%)"
    )

    print(
        f"Reached >=1.00R before losing    : "
        f"{near_target} ({near_target_pct:.2f}%)"
    )

    print()

    if meaningful_pct < 20.0:
        print(
            "Initial interpretation: most losing trades were immediate "
            "failures. Entry quality is likely the primary issue."
        )
    elif meaningful_pct < 40.0:
        print(
            "Initial interpretation: losses are mixed between poor entries "
            "and trades that achieved some favorable movement."
        )
    else:
        print(
            "Initial interpretation: a substantial portion of losing trades "
            "moved favorably before reversing. Trade management deserves "
            "controlled investigation."
        )

    if strong_pct >= 25.0:
        print(
            "Important: many losing trades reached +0.75R or more. "
            "Do not change the entry strategy until management has been "
            "tested separately."
        )

    if near_target_pct >= 15.0:
        print(
            "Important: a meaningful number of losers reached +1R or more. "
            "A fixed 2R target may be leaving useful information on the table, "
            "but this must be tested rather than assumed."
        )

    print()
    print(
        "No strategy parameters were changed by this analysis."
    )


def analyze(
    trades_path: Path = TRADES_FILE,
) -> pd.DataFrame:
    """Run the complete MFE/MAE management analysis."""

    trades = load_trades(trades_path)
    prepared = _prepare_trades(trades)

    reason = prepared["reason"].astype(str).str.upper()

    losing_trades = prepared[
        reason.isin(
            {
                "STOP",
                "STOP_TARGET_AMBIGUITY_STOP",
            }
        )
    ].copy()

    winning_trades = prepared[
        reason.eq("TARGET")
    ].copy()

    print()
    print("=" * 72)
    print("STRATEGY #002 — MFE/MAE MANAGEMENT ANALYSIS")
    print("=" * 72)

    print(f"Strategy              : {STRATEGY_NAME}")
    print(f"Trades analyzed       : {len(prepared)}")
    print(f"Stop trades           : {len(losing_trades)}")
    print(f"Target trades         : {len(winning_trades)}")

    mfe_available = int(
        prepared["mfe_r"].notna().sum()
    )

    print(
        f"MFE records available : "
        f"{mfe_available}/{len(prepared)}"
    )

    _print_mfe_distribution(
        losing_trades,
        "LOSING TRADES — MAX MFE DISTRIBUTION",
    )

    _print_mfe_distribution(
        winning_trades,
        "TARGET TRADES — MAX MFE DISTRIBUTION",
    )

    _print_threshold_analysis(losing_trades)

    _print_failure_classification(losing_trades)

    _print_stop_vs_target_summary(prepared)

    _print_high_mfe_losers(losing_trades)

    _print_conclusion(losing_trades)

    print()
    print("=" * 72)

    return prepared


def main() -> None:
    """CLI entry point."""

    try:
        analyze()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()