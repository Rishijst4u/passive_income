"""Initial research universe for Experiment #001."""

from __future__ import annotations

RESEARCH_UNIVERSE: tuple[str, ...] = (
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "AXISBANK",
    "KOTAKBANK",
    "INFY",
    "TCS",
    "WIPRO",
    "HCLTECH",
    "LT",
    "BHARTIARTL",
    "ITC",
    "HINDUNILVR",
    "MARUTI",
    "M&M",
    "SUNPHARMA",
    "TITAN",
    "ADANIENT",
    "BAJFINANCE",
)


def get_research_universe() -> tuple[str, ...]:
    """Return the initial Experiment #001 research universe."""

    return RESEARCH_UNIVERSE