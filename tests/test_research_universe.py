from src.research_universe import (
    RESEARCH_UNIVERSE,
    get_research_universe,
)


def test_research_universe_has_20_stocks():
    universe = get_research_universe()

    assert len(universe) == 20


def test_research_universe_contains_unique_symbols():
    universe = get_research_universe()

    assert len(universe) == len(set(universe))


def test_research_universe_contains_expected_core_stocks():
    universe = get_research_universe()

    expected = {
        "RELIANCE",
        "HDFCBANK",
        "ICICIBANK",
        "SBIN",
        "INFY",
        "TCS",
        "LT",
        "BHARTIARTL",
    }

    assert expected.issubset(set(universe))


def test_research_universe_constant_matches_function():
    assert get_research_universe() == RESEARCH_UNIVERSE