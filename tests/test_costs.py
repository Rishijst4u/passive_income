import pytest

from src.costs import (
    CostSettings,
    calculate_brokerage,
    calculate_trade_costs,
)


def test_brokerage_uses_percentage_below_cap():
    brokerage = calculate_brokerage(10_000)

    assert brokerage == pytest.approx(3.0)


def test_brokerage_is_capped_at_twenty_rupees():
    brokerage = calculate_brokerage(100_000)

    assert brokerage == pytest.approx(20.0)


def test_brokerage_is_calculated_per_order():
    buy = calculate_brokerage(10_000)
    sell = calculate_brokerage(10_200)

    assert buy == pytest.approx(3.0)
    assert sell == pytest.approx(3.06)


def test_trade_costs_calculate_turnover():
    costs = calculate_trade_costs(
        entry_price=100,
        exit_price=102,
        quantity=100,
    )

    assert costs.buy_turnover == pytest.approx(10_000)
    assert costs.sell_turnover == pytest.approx(10_200)
    assert costs.total_turnover == pytest.approx(20_200)


def test_stt_is_only_on_sell_turnover():
    costs = calculate_trade_costs(
        entry_price=100,
        exit_price=102,
        quantity=100,
        settings=CostSettings(slippage_bps=0),
    )

    assert costs.stt == pytest.approx(
        10_200 * 0.00025
    )


def test_stamp_duty_is_only_on_buy_turnover():
    costs = calculate_trade_costs(
        entry_price=100,
        exit_price=102,
        quantity=100,
        settings=CostSettings(slippage_bps=0),
    )

    assert costs.stamp_duty == pytest.approx(
        10_000 * 0.00003
    )


def test_gross_pnl_is_correct():
    costs = calculate_trade_costs(
        entry_price=100,
        exit_price=102,
        quantity=100,
    )

    assert costs.gross_pnl == pytest.approx(200.0)


def test_slippage_can_be_disabled():
    costs = calculate_trade_costs(
        entry_price=100,
        exit_price=102,
        quantity=100,
        settings=CostSettings(slippage_bps=0),
    )

    assert costs.slippage == pytest.approx(0.0)


def test_slippage_is_based_on_round_trip_turnover():
    costs = calculate_trade_costs(
        entry_price=100,
        exit_price=102,
        quantity=100,
        settings=CostSettings(slippage_bps=5),
    )

    expected = 20_200 * 5 / 10_000

    assert costs.slippage == pytest.approx(expected)


def test_net_pnl_is_gross_pnl_minus_total_cost():
    costs = calculate_trade_costs(
        entry_price=100,
        exit_price=102,
        quantity=100,
    )

    assert costs.net_pnl == pytest.approx(
        costs.gross_pnl - costs.total_cost
    )


def test_invalid_entry_price_is_rejected():
    with pytest.raises(ValueError):
        calculate_trade_costs(
            entry_price=0,
            exit_price=102,
            quantity=100,
        )


def test_invalid_exit_price_is_rejected():
    with pytest.raises(ValueError):
        calculate_trade_costs(
            entry_price=100,
            exit_price=0,
            quantity=100,
        )


def test_invalid_quantity_is_rejected():
    with pytest.raises(ValueError):
        calculate_trade_costs(
            entry_price=100,
            exit_price=102,
            quantity=0,
        )