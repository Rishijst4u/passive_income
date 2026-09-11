"""Trading cost model for NSE equity intraday trades."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CostSettings:
    """Configurable assumptions for NSE equity intraday trading."""

    brokerage_rate: float = 0.0003
    brokerage_max_per_order: float = 20.0

    stt_rate: float = 0.00025

    exchange_transaction_rate: float = 0.0000307

    sebi_rate: float = 0.000001

    stamp_duty_rate: float = 0.00003

    gst_rate: float = 0.18

    ipft_rate: float = 0.000000001

    slippage_bps: float = 5.0


@dataclass(frozen=True)
class TradingCosts:
    """Detailed cost breakdown for one completed trade."""

    buy_turnover: float
    sell_turnover: float
    total_turnover: float

    brokerage: float
    stt: float
    exchange_transaction_charges: float
    sebi_charges: float
    stamp_duty: float
    gst: float
    ipft: float

    slippage: float
    total_cost: float

    gross_pnl: float
    net_pnl: float


def calculate_brokerage(
    turnover: float,
    settings: CostSettings | None = None,
) -> float:
    """Calculate brokerage for one executed order."""
    settings = settings or CostSettings()

    turnover = float(turnover)

    if turnover <= 0:
        return 0.0

    return min(
        turnover * settings.brokerage_rate,
        settings.brokerage_max_per_order,
    )


def calculate_trade_costs(
    entry_price: float,
    exit_price: float,
    quantity: int,
    settings: CostSettings | None = None,
) -> TradingCosts:
    """
    Calculate all applicable costs for one NSE equity intraday trade.

    The trade is assumed to be:
        BUY quantity at entry_price
        SELL quantity at exit_price

    Slippage is modeled separately as a cost based on total turnover.
    """

    settings = settings or CostSettings()

    entry_price = float(entry_price)
    exit_price = float(exit_price)
    quantity = int(quantity)

    if entry_price <= 0:
        raise ValueError("entry_price must be greater than zero.")

    if exit_price <= 0:
        raise ValueError("exit_price must be greater than zero.")

    if quantity <= 0:
        raise ValueError("quantity must be greater than zero.")

    buy_turnover = entry_price * quantity
    sell_turnover = exit_price * quantity
    total_turnover = buy_turnover + sell_turnover

    # Brokerage is charged separately on BUY and SELL orders.
    buy_brokerage = calculate_brokerage(
        buy_turnover,
        settings,
    )

    sell_brokerage = calculate_brokerage(
        sell_turnover,
        settings,
    )

    brokerage = buy_brokerage + sell_brokerage

    # STT for equity intraday is charged on the SELL side.
    stt = sell_turnover * settings.stt_rate

    # Exchange transaction charges apply to both sides.
    exchange_transaction_charges = (
        total_turnover
        * settings.exchange_transaction_rate
    )

    # SEBI turnover fee applies to total turnover.
    sebi_charges = (
        total_turnover
        * settings.sebi_rate
    )

    # Stamp duty applies to the BUY side.
    stamp_duty = (
        buy_turnover
        * settings.stamp_duty_rate
    )

    # NSE IPFT applies to total traded value.
    ipft = (
        total_turnover
        * settings.ipft_rate
    )

    # GST is applied to brokerage + exchange transaction
    # charges + SEBI charges + IPFT.
    gst_base = (
        brokerage
        + exchange_transaction_charges
        + sebi_charges
        + ipft
    )

    gst = gst_base * settings.gst_rate

    # Gross trading P&L before transaction costs.
    gross_pnl = (
        exit_price - entry_price
    ) * quantity

    # Slippage is represented as an additional round-trip
    # trading cost. bps = basis points.
    slippage = (
        total_turnover
        * settings.slippage_bps
        / 10_000
    )

    total_cost = (
        brokerage
        + stt
        + exchange_transaction_charges
        + sebi_charges
        + stamp_duty
        + gst
        + ipft
        + slippage
    )

    net_pnl = gross_pnl - total_cost

    return TradingCosts(
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
        total_turnover=total_turnover,
        brokerage=brokerage,
        stt=stt,
        exchange_transaction_charges=exchange_transaction_charges,
        sebi_charges=sebi_charges,
        stamp_duty=stamp_duty,
        gst=gst,
        ipft=ipft,
        slippage=slippage,
        total_cost=total_cost,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
    )