import pandas as pd

from src.config import load_settings
from src.risk import (
    RiskState,
    calculate_position_size,
    check_entry,
    record_entry,
    record_exit,
    reset_day,
)


def test_position_size_respects_risk_and_position_value():
    settings = load_settings()

    # ₹500 entry, ₹490 stop.
    # Risk/share = ₹10.
    # Maximum risk = ₹250.
    # Risk-based quantity = 25.
    # But max position value = ₹10,000 => 20 shares.
    size = calculate_position_size(500, 490, settings)

    assert size.quantity == 20
    assert size.risk_per_share == 10
    assert size.planned_risk == 200
    assert size.position_value == 10_000


def test_position_size_rejects_invalid_stop():
    settings = load_settings()

    size = calculate_position_size(500, 500, settings)

    assert size.quantity == 0
    assert size.planned_risk == 0
    assert size.position_value == 0


def test_entry_is_approved_when_all_risk_rules_pass():
    settings = load_settings()
    state = RiskState()

    timestamp = pd.Timestamp("2026-08-03 10:00", tz="Asia/Kolkata")

    decision = check_entry(
        timestamp=timestamp,
        entry=500,
        stop=490,
        settings=settings,
        state=state,
    )

    assert decision.approved is True
    assert decision.reason == "APPROVED"
    assert decision.position_size.quantity == 20


def test_entry_rejected_outside_entry_window():
    settings = load_settings()
    state = RiskState()

    timestamp = pd.Timestamp("2026-08-03 09:20", tz="Asia/Kolkata")

    decision = check_entry(
        timestamp=timestamp,
        entry=500,
        stop=490,
        settings=settings,
        state=state,
    )

    assert decision.approved is False
    assert decision.reason == "OUTSIDE_ENTRY_WINDOW"


def test_max_trades_per_day_is_enforced():
    settings = load_settings()
    state = RiskState(trades_today=settings.max_trades_per_day)

    timestamp = pd.Timestamp("2026-08-03 10:00", tz="Asia/Kolkata")

    decision = check_entry(
        timestamp=timestamp,
        entry=500,
        stop=490,
        settings=settings,
        state=state,
    )

    assert decision.approved is False
    assert decision.reason == "MAX_TRADES_PER_DAY"


def test_only_one_position_can_be_open():
    settings = load_settings()
    state = RiskState(position_open=True)

    timestamp = pd.Timestamp("2026-08-03 10:00", tz="Asia/Kolkata")

    decision = check_entry(
        timestamp=timestamp,
        entry=500,
        stop=490,
        settings=settings,
        state=state,
    )

    assert decision.approved is False
    assert decision.reason == "POSITION_OPEN"


def test_daily_loss_limit_is_enforced():
    settings = load_settings()

    # ₹500 daily loss limit.
    state = RiskState(realized_pnl=-400)

    timestamp = pd.Timestamp("2026-08-03 10:00", tz="Asia/Kolkata")

    # Planned risk = ₹200.
    # Projected loss = -₹600, so trade must be rejected.
    decision = check_entry(
        timestamp=timestamp,
        entry=500,
        stop=490,
        settings=settings,
        state=state,
    )

    assert decision.approved is False
    assert decision.reason == "DAILY_LOSS_LIMIT"


def test_entry_and_exit_update_risk_state():
    state = RiskState()

    record_entry(state)

    assert state.position_open is True
    assert state.trades_today == 1

    record_exit(state, -200)

    assert state.position_open is False
    assert state.realized_pnl == -200


def test_reset_day_resets_daily_controls():
    state = RiskState(
        realized_pnl=-200,
        trades_today=3,
        position_open=True,
    )

    reset_day(state)

    assert state.realized_pnl == -200
    assert state.trades_today == 0
    assert state.position_open is False