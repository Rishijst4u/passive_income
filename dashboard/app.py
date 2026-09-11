import streamlit as st
from src.sample_data import make_sample_data
from src.backtest import run_simple_backtest

st.set_page_config(page_title="AI Trading Platform", layout="wide")
st.title("AI Trading Platform — Experiment 001")
st.warning("PAPER TRADING / RESEARCH ONLY")

df = make_sample_data()
trades = run_simple_backtest(df)

c1, c2, c3 = st.columns(3)
c1.metric("Sample trades", len(trades))
c2.metric("Raw P&L", f"₹{sum(t.pnl for t in trades):,.2f}")
c3.metric("Capital", "₹50,000")

if trades:
    st.dataframe([t.__dict__ for t in trades], use_container_width=True)
else:
    st.info("No sample trades generated.")
