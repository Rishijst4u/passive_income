# AI Trading Platform — Experiment 001

Research-only starter project for a VWAP + Momentum + Volume Breakout strategy.

## Safety
This repository is PAPER TRADING / BACKTESTING ONLY. It contains no broker execution code.

## Stack
- Python 3.11+
- pandas / numpy
- DuckDB
- yfinance (research data only)
- ta
- Streamlit
- pytest

## Quick start

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest
python -m src.cli backtest
streamlit run dashboard/app.py
```

### macOS/Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest
python -m src.cli backtest
streamlit run dashboard/app.py
```

The initial backtest uses a small deterministic sample generator so the project runs without paid market data.
Real market-data ingestion will be added next.
