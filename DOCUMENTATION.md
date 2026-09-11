# AI Trading Platform — Full System Documentation

## 1. Purpose

This project is a research-first automated trading platform for Indian equity markets.

The initial objective is NOT to guarantee profit. The first objective is to determine whether a deterministic intraday strategy has a statistically credible positive expectancy over a sufficiently large paper-trading sample.

Initial virtual capital: **₹50,000**.

Initial strategy experiment: **VWAP + Momentum + Volume Breakout (Experiment 001)**.

The platform is designed to evolve from:

1. Historical backtesting
2. Paper trading
3. AI-assisted market intelligence
4. Extended paper validation
5. Only then, optional live execution

No live broker execution is included in Experiment 001.

---

## 2. Core principles

### 2.1 Research before live trading

The system must prove an edge before real capital is exposed.

The first milestone is at least 100 paper trades, followed by deeper validation. A strategy must not be declared successful merely because it produces a positive result on one historical period.

### 2.2 Avoid overfitting

Do not optimize parameters until the historical sample produces a desired result.

Use:
- train/test periods
- out-of-sample validation
- multiple market regimes
- walk-forward testing
- ablation testing

### 2.3 AI is an intelligence layer, not an unrestricted execution authority

An LLM may analyze:
- news
- market context
- investor/fund disclosures
- sector context
- qualitative information

The LLM must return structured information.

A deterministic strategy and deterministic risk engine must remain between analysis and order execution.

### 2.4 Risk is mandatory

Every executable paper/live order must pass risk checks.

No order should be accepted without:
- entry
- stop loss
- position size
- maximum loss
- valid market/session conditions

### 2.5 Paper trading is the default

Until explicitly promoted through a controlled release process, execution mode must remain PAPER.

---

# 3. Initial objective

The initial research question is:

> Does the VWAP + Momentum + Volume Breakout strategy produce positive risk-adjusted expectancy after realistic Indian equity transaction costs and slippage?

The project should not assume a target such as ₹2,000/day is achievable.

₹2,000/day on ₹50,000 represents 4% per trading day and is an extremely aggressive target. The system should measure actual expectancy instead.

---

# 4. Initial strategy — Experiment 001

## Name

**VWAP Momentum Breakout v1**

## Universe

Initial target:
- NSE equities
- preferably liquid NIFTY 500 constituents
- additional liquidity filters before execution

The exact historical universe must be point-in-time correct in the production research system to avoid survivorship bias.

## Direction

Long only initially.

Short selling can be evaluated later.

## Timeframes

Primary:
- 5-minute bars

Context:
- 15-minute bars

## Entry window

Initial research window:
- 09:30 to 14:30 IST

Avoid entering immediately after the open while spreads and volatility can be unusually high.

## Entry conditions

A BUY candidate requires all of the following:

1. Close > VWAP
2. EMA(9) > EMA(21)
3. RSI(14) between 55 and 70
4. Current volume / rolling average volume > 1.5
5. Close breaks the prior short-term high

The current prototype uses a three-bar breakout lookback.

## Initial stop

Initial prototype:
- stop below the signal bar low

This is a research baseline and may be replaced after testing.

## Target

Minimum initial risk/reward:
- 1:2

Example:

Entry = ₹100
Stop = ₹99
Risk = ₹1
Target = ₹102

## Candidate ranking

When multiple stocks qualify, rank candidates using a deterministic score.

Initial score components:
- trend
- VWAP relationship
- momentum
- volume anomaly
- breakout strength

The scoring model will be expanded later.

---

# 5. Capital and risk model

Initial virtual capital:

**₹50,000**

Initial limits:

- Maximum risk per trade: 0.5% of capital
- Maximum position value: ₹10,000
- Maximum daily loss: 1% of capital
- Maximum trades/day: 3
- Maximum simultaneous positions: 1
- Minimum risk/reward: 1:2
- Stop loss: mandatory

Position sizing formula:

`quantity = floor(max_risk_amount / (entry_price - stop_price))`

Then cap quantity using the maximum position value.

Example:

Capital = ₹50,000
Maximum risk = 0.5% = ₹250
Entry = ₹100
Stop = ₹99

Risk/share = ₹1

Maximum theoretical quantity = 250 shares.

Position value = ₹25,000.

Because the initial maximum position value is ₹10,000:

Final quantity = 100 shares.

Maximum planned loss ≈ ₹100 before costs/slippage.

---

# 6. Transaction costs

A production-quality backtest must model Indian equity trading costs.

The cost model should eventually include, as applicable:

- brokerage
- STT
- exchange transaction charges
- GST
- SEBI charges
- stamp duty
- slippage
- bid/ask spread

Do not use raw entry-to-exit price difference as final performance.

The exact cost schedule must be configurable and updated from authoritative broker/exchange/tax sources when the live system is implemented.

---

# 7. System architecture

## Current architecture

```text
Market Data
    |
    v
Data Processing
    |
    v
Indicators
    |
    v
Strategy Engine
    |
    v
Backtest / Paper Broker
    |
    v
Trade Journal
    |
    v
Analytics
```

## Target architecture

```text
                         +--------------------+
                         |    Dashboard       |
                         +---------+----------+
                                   |
                                   v
+-------------+          +--------------------+
| Market Data | -------> | Market Intelligence|
+-------------+          +---------+----------+
                                   |
                  +----------------+----------------+
                  |                |                |
                  v                v                v
             Technical          News          Investor Data
               Engine          Engine             Engine
                  |                |                |
                  +----------------+----------------+
                                   |
                                   v
                         +--------------------+
                         | Strategy Ensemble  |
                         +---------+----------+
                                   |
                                   v
                         +--------------------+
                         | Risk Engine        |
                         +---------+----------+
                                   |
                                   v
                         +--------------------+
                         | Paper Broker       |
                         +---------+----------+
                                   |
                                   v
                         +--------------------+
                         | Trade Journal      |
                         +---------+----------+
                                   |
                                   v
                         +--------------------+
                         | Analytics / ML     |
                         +--------------------+
```

---

# 8. AI market-intelligence layer

The AI layer is planned but is not required for Experiment 001.

Future inputs may include:

- market news
- corporate announcements
- earnings
- sector news
- FII/DII information
- mutual fund portfolio disclosures
- institutional positioning where legally/publicly available
- promoter shareholding disclosures
- corporate actions
- macroeconomic events
- market regime

The AI should NOT blindly copy an investor.

There is no single "promoter strategy" or universal "apex investor strategy".

Instead, the system should infer observable styles such as:
- value
- quality
- growth
- momentum
- defensive
- cyclical
- concentration
- sector rotation

---

# 9. AI output contract

The AI layer should eventually return structured output similar to:

```json
{
  "symbol": "RELIANCE",
  "market_view": "BULLISH",
  "news_sentiment": 0.72,
  "sector_view": 0.68,
  "risk_flags": [],
  "confidence": 0.81,
  "summary": "Positive market context with supportive sector momentum."
}
```

The AI must never return a free-form sentence that is directly interpreted as an order.

The downstream deterministic engine converts validated information into an executable decision.

---

# 10. NO-TRADE capability

The system must be allowed to decide:

**NO TRADE**

Examples:
- conflicting signals
- poor liquidity
- excessive volatility
- insufficient risk/reward
- market regime unsuitable
- daily loss limit reached
- news risk too high
- technical setup invalidated

Avoid forcing the system to trade every day.

---

# 11. Backtesting requirements

The backtester must eventually support:

- historical OHLCV data
- point-in-time universe
- corporate actions
- realistic costs
- realistic slippage
- order timing
- stop/target execution rules
- partial fills if modelled
- market session rules
- position limits
- daily loss limits

Avoid look-ahead bias.

For example, a signal generated from a candle cannot use information from the candle's future close if the simulated order is assumed to execute before that close is known.

---

# 12. Performance metrics

After each experiment calculate:

## Return metrics
- total P&L
- return %
- average daily P&L

## Trade metrics
- number of trades
- win rate
- average winner
- average loser
- expectancy
- profit factor
- payoff ratio

## Risk metrics
- maximum drawdown
- average drawdown
- maximum consecutive losses
- volatility
- Sharpe ratio
- Sortino ratio

## Execution metrics
- average slippage
- estimated transaction costs
- rejected signals
- missed trades

---

# 13. Required 100-trade experiment

Experiment 001 should collect at least 100 paper trades.

Record EVERY candidate signal, not only trades.

For each signal/trade store:

- timestamp
- symbol
- strategy version
- timeframe
- market regime
- sector
- entry
- stop
- target
- quantity
- strategy score
- RSI
- EMA values
- VWAP
- volume ratio
- news score
- AI score when available
- exit
- exit reason
- gross P&L
- transaction costs
- slippage
- net P&L
- MAE
- MFE

---

# 14. Experiment methodology

Do not change the strategy after every losing trade.

Use controlled experiments.

Example:

### Experiment A
Technical signals only

### Experiment B
Technical + market regime

### Experiment C
Technical + market regime + news

### Experiment D
Technical + market regime + news + AI

Compare them using identical evaluation periods.

This determines whether each additional component adds measurable value.

---

# 15. Project structure

```text
ai-trading-platform/
|
├── README.md
├── DOCUMENTATION.md
├── requirements.txt
├── .env.example
├── .gitignore
|
├── config/
│   └── settings.yaml
|
├── data/
│   ├── raw/
│   └── processed/
|
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── sample_data.py
│   ├── strategy.py
│   ├── backtest.py
│   |
│   ├── data/
│   ├── indicators/
│   ├── strategies/
│   ├── backtesting/
│   ├── paper_trading/
│   ├── risk/
│   └── ai/
|
├── tests/
│   └── test_strategy.py
|
├── notebooks/
|
├── dashboard/
│   └── app.py
|
└── .github/
    └── workflows/
        ├── test.yml
        └── paper_daily.yml
```

---

# 16. Technology stack

## Current

- Python 3.11+
- pandas
- numpy
- DuckDB
- PyArrow
- ta
- yfinance for initial research/development only
- Streamlit
- pytest
- PyYAML
- python-dotenv

## Planned

Potential additions:
- FastAPI
- PostgreSQL/TimescaleDB if operational scale requires it
- Kafka/Redpanda only if streaming scale requires it
- Docker
- Terraform
- Telegram notifications
- broker API

Do not introduce distributed infrastructure prematurely.

---

# 17. Storage strategy

Initial:

```text
Parquet = historical/event data
DuckDB  = analytics/query layer
```

Potential future:

```text
S3/Object Storage
       |
       +-- Parquet market data
       |
       +-- trade history

PostgreSQL/TimescaleDB
       |
       +-- live positions
       +-- orders
       +-- signals
       +-- configuration
```

---

# 18. Development workflow

Use Git.

Recommended branches:

```text
main
  |
  +-- develop
       |
       +-- feature/data-ingestion
       +-- feature/backtester
       +-- feature/news-engine
       +-- feature/paper-broker
```

Every strategy change must have:
- version number
- changelog entry
- test
- experiment identifier

Never silently modify an old strategy version.

---

# 19. Deployment strategy

## Phase 1

Laptop:
- development
- testing
- research

GitHub:
- source control
- CI

GitHub Actions:
- automated tests
- lightweight scheduled experiments

## Phase 2

Cloud:
- paper-trading engine
- dashboard
- scheduled jobs

## Phase 3

Live trading only after sufficient validation.

Live execution should have:
- explicit mode switch
- independent risk service
- kill switch
- maximum daily loss
- maximum order value
- duplicate-order protection
- broker reconciliation
- alerting
- audit log

---

# 20. Security rules

Never commit:
- broker API keys
- API secrets
- Telegram tokens
- LLM API keys
- passwords
- session tokens

Use:
- `.env` locally
- GitHub Secrets for GitHub Actions
- cloud secret management for deployment

`.env.example` contains placeholders only.

---

# 21. Current execution modes

The platform should eventually support:

```text
BACKTEST
PAPER
LIVE
```

Default:

```text
PAPER
```

LIVE must require an explicit configuration change and preferably an operational approval step.

---

# 22. Current limitations

Experiment 001 is intentionally incomplete.

It currently has:
- basic strategy implementation
- sample data generator
- basic backtest
- basic Streamlit dashboard
- tests
- GitHub Actions skeleton

It does NOT yet have:
- authoritative NSE historical data ingestion
- complete Indian transaction-cost model
- production-quality execution simulation
- point-in-time NIFTY 500 universe
- news ingestion
- AI market analysis
- investor/fund intelligence
- broker integration
- live trading

Do not interpret the sample-data output as strategy performance.

---

# 23. Roadmap

## Milestone 1 — Foundation
- Git repository
- Python environment
- tests
- project structure

## Milestone 2 — Historical market data
- NIFTY 500 universe
- 5-minute data
- 15-minute data
- data validation
- storage in Parquet

## Milestone 3 — Backtester
- realistic execution
- costs
- slippage
- session handling
- metrics

## Milestone 4 — Experiment 001
- generate >=100 trades
- analyze results
- identify weaknesses

## Milestone 5 — Robustness
- out-of-sample testing
- walk-forward testing
- market regimes
- parameter sensitivity

## Milestone 6 — Paper trading
- automated daily/intraday execution
- paper portfolio
- Telegram alerts
- dashboard

## Milestone 7 — AI intelligence
- news
- market regime
- investor/fund data
- AI scoring

## Milestone 8 — Extended validation
- hundreds of trades
- multiple regimes
- cost-adjusted performance

## Milestone 9 — Live pilot
Only if validation supports it.

Start with a small amount of capital and strict risk controls.

---

# 24. Instructions for another AI/LLM

If another AI/LLM is given this repository and this document, it should understand the following:

1. This is a research project for automated Indian equity trading.
2. The current strategy is VWAP Momentum Breakout v1.
3. The current capital is virtual ₹50,000.
4. The project is PAPER ONLY.
5. Do not add live broker execution unless explicitly requested.
6. Do not claim profitability without statistically meaningful evidence.
7. Do not use sample-data results as financial evidence.
8. Preserve reproducibility.
9. Do not introduce look-ahead bias.
10. Model transaction costs before evaluating profitability.
11. Do not overfit strategy parameters to historical data.
12. Keep AI analysis separate from deterministic risk controls.
13. Always allow NO TRADE.
14. Never commit secrets.
15. Every strategy modification should have a version and experiment record.
16. Prefer open-source/free/local technologies during research.
17. The next major milestone is reliable historical NSE data ingestion and a realistic backtester.
18. The ultimate system may include AI news/investor intelligence and broker execution, but those are later phases.

---

# 25. Definition of done for Experiment 001

Experiment 001 is complete only when:

- historical data is validated
- strategy rules are deterministic
- costs are modelled
- at least 100 paper trades are collected
- every trade has an audit record
- performance metrics are calculated
- drawdown is measured
- out-of-sample results are available
- no look-ahead bias is identified
- results are reproducible from Git
- strategy version is frozen

A positive result is useful.

A negative result is also useful because it tells us what not to build.

---

# 26. Guiding principle

The platform is not designed to "predict the market."

It is designed to answer:

> **Can a repeatable, measurable trading process generate positive risk-adjusted expectancy after costs, and can additional market intelligence improve that expectancy without introducing overfitting?**

That is the central research question for the entire project.
