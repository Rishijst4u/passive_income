# AI Trading Platform — Codex Project Instructions

## 1. Project Purpose

This repository is a research and experimentation platform for developing and validating an AI-assisted Indian equity trading system.

The immediate objective is NOT to make money or deploy live trading.

The immediate objective is:

1. Build a technically reliable trading research platform.
2. Develop deterministic trading strategies.
3. Backtest them against reliable historical Indian market data.
4. Paper trade them.
5. Collect at least 100 paper trades for Experiment #001.
6. Evaluate statistical performance before making strategy changes.
7. Only consider live trading after substantial validation and explicit user authorization.

Never claim that a strategy is profitable based on synthetic/sample data.

---

## 2. Current Experiment

### Experiment #001

Strategy name:

**VWAP + Momentum + Volume Breakout v1**

Primary timeframe:

**5-minute**

Context timeframe:

**15-minute**

Trading session:

- Entry start: 09:30 IST
- Entry end: 14:30 IST
- Long-only initially

Entry conditions:

1. Close > VWAP
2. EMA(9) > EMA(21)
3. RSI(14) between 55 and 70
4. Volume ratio > 1.5
5. Close breaks the previous short-term high
6. Current prototype uses a 3-bar breakout lookback

Initial stop:

- Below the signal bar low

Target:

- Minimum 2:1 reward-to-risk

A `NO TRADE` decision is valid and must not be treated as a failure.

Do not weaken or silently change these rules.

Any strategy change must:

- Have a new strategy/version identifier.
- Be documented.
- Be tested separately.
- Not overwrite the historical results of the previous version.

---

## 3. Virtual Capital and Risk Rules

Experiment #001 uses:

- Virtual capital: ₹50,000
- Maximum risk per trade: 0.5% of capital
- Initial maximum risk: ₹250
- Maximum position value: ₹10,000
- Maximum daily loss: 1%
- Maximum trades per day: 3
- Maximum simultaneous positions: 1

Do not introduce:

- Options
- Futures
- Leverage
- Martingale
- Averaging down
- Unbounded position sizing
- Overnight positions

Every simulated trade must have a defined stop.

Risk management must eventually be implemented as a deterministic engine.

---

## 4. Critical Architecture Rule

AI must NEVER have unrestricted authority to place broker orders.

The intended architecture is:

Market/Data
    ↓
Data Validation
    ↓
Indicators / Features
    ↓
Deterministic Strategy
    ↓
AI / Market Intelligence
    ↓
Deterministic Risk Engine
    ↓
Paper Broker
    ↓
Trade Journal
    ↓
Analytics

For future live trading:

Deterministic Risk Engine
    ↓
Execution Safety Checks
    ↓
Broker API

The AI layer may analyze:

- News
- Corporate announcements
- Earnings
- Sector conditions
- Market regime
- FII/DII information
- Mutual-fund disclosures
- Promoter disclosures
- Macro events
- Observable institutional behavior

However, AI output must be structured and must pass deterministic risk controls.

---

## 5. Trading Modes

The platform should eventually support:

- `BACKTEST`
- `PAPER`
- `LIVE`

Default mode:

**PAPER**

Never enable LIVE mode accidentally.

Never add live broker order placement unless the user explicitly requests it.

Even when live trading is eventually requested, require explicit configuration/authorization and retain deterministic risk controls.

---

## 6. No Look-Ahead Bias

This is one of the highest-priority rules.

Never use information that would not have been available at the simulated decision time.

Examples of prohibited behavior:

- Using future candles to generate current signals.
- Using future earnings information.
- Using revised historical information that was unavailable at the time.
- Using future constituent membership.
- Using future corporate actions incorrectly.
- Calculating indicators using future rows.
- Selecting today's best-performing stocks and pretending they were known beforehand.

When working with historical stock universes, avoid survivorship bias.

Prefer point-in-time universe membership where practical.

---

## 7. Historical Data

The next major milestone is reliable historical Indian market data.

Target:

- NSE equities
- Initially liquid NIFTY 500 universe
- 5-minute OHLCV data
- 15-minute contextual data
- Correct Indian market sessions
- Correct timestamps/timezone
- Corporate-action awareness where applicable
- Validation before backtesting

Preferred initial storage:

- Parquet for market/event datasets
- DuckDB for analytical queries

Do not treat `yfinance` as a production-quality NSE intraday data source.

It may be used for research/prototyping where appropriate, but historical-data limitations must be understood and documented.

---

## 8. Configuration

Do not hard-code strategy parameters or risk parameters inside Python modules when they belong in configuration.

Use:

`config/settings.yaml`

Configuration should eventually control:

- Capital
- Risk percentage
- Position limits
- Daily loss limit
- Maximum trades
- Session times
- Timeframe
- EMA periods
- RSI thresholds
- Volume threshold
- Breakout lookback
- Risk/reward ratio
- Execution assumptions

Configuration changes must be tested.

Avoid duplicated configuration values across multiple Python files.

---

## 9. Backtesting Requirements

The current backtester is intentionally simple and is NOT production-grade.

The realistic backtester must eventually model:

- Entry timing
- Exit timing
- Stop-loss
- Target
- Position sizing
- Maximum position value
- Daily loss limits
- Maximum trades/day
- Transaction costs
- Brokerage
- STT
- Exchange charges
- SEBI charges
- GST
- Stamp duty
- Slippage
- Bid/ask spread where data permits
- Gaps
- Ambiguous stop/target ordering
- Partial fills where relevant

Never report gross P&L as if it were net trading performance.

Always distinguish:

- Gross P&L
- Estimated costs
- Slippage
- Net P&L

---

## 10. Trade Journal

Every generated trade/signal should eventually have an auditable record.

Important fields include:

- Timestamp
- Symbol
- Strategy version
- Timeframe
- Market regime
- Sector
- Entry
- Stop
- Target
- Quantity
- Strategy score
- RSI
- EMA values
- VWAP
- Volume ratio
- Breakout strength
- News score
- AI score
- Exit
- Exit reason
- Gross P&L
- Transaction costs
- Slippage
- Net P&L
- MAE
- MFE

Do not delete historical experiment results when improving the system.

---

## 11. Performance Evaluation

Do not evaluate strategies using win rate alone.

Important metrics:

- Number of trades
- Total net P&L
- Return %
- Average trade
- Average daily P&L
- Win rate
- Average winner
- Average loser
- Payoff ratio
- Expectancy
- Profit factor
- Maximum drawdown
- Maximum consecutive losses
- Volatility
- Sharpe ratio
- Sortino ratio
- Slippage
- Transaction costs
- Rejected signals
- Missed signals

A strategy with a high win rate can still lose money.

A strategy with a low win rate can still be profitable if payoff and expectancy are favorable.

---

## 12. Experiment Methodology

Do not optimize endlessly on the same historical period.

Use:

- Training period
- Validation period
- Out-of-sample test period
- Walk-forward testing where appropriate
- Multiple market regimes
- Parameter sensitivity analysis

Avoid overfitting.

When comparing strategies, use the same:

- Universe
- Time period
- Data quality
- Cost model
- Slippage assumptions
- Evaluation methodology

Controlled ablation experiments should eventually compare:

1. Technical-only
2. Technical + market regime
3. Technical + regime + news
4. Technical + regime + news + AI

The purpose is to determine whether AI adds measurable value.

---

## 13. AI Layer

The AI layer should produce structured information rather than unrestricted prose.

Example conceptual output:

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

AI outputs should be:

- Structured
- Auditable
- Versioned
- Timestamped
- Separately stored from deterministic strategy outputs

AI confidence must never override hard risk limits.

AI must be allowed to return:

`NO_TRADE`

---

## 14. Investor / Institutional Intelligence

Do not assume there is one universal "promoter strategy" or "apex investor strategy."

Instead identify observable styles such as:

- Value
- Quality
- Growth
- Momentum
- Defensive
- Cyclical
- Concentrated ownership
- Sector rotation

Use publicly available information and respect its publication timing.

Never use information that was unavailable at the simulated historical timestamp.

---

## 15. Coding Standards

Prefer:

- Python 3.11+
- pandas
- numpy
- DuckDB
- PyArrow
- PyYAML
- pytest
- Streamlit

Keep modules small and logically separated.

Prefer explicit, readable code over clever abstractions.

Use type hints for public functions.

Document non-obvious financial/trading logic.

Do not introduce a large framework when a small implementation is sufficient.

Avoid unnecessary dependencies.

---

## 16. Testing Requirements

Every meaningful code change should include appropriate tests.

At minimum:

```bash
pytest -q
```

must pass before considering a change complete.

For trading logic, test:

- Indicator calculations
- Signal conditions
- Boundary conditions
- Session filtering
- Stop calculations
- Target calculations
- Position sizing
- Risk limits
- Daily loss limits
- Multiple trades/day
- No-look-ahead behavior
- Cost calculations

When fixing a bug, add a regression test where practical.

Never remove tests simply because they fail after a code change.

---

## 17. Reproducibility

Research results must be reproducible.

Record:

- Strategy version
- Configuration version
- Dataset/version
- Backtest period
- Cost assumptions
- Slippage assumptions
- Code version/commit where practical

Do not silently change historical experiment assumptions.

---

## 18. Security

Never commit:

- API keys
- Broker credentials
- Telegram tokens
- OpenAI API keys
- Passwords
- `.env` files
- Authentication cookies
- Private certificates

Use environment variables/secrets.

`.env` must remain ignored by Git.

Never print secrets in logs.

---

## 19. Broker Integration

Zerodha Kite Connect may eventually be used for live/paper execution.

Do NOT implement live order placement as part of routine development.

Before any live integration, the platform must have:

- Reliable market data
- Validated backtesting
- Realistic costs
- Risk engine
- Paper broker
- Trade journal
- Monitoring
- Error handling
- Kill switch
- Explicit LIVE mode
- Extensive paper-trading evidence

No live trading should happen merely because a broker API is configured.

---

## 20. Current Development Priority

Work in this order unless there is a strong technical reason to change it:

### Milestone 1
Configuration loader.

### Milestone 2
Reliable historical NSE data ingestion.

### Milestone 3
Data validation and Parquet storage.

### Milestone 4
Realistic backtester.

### Milestone 5
Deterministic risk engine.

### Milestone 6
Transaction-cost and slippage model.

### Milestone 7
Trade journal.

### Milestone 8
Performance analytics.

### Milestone 9
Run Experiment #001 until at least 100 paper trades are collected.

### Milestone 10
Analyze results and decide whether the strategy should be modified.

### Milestone 11
Paper-trading engine.

### Milestone 12
Market/news intelligence.

### Milestone 13
AI intelligence layer.

### Milestone 14
Only after substantial validation: evaluate live trading architecture.

Do not skip directly to AI, Telegram alerts, or live execution while the data/backtesting foundation is unreliable.

---

## 21. How Codex Should Work

Before modifying code:

1. Inspect the repository.
2. Read `README.md`.
3. Read `DOCUMENTATION.md`.
4. Read this `AGENTS.md`.
5. Understand existing architecture.
6. Identify the smallest safe change.
7. Explain any important assumptions.

When implementing:

- Make focused changes.
- Avoid unrelated refactoring.
- Preserve existing working behavior unless the task explicitly changes it.
- Add/update tests.
- Run tests.
- Report what changed.
- Report tests executed and results.
- Mention known limitations.

Do not invent APIs or undocumented behavior.

If an external API/library behavior matters, verify current documentation before implementing it.

---

## 22. Git Discipline

Use small logical commits.

Preferred pattern:

```text
feat: add configuration loader
test: add configuration loader tests
feat: add NSE historical data ingestion
test: validate historical data ingestion
feat: add realistic transaction cost model
```

Do not combine unrelated features into one large commit.

Never commit secrets or generated large datasets unless explicitly intended.

---

## 23. Important Safety Principle

This project is a financial trading research system.

The objective is **not** to maximize the number of trades.

The objective is to determine whether a strategy has robust positive expectancy after realistic costs and across different market regimes.

A correct outcome may be:

`NO TRADE`

A correct experiment outcome may also be:

`STRATEGY REJECTED`

Do not modify the strategy merely to make the backtest look better.

Statistical evidence takes priority over intuition.

---

## 24. Current Repository State

The current repository is an early prototype.

Existing components include:

- Synthetic sample data
- Basic strategy implementation
- Basic backtester
- CLI
- Streamlit dashboard
- Initial tests
- GitHub Actions
- YAML configuration file
- Documentation

Important known limitation:

The YAML configuration exists, but the original prototype did not fully wire configuration values into the strategy/backtester.

The next implementation task is therefore the configuration-loading foundation.

---

## 25. Immediate Task

When asked to continue development from the current state, start with:

**Configuration-loading foundation**

Requirements:

1. Load `config/settings.yaml`.
2. Expose configuration cleanly to the application.
3. Replace duplicated hard-coded strategy parameters with configuration values.
4. Preserve current strategy behavior.
5. Add tests for configuration loading.
6. Keep the implementation simple.
7. Do NOT implement position sizing yet.
8. Do NOT implement historical data ingestion yet.
9. Do NOT implement broker integration.
10. Do NOT implement AI.
11. Run the full test suite.

After completion, report:

- Files changed
- What was implemented
- Tests added
- Test command
- Test result
- Any remaining limitations

Do not proceed to the next milestone unless requested or explicitly directed by the project plan.