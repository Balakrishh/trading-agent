# Autonomous Options Credit Spread Trading Agent

An autonomous trading agent specialized in generating daily income through high-probability, risk-defined options credit spreads. The agent's primary goal is **capital preservation** — it only enters trades where the maximum loss is known and capped, prioritizing time decay (Theta) over directional speculation.

## Architecture

The agent operates in a continuous four-phase loop:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  I. PERCEIVE │────▶│ II. CLASSIFY │────▶│  III. PLAN   │────▶│   IV. ACT    │
│              │     │              │     │              │     │              │
│ Fetch 200d   │     │ Determine    │     │ Select spread│     │ Validate risk│
│ price data + │     │ regime:      │     │ type + pick  │     │ guardrails + │
│ option chain │     │ Bull/Bear/   │     │ strikes from │     │ fire order   │
│ with Greeks  │     │ Sideways     │     │ option chain │     │ to Alpaca    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### Market Regime → Strategy Matrix

| Regime | Detection Rule | Strategy |
|--------|---------------|----------|
| **Bullish** | Price > SMA-200 AND SMA-50 slope > 0 | Bull Put Spread |
| **Bearish** | Price < SMA-200 AND SMA-50 slope < 0 | Bear Call Spread |
| **Sideways** | Between SMAs or narrow Bollinger Bands | Iron Condor |

## Project Structure

```
Trading Agent/
├── .env                        # API keys and config (not committed)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── trading_agent/
│   ├── __init__.py
│   ├── config.py               # Environment config loader
│   ├── logger_setup.py         # Logging configuration
│   ├── market_data.py          # Phase I  — Yahoo Finance + Alpaca data
│   ├── regime.py               # Phase II — Regime classifier
│   ├── strategy.py             # Phase III — Strategy planner
│   ├── risk_manager.py         # Phase IV — Risk guardrails
│   ├── executor.py             # Phase IV — Order execution
│   └── agent.py                # Main orchestrator + CLI entry point
├── tests/
│   ├── conftest.py             # Shared fixtures (synthetic data)
│   ├── test_config.py          # Config loading tests
│   ├── test_market_data.py     # Technical indicator tests
│   ├── test_regime.py          # Regime classification tests
│   ├── test_strategy.py        # Strategy selection tests
│   ├── test_risk_manager.py    # Risk guardrail tests
│   ├── test_executor.py        # Order execution tests
│   └── test_agent_integration.py  # End-to-end pipeline tests
├── logs/                       # Runtime log files
└── trade_plans/                # JSON trade plan audit trail
```

## Risk Management Guardrails

Every trade must pass **all six checks** before execution:

1. **Plan Validity** — The strategy planner found valid strikes and contracts
2. **Credit-to-Width Ratio ≥ 1/3** — Collect at least $1.65 on a $5 wide spread
3. **Sold Delta ≤ 0.20** — High probability of expiring worthless (~80%+)
4. **Max Loss ≤ 2% of Account** — Position sizing relative to equity
5. **Account Type = Paper** — Safety assertion against live trading
6. **Market is Open** — No orders outside trading hours

## Setup

### 1. Install dependencies

```bash
cd "Trading Agent"
pip install -r requirements.txt
```

### 2. Configure environment

Edit the `.env` file with your Alpaca Paper Trading credentials:

```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
TICKERS=SPY,QQQ
DRY_RUN=false
```

### 3. Run the agent

```bash
# Live paper trading mode (uses .env settings)
python -m trading_agent.agent

# Force dry-run mode (calculates but doesn't execute)
python -m trading_agent.agent --dry-run

# Specify custom .env file
python -m trading_agent.agent --env /path/to/.env
```

### 4. Run tests

```bash
pytest tests/ -v
```

## Trade Plan Audit Trail

Every trade cycle produces a JSON file in `trade_plans/` containing the full reasoning log:

```json
{
  "trade_plan": {
    "ticker": "SPY",
    "strategy": "Bull Put Spread",
    "regime": "bullish",
    "legs": [...],
    "spread_width": 5.0,
    "net_credit": 1.70,
    "max_loss": 330.0,
    "credit_to_width_ratio": 0.34,
    "reasoning": "Bull Put Spread on SPY (bullish regime). Sold 480 (Δ=-0.150)..."
  },
  "risk_verdict": {
    "approved": true,
    "account_balance": 100000,
    "checks_passed": ["..."],
    "checks_failed": []
  },
  "mode": "live"
}
```

## Data Sources

| Source | Purpose | What It Provides |
|--------|---------|-----------------|
| **Yahoo Finance** | Regime Detection | 200-day historical OHLCV for SMAs, RSI, Bollinger Bands |
| **Alpaca Market Data** | Option Snapshots | Real-time Greeks (Delta, Theta, Vega), Bid/Ask spreads |
| **Alpaca Paper API** | Order Execution | Paper trading sandbox for safe order submission |

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `TICKERS` | `SPY,QQQ` | Comma-separated list of underlyings to trade |
| `MODE` | `dry_run` | Operating mode |
| `DRY_RUN` | `true` | If true, log plans but don't submit orders |
| `MAX_RISK_PCT` | `0.02` | Maximum loss per trade as % of account |
| `MIN_CREDIT_RATIO` | `0.33` | Minimum credit collected / spread width |
| `MAX_DELTA` | `0.20` | Maximum absolute delta of sold strike |
| `LOG_LEVEL` | `INFO` | Python logging level |
