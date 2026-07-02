# Project 1 — Systematic Backtesting Framework

**Testing my own discretionary trading strategy honestly — with realistic costs, no look-ahead, and no self-deception.**

For two years I've traded a Smart Money Concepts setup (a "sweep + shift" reversal) on BTC and gold. It *feels* like an edge. This project asks the uncomfortable question every trader avoids: **is it actually one, once you account for real trading costs?**

Rather than build a generic backtester on someone else's strategy, I encoded *my own* discretionary rules into exact, testable logic — then ran them through an honest event-driven engine.

---

## What this is

An event-driven backtesting framework built from scratch (no `backtrader`/`vectorbt` black box) that:

- Converts a **discretionary** SMC strategy into **exact, code-checkable rules**
- Simulates execution **bar by bar with no look-ahead**
- Charges **realistic commission + slippage** on every fill
- Sizes every position to a **fixed 1% risk** per trade
- Reports the metrics a quant actually reads (Sharpe, Sortino, Calmar, max drawdown, profit factor, expectancy in R)
- Benchmarks against **buy-and-hold**

## The strategy, defined precisely

The hardest — and most valuable — part of this project was turning "I can see the setup on the chart" into rules a computer can execute with zero ambiguity:

| Component | Exact rule |
|---|---|
| **Swing pivot** | Fractal, half-width 3 (a high/low that exceeds the 3 bars on each side). Confirmed only 3 bars later — no look-ahead. |
| **Liquidity sweep** | A candle whose wick pierces the most recent confirmed swing high/low **but closes back inside** it. |
| **The shift (CHoCH)** | After a sweep, price must **close beyond the most recent minor swing** in the reversal direction — the change of character. |
| **Entry** | At the close of the candle that confirms the CHoCH. |
| **Stop-loss** | 1.5 × ATR(14). |
| **Take-profit** | Fixed 2.5R (v1). A trailing-stop variant is included as `--exit trailing`. |

## How to run it

```bash
# Clone and install
pip install pandas numpy requests matplotlib

# On your machine — real BTC data from Binance (no API key needed):
python -m p1_backtester.run --source binance --start 2023-01-01 --end 2024-01-01

# Anywhere — synthetic data, to verify the engine runs:
python -m p1_backtester.run --source synthetic

# Test the trailing-stop exit instead of fixed 2.5R:
python -m p1_backtester.run --source binance --exit trailing
```

## Correctness is tested, not assumed

A backtest you can't trust is worse than none. The engine ships with a `pytest` suite that proves the parts that matter:

- No look-ahead in pivots or ATR
- Costs can only *subtract* from P&L
- Stop-outs lose ≈ 1R; targets pay ≈ 2.5R
- Risk-per-trade is respected
- Buy-and-hold matches the raw price move

```bash
python -m pytest tests/ -v      # 13 passing
```

## The honest part

> **Note on results:** the numbers you get depend entirely on the data window you run. This repo ships with a synthetic-data mode so anyone can verify the engine works without market access — those numbers are **not** a real result and are labelled as such. The real test is running `--source binance` on live history.

What I learned building this matters more than any single equity curve:

- **A backtest without costs is a fantasy.** The gap between gross and net P&L on an intraday strategy is brutal — most of the "edge" on lower timeframes can be commission and slippage.
- **"It works when I watch it live" is not evidence.** Discretionary confirmation bias is real; a mechanical test removes it.
- **Encoding the strategy exposed its ambiguity.** The CHoCH rule has genuine interpretation room. Where a rule is hard to define precisely, that itself is a finding: the edge may live in discretion, not in the rules.
- **The point of testing isn't to confirm you're right — it's to find where you're wrong before the market charges you the tuition.**

## Structure

```
core/          shared data loaders, metrics, plotting theme
p1_backtester/
  strategy.py  the A+ Sweep-Shift logic, made exact  (the real work)
  engine.py    event-driven execution + realistic costs
  run.py       CLI entry point
tests/         correctness suite (pytest)
```

## Roadmap (next iterations)

- [ ] Walk-forward validation (in-sample vs out-of-sample)
- [ ] Monte-Carlo trade reshuffling (is the result luck?)
- [ ] Parameter-sensitivity heatmaps
- [ ] Validation against my real executed trades

---

*Part of [quant-lab](../) — a build-and-learn portfolio of three interconnected quantitative-finance projects. Not financial advice.*
