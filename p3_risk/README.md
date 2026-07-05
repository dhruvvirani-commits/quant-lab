# Project 3 — Multi-Asset Risk & Portfolio Analytics

**An institutional risk dashboard built from scratch: how much can a portfolio lose, and how do its assets move together?**

Projects 1 and 2 were about finding and pricing edges. This one is about the other half of quant: not blowing up while you pursue them. It's the risk-management layer every real desk runs.

---

## What this is

A from-scratch risk toolkit over a multi-asset portfolio (BTC, ETH, and other assets):

- **Value-at-Risk (VaR)** — three methods (historical, parametric, Monte-Carlo): how much you could lose on a bad day
- **Expected Shortfall (ES)** — the average loss *when* a bad day happens; the measure regulators moved to
- **Correlation analysis** — static and rolling; the math of diversification, and how it breaks in a crash
- **Efficient frontier** — Markowitz optimization: the minimum-variance and maximum-Sharpe portfolios
- **A visual dashboard** — all of it in one institutional-style figure

## The core ideas

**Risk is about movement, not price.** Everything works on returns, never price levels.

**VaR vs ES.** VaR is the threshold of a bad day ("95% of days you lose less than X"). ES is how bad it gets beyond that threshold — the average of the worst outcomes. ES is always ≥ VaR, and it's the more honest number.

**Diversification is the only free lunch.** Combining assets that don't move together reduces risk without sacrificing return. But the catch: correlations spike toward 1 in a crash, exactly when you need diversification most. Rolling correlation shows this.

**The efficient frontier, honestly.** Markowitz optimization finds the best risk/return mix — but it assumes you know future returns and correlations. You don't; you estimate them from unstable history. So the "optimal" weights are fragile. Understanding that limitation matters as much as the math.

## How to run it

```bash
pip install -r requirements.txt
python -m pytest tests/test_risk.py -v

# On your machine, real crypto data:
python -m p3_risk.run --source binance

# Anywhere, synthetic multi-asset data:
python -m p3_risk.run --source synthetic
```

## Correctness is tested

- ES is always ≥ VaR (tail is worse than threshold)
- Higher confidence → higher VaR
- The three VaR methods agree on near-normal data
- Diversification ratio > 1 (portfolio less volatile than sum of parts)
- Min-variance portfolio has the lowest vol; max-Sharpe has the highest Sharpe

## Structure

```
core/            shared plotting theme (reused across projects)
p3_risk/
  returns.py       returns foundation + multi-asset table
  var_es.py        VaR (3 methods) + Expected Shortfall
  correlation.py   correlation matrix, rolling correlation, diversification
  frontier.py      Markowitz efficient frontier + optimal portfolios
  dashboard.py     the multi-panel risk dashboard
  run.py           end-to-end pipeline
tests/             correctness suite
```

## What this project demonstrates

- The **statistics of risk** (distributions, tails, percentiles) applied to real portfolios
- **Portfolio theory** (correlation, diversification, optimization) implemented from scratch
- Knowing the **limitations** of the models, not just the formulas — the mark of real risk literacy

---

*Part of [quant-lab](../). Not financial advice. Built by Dhruv Virani — MSc Artificial Intelligence, ECE Paris.*
