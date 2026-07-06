# quant-lab

**A build-and-learn portfolio of quantitative-finance projects — built to learn the field by doing, with honest results over pretty ones.**

I came to AI through financial markets. `quant-lab` is where I turn years of self-taught, discretionary market analysis into rigorous, systematic, tested code.

| Project | Pillar | Status |
|---|---|---|
| **P1 · Backtester** | Systematic research & testing rigor | ✅ Shipped |
| **P2 · Volatility & Options** | Derivatives pricing & vol modeling | ✅ Shipped |
| **P3 · Risk & Portfolio** | Risk management & portfolio construction | ✅ Shipped |

Together the three cover the full quant workflow: **find** an edge (P1), **price** it (P2), and **manage the risk** of holding it (P3).

## P1 — Systematic Backtesting
Encoded a discretionary SMC strategy into an honest event-driven backtester, tested across 5 assets / 4 years / 230 trades, and concluded without curve-fitting that it isn't a mechanical edge. See [`p1_backtester/`](p1_backtester/).

## P2 — Volatility & Options
Black-Scholes pricer, all Greeks, and an implied-vol solver from scratch (verified against Deribit), plus a live BTC volatility smile and a GARCH volatility forecast. See [`p2_volatility/`](p2_volatility/).

## P3 — Risk & Portfolio Analytics
A multi-asset risk dashboard from scratch: Value-at-Risk (3 methods), Expected Shortfall, correlation/diversification analysis, and the Markowitz efficient frontier. See [`p3_risk/`](p3_risk/).

## Why a monorepo
One `core/` module (data, metrics, plotting) feeds every project — reusable-systems thinking, not disconnected scripts.

## Structure
```
core/            shared: data loaders, metrics, plotting theme
p1_backtester/   event-driven backtester + multi-asset study
p2_volatility/   options pricer, Greeks, IV solver, vol surface, GARCH
p3_risk/         VaR, Expected Shortfall, correlation, efficient frontier
tests/           correctness suite (87 passing)
```

## Quick start
```bash
pip install -r requirements.txt
python -m pytest tests/ -v
python -m p3_risk.run --source synthetic
```

---

