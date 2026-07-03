# quant-lab

**A build-and-learn portfolio of quantitative-finance projects — built to learn the field by doing, with honest results over pretty ones.**

I came to AI through financial markets. `quant-lab` is where I turn years of self-taught, discretionary market analysis into rigorous, systematic, tested code.

| Project | Pillar | Status |
|---|---|---|
| **P1 · Backtester** | Systematic research & testing rigor | ✅ Shipped — honest conclusion reached |
| **P2 · Volatility & Options** | Derivatives pricing & vol modeling | ✅ Shipped — pricer, Greeks, IV solver, live vol surface |
| **P3 · Risk** | Portfolio risk & construction | ⏳ planned |

## P1 — Systematic Backtesting
Encoded a discretionary SMC strategy into an honest event-driven backtester, tested across 5 assets / 4 years / 230 trades, and concluded — without curve-fitting — that it is not a mechanical edge on the 15m timeframe. Reaching an honest negative conclusion is the point. See [`p1_backtester/`](p1_backtester/).

## P2 — Volatility & Options
Built the Black-Scholes pricer, all five Greeks, and an implied-volatility solver from scratch (each verified against known values), then pulled the live BTC options chain from Deribit to construct the implied-volatility smile and surface. See [`p2_volatility/`](p2_volatility/).

## Why a monorepo
One `core/` module (data, metrics, plotting) feeds every project — proof of reusable-systems thinking, not disconnected scripts.

## Structure
```
core/            shared: data loaders, performance metrics, plotting theme
p1_backtester/   event-driven backtester + multi-asset research study
p2_volatility/   options pricer, Greeks, IV solver, live vol surface
tests/           correctness suite (71 passing)
```

## Quick start
```bash
pip install -r requirements.txt
python -m pytest tests/ -v
python -m p2_volatility.run --source synthetic
```

---


