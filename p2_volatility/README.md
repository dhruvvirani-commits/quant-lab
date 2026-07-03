# Project 2 — Volatility & Options

**A from-scratch options-pricing engine and a live BTC volatility surface — built to learn derivatives by building them, not just reading about them.**

Options are, at their core, a way to trade volatility. This project builds the full machinery from first principles: price an option, measure its risk, extract the market's volatility expectation, and visualize how that expectation curves across strikes and expiries.

---

## What this is

Everything here is built from scratch (no options library) and verified against known values:

- **Black-Scholes pricer** — fair value of European calls and puts from the 5 inputs (S, K, T, r, σ)
- **The five Greeks** — delta, gamma, vega, theta, rho, each verified numerically against finite differences
- **Implied-volatility solver** — invert an option's market price to find the volatility it implies (bisection + Newton-Raphson)
- **Live volatility surface** — pull the BTC options chain from Deribit and build the implied-vol smile and 3-D surface
- **Implied vs realized vol** — the volatility risk premium, the structural edge behind option selling

## The core idea

Four Black-Scholes inputs are observable; only volatility isn't. So options pricing really reduces to one question: *how volatile will the asset be?* Everything interesting — the smile, the skew, the surface — is about volatility. This project makes that concrete.

## Why the smile matters

Black-Scholes assumes one constant volatility. But invert real market prices and you find implied vol **curves** with strike — the "smile." That curve is the market correcting Black-Scholes' flawed assumptions (no jumps, constant vol). The smile is the fingerprint of everything the model leaves out. Building it from a live crypto options chain is the centerpiece of this project.

## How to run it

```bash
pip install -r requirements.txt
python -m pytest tests/test_black_scholes.py tests/test_greeks.py tests/test_implied_vol.py tests/test_vol_surface.py tests/test_realized_vol.py -v

# On your machine — live Deribit BTC options:
python -m p2_volatility.run --source deribit --currency BTC

# Anywhere — synthetic chain (verifies the full pipeline):
python -m p2_volatility.run --source synthetic
```

## Correctness is proven, not assumed

Every component is tested:

- **Pricer**: matches the canonical textbook value (S=K=100, T=1, r=5%, σ=20% → call = 10.4506) and satisfies put-call parity to 1e-6
- **Greeks**: each analytical Greek matches an independent finite-difference estimate
- **IV solver**: round-trip — price with a known σ, recover that exact σ
- **Smile pipeline**: impose a known smile on a synthetic chain, invert every price, recover the smile with zero error

## Structure

```
core/            shared plotting theme (reused from P1)
p2_volatility/
  black_scholes.py   the pricer (calls & puts) from scratch
  greeks.py          all five Greeks, analytically
  implied_vol.py     IV solver — bisection + Newton-Raphson
  deribit.py         live options-chain loader + synthetic generator
  vol_surface.py     smile & 3-D surface construction and plots
  realized_vol.py    realized vol + volatility risk premium
  run.py             end-to-end pipeline
tests/               correctness suite (per-module)
```

## What this project demonstrates

- Implementing **graduate-level derivatives math** from first principles
- **Numerical methods** (root-finding) for a problem with no closed-form solution
- Working with **live crypto-native market data** (Deribit)
- **Verification discipline** — every formula checked against known values or independent methods

---

*Part of [quant-lab](../). Not financial advice. Built by Dhruv Virani — MSc Artificial Intelligence, ECE Paris.*
