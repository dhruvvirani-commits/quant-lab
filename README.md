# quant-lab

**A build-and-learn portfolio of three interconnected quantitative-finance projects — built to learn the field by doing, not just reading.**

I came to AI through financial markets. `quant-lab` is where I turn years of self-taught, discretionary market analysis into rigorous, systematic, tested code — crossing the bridge from "trader who reads charts" to "person who reasons quantitatively about markets."

The three projects are not random. They are the three legs every quant stands on, and they share one data layer and one analytics core.

| Project | Pillar | Status |
|---|---|---|
| **P1 · Backtester** | Systematic research & testing rigor | ✅ v1 shipped |
| **P2 · Volatility** | Derivatives pricing & vol modeling | ⏳ planned |
| **P3 · Risk** | Portfolio risk & construction | ⏳ planned |

## Why a monorepo

One `core/` module (data, metrics, plotting) feeds all three projects — proof of reusable-systems thinking rather than three copy-pasted scripts. The projects interconnect: P2's volatility estimates feed P3's risk models; signals found in P3 can be tested in P1.

## Structure

```
core/           shared: data loaders, performance metrics, plotting theme
p1_backtester/  event-driven backtester for the A+ Sweep-Shift strategy
tests/          correctness suite (pytest)
data/           cached datasets (git-ignored)
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
python -m p1_backtester.run --source synthetic
```

See each project's own README for detail. Start with [`p1_backtester/`](p1_backtester/).

---

*Not financial advice. Built by Dhruv Virani — MSc Artificial Intelligence, ECE Paris.*
