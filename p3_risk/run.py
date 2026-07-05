"""
p3_risk/run.py
--------------
Entry point for Project 3. Builds the multi-asset risk dashboard.

USAGE (from quant-lab/):
    # On your machine, real data from Binance (crypto) via P1's loader:
    python -m p3_risk.run --source binance

    # Anywhere, synthetic multi-asset data:
    python -m p3_risk.run --source synthetic
"""

from __future__ import annotations
import argparse
import numpy as np

from p3_risk.returns import build_returns_table, make_synthetic_prices
from p3_risk.dashboard import build_dashboard
from p3_risk.var_es import risk_report
from p3_risk.returns import portfolio_returns


def main():
    ap = argparse.ArgumentParser(description="P3 - Multi-Asset Risk")
    ap.add_argument("--source", choices=["binance", "synthetic"], default="synthetic")
    ap.add_argument("--capital", type=float, default=10_000.0)
    args = ap.parse_args()

    if args.source == "binance":
        # crypto assets that Binance serves; reuse P1's loader
        from core.data import load_binance, validate
        symbols = {"BTC": "BTCUSDT", "ETH": "ETHUSDT",
                   "SOL": "SOLUSDT", "BNB": "BNBUSDT"}
        prices = {}
        for name, sym in symbols.items():
            print(f"Downloading {sym} ...")
            df = validate(load_binance(sym, "1d", start="2022-01-01"))
            prices[name] = df["close"]
    else:
        print("Using synthetic multi-asset data ...")
        prices = make_synthetic_prices()

    rets = build_returns_table(prices)
    print(f"Loaded {len(rets)} days across {rets.shape[1]} assets: {list(rets.columns)}")

    out = build_dashboard(rets, capital=args.capital, outfile="risk_dashboard.png")
    print(f"\nDashboard saved -> {out}")

    # text summary
    port = portfolio_returns(rets, np.ones(rets.shape[1])/rets.shape[1])
    rep = risk_report(port, capital=args.capital)
    print(f"\n  Equal-weight portfolio, 95% 1-day risk on ${args.capital:,.0f}:")
    print(f"    VaR (historical): ${rep['var_historical_$']:,.0f} ({rep['var_historical_pct']:.2f}%)")
    print(f"    ES  (historical): ${rep['es_historical_$']:,.0f} ({rep['es_historical_pct']:.2f}%)")


if __name__ == "__main__":
    main()
