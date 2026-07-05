"""
p2_volatility/run_garch.py
--------------------------
Fit GARCH on real BTC price history and compare forecast vs realized vs implied
volatility.

USAGE (from quant-lab/):
    # On your machine, real BTC history from Binance (reuses P1's data loader):
    python -m p2_volatility.run_garch --source binance --implied 0.39

    # Anywhere, synthetic (pipeline test):
    python -m p2_volatility.run_garch --source synthetic
"""

from __future__ import annotations
import argparse
from p2_volatility.garch import fit_garch_forecast
from p2_volatility.garch_chart import plot_garch_comparison


def main():
    ap = argparse.ArgumentParser(description="GARCH volatility forecast")
    ap.add_argument("--source", choices=["binance", "synthetic"], default="synthetic")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--implied", type=float, default=None,
                    help="current ATM implied vol from the smile, e.g. 0.39")
    ap.add_argument("--window", type=int, default=30)
    args = ap.parse_args()

    # ---- get daily price history ----
    if args.source == "binance":
        from core.data import load_binance, validate
        print(f"Downloading {args.symbol} daily history from Binance ...")
        df = validate(load_binance(args.symbol, "1d", start="2022-01-01"))
        prices = df["close"]
    else:
        from p2_volatility.realized_vol import make_synthetic_prices
        print("Using synthetic prices (pipeline test) ...")
        prices = make_synthetic_prices(n=800, true_vol=0.55)
    print(f"Loaded {len(prices)} daily prices.")

    # ---- single next-day forecast ----
    fc, res = fit_garch_forecast(prices)
    print(f"\nGARCH next-day annualized vol forecast: {fc*100:.1f}%")

    # ---- forecast-vs-realized chart ----
    out, stats = plot_garch_comparison(prices, implied_vol=args.implied,
                                       window=args.window,
                                       outfile="garch_comparison.png")
    print(f"Chart saved -> {out}")
    print(f"\n  Avg GARCH forecast : {stats['avg_garch']*100:.1f}%")
    print(f"  Avg realized vol   : {stats['avg_realized']*100:.1f}%")
    if args.implied:
        premium = args.implied - stats["avg_realized"]
        print(f"  Implied vol (market): {args.implied*100:.1f}%")
        print(f"  Vol risk premium   : {premium*100:+.1f}% "
              f"({'market pricing MORE fear than realized' if premium>0 else 'realized above implied'})")


if __name__ == "__main__":
    main()
