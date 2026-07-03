"""
p2_volatility/run.py
--------------------
Entry point for Project 2. Pulls the options chain, builds the vol smile and
surface, and reports the implied-vs-realized volatility gap.

USAGE (from quant-lab/):
    # On your machine, live Deribit data:
    python -m p2_volatility.run --source deribit --currency BTC

    # Anywhere, synthetic data (verifies the pipeline):
    python -m p2_volatility.run --source synthetic
"""

from __future__ import annotations
import argparse
import numpy as np

from p2_volatility.deribit import load_deribit_chain, make_synthetic_chain
from p2_volatility.vol_surface import compute_chain_iv, plot_smile, plot_surface_3d
from p2_volatility.realized_vol import realized_vol_close_to_close


def main():
    ap = argparse.ArgumentParser(description="P2 - Volatility & Options")
    ap.add_argument("--source", choices=["deribit", "synthetic"], default="synthetic")
    ap.add_argument("--currency", default="BTC")
    ap.add_argument("--rate", type=float, default=0.05, help="risk-free rate")
    args = ap.parse_args()

    # ---- 1. get the options chain ----
    if args.source == "deribit":
        print(f"Downloading live {args.currency} options chain from Deribit ...")
        chain = load_deribit_chain(args.currency)
    else:
        print("Using synthetic options chain (pipeline test) ...")
        chain = make_synthetic_chain()
    print(f"Loaded {len(chain)} options across {chain['T'].nunique()} expiries.")

    # ---- 2. invert every price to implied vol ----
    print("Computing implied volatility for each option (our solver) ...")
    iv = compute_chain_iv(chain, r=args.rate)
    print(f"Solved IV for {len(iv)} options.")

    # ---- 3. build the smile + surface ----
    smile = plot_smile(iv, "vol_smile.png")
    surf = plot_surface_3d(iv, "vol_surface_3d.png")
    iv.to_csv("vol_chain_iv.csv", index=False)
    print(f"IV data saved -> vol_chain_iv.csv")
    print(f"\nVol smile   -> {smile}")
    print(f"Vol surface -> {surf}")

    # ---- 4. summary of the smile ----
    atm = iv[np.abs(iv["log_moneyness"]) < 0.05]["iv"].mean()
    wings = iv[np.abs(iv["log_moneyness"]) > 0.25]["iv"].mean()
    print(f"\n  ATM implied vol : {atm*100:.1f}%")
    print(f"  Wing implied vol: {wings*100:.1f}%")
    print(f"  Smile curvature : {'present (wings > ATM)' if wings > atm else 'flat/inverted'}")

    # ---- 5. implied vs realized (needs a price history) ----
    # On your machine you can feed real BTC history here (yfinance/Binance from
    # P1's core.data). We show the structure using the chain's ATM implied vol.
    print(f"\n  (Implied vs realized: feed a price series to realized_vol on your")
    print(f"   machine to compute the volatility risk premium.)")


if __name__ == "__main__":
    main()
