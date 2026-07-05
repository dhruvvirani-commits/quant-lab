"""
p2_volatility/garch_chart.py
----------------------------
Build the GARCH story chart: forecasted volatility vs realized volatility over
time, with the current implied vol (from the options smile) marked for context.

This is the visual that completes P2 -- it puts a NUMBER on the volatility risk
premium the smile only hinted at.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from core.plotting import NAVY_BG, PANEL, TEAL, GOLD, RED, GREY, TEXT
from .garch import rolling_garch_vs_realized


def plot_garch_comparison(prices: pd.Series, implied_vol: float | None = None,
                          window: int = 30, outfile="garch_comparison.png"):
    """
    Plot GARCH-forecasted vol vs realized vol over time. If implied_vol is
    given (the current ATM implied vol from the smile), draw it as a line so
    the viewer sees the gap = the volatility risk premium.
    """
    comp = rolling_garch_vs_realized(prices, window=window)
    comp.to_csv("garch_data.csv")

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=NAVY_BG)
    ax.plot(comp.index, comp["garch_forecast"]*100, color=TEAL, linewidth=1.8,
            label="GARCH forecast")
    ax.plot(comp.index, comp["realized"]*100, color=GOLD, linewidth=1.6,
            alpha=0.9, label="Realized volatility")

    if implied_vol is not None:
        ax.axhline(implied_vol*100, color=RED, linestyle="--", linewidth=1.6,
                   label=f"Implied vol (market): {implied_vol*100:.0f}%")

    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color("#263B55")
    ax.tick_params(colors=GREY, labelsize=9)
    ax.set_ylabel("Annualized volatility (%)", color=GREY, fontsize=11)
    ax.set_title("BTC Volatility: GARCH Forecast vs Realized vs Implied",
                 color=TEXT, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, color="#1C2E44", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(facecolor=PANEL, edgecolor="#263B55", labelcolor=TEXT, fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(outfile, dpi=140, facecolor=NAVY_BG)
    plt.close(fig)

    # summary numbers
    avg_garch = comp["garch_forecast"].mean()
    avg_real = comp["realized"].mean()
    return outfile, {"avg_garch": avg_garch, "avg_realized": avg_real,
                     "implied": implied_vol}


if __name__ == "__main__":
    from p2_volatility.realized_vol import make_synthetic_prices
    prices = make_synthetic_prices(n=600, true_vol=0.55)
    out, stats = plot_garch_comparison(prices, implied_vol=0.68)
    print(f"Chart -> {out}")
    print(f"Avg GARCH forecast : {stats['avg_garch']*100:.1f}%")
    print(f"Avg realized       : {stats['avg_realized']*100:.1f}%")
