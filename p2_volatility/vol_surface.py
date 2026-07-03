"""
p2_volatility/vol_surface.py
----------------------------
Build and visualize the implied-volatility smile and surface from an options
chain. This is the payoff of Project 2.

============================================================================
 CONCEPT CHECK — Smile & Surface  (full detail: Part 6 of your doc)
============================================================================
Black-Scholes assumes ONE constant volatility. But when you extract implied vol
from real prices at different strikes, you DON'T get a flat line -- you get a
curve (the "smile" or "skew"). Plotting IV vs strike AND expiry gives a 3-D
"volatility surface": the market's full map of how it prices uncertainty. The
smile is the fingerprint of everything BSM leaves out (jumps, fat tails).
============================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .implied_vol import implied_vol
from core.plotting import NAVY_BG, PANEL, TEAL, GOLD, RED, GREY, TEXT

def _otm_only(chain_iv):
    """Keep only OTM options: puts below spot, calls above spot. One clean value per strike."""
    spot = chain_iv["underlying_price"].iloc[0]
    puts_below = (chain_iv["type"] == "put") & (chain_iv["strike"] <= spot)
    calls_above = (chain_iv["type"] == "call") & (chain_iv["strike"] > spot)
    return chain_iv[puts_below | calls_above].copy()

def _filter_liquid(chain_iv, moneyness_lo=0.5, moneyness_hi=1.6):
    """Keep only near-the-money strikes where prices are reliable."""
    spot = chain_iv["underlying_price"].iloc[0]
    m = chain_iv["strike"] / spot
    return chain_iv[(m >= moneyness_lo) & (m <= moneyness_hi)].copy()


def _select_expiries(chain_iv, targets_days=(7, 30, 60, 90, 180)):
    """Pick a few clean representative expiries instead of all of them."""
    avail = sorted(chain_iv["T"].unique())
    chosen = []
    for td in targets_days:
        target_T = td / 365
        closest = min(avail, key=lambda x: abs(x - target_T))
        if closest not in chosen:
            chosen.append(closest)
    return chain_iv[chain_iv["T"].isin(chosen)].copy()

def compute_chain_iv(chain: pd.DataFrame, r: float = 0.05) -> pd.DataFrame:
    """
    For every option in the chain, invert its market price to implied vol using
    OUR solver. Adds an 'iv' column (as a fraction, e.g. 0.55 = 55%).

    We drop options where the solve fails (price below intrinsic, illiquid, etc).
    """
    out = chain.copy()
    ivs = []
    for _, row in out.iterrows():
        iv = implied_vol(row["mark_price_usd"], row["underlying_price"],
                         row["strike"], row["T"], r, row["type"])
        ivs.append(iv)
    out["iv"] = ivs
    out = out[out["iv"].notna() & (out["iv"] > 0.01) & (out["iv"] < 5)]
    # log-moneyness is the natural x-axis for a smile
    out["log_moneyness"] = np.log(out["strike"] / out["underlying_price"])
    return out.reset_index(drop=True)


def _style(ax, title):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color("#263B55")
    ax.tick_params(colors=GREY, labelsize=9)
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, color="#1C2E44", linewidth=0.6, alpha=0.8)


def plot_smile(chain_iv: pd.DataFrame, outfile="vol_smile.png"):
    """
    Plot the volatility smile: implied vol vs strike, one curve per expiry.
    This is the v1 deliverable -- clear proof the smile exists.
    """
    chain_iv = _select_expiries(_filter_liquid(_otm_only(chain_iv)))
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=NAVY_BG)
    
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=NAVY_BG)
    colors = [TEAL, GOLD, RED, "#7C9CBF", "#43C97F"]
    for i, (T, grp) in enumerate(sorted(chain_iv.groupby("T"))):
        grp = grp.sort_values("strike")
        days = int(round(T * 365))
        ax.plot(grp["strike"], grp["iv"] * 100, "o-", markersize=4,
                linewidth=1.6, color=colors[i % len(colors)],
                label=f"{days}d expiry")
    spot = chain_iv["underlying_price"].iloc[0]
    ax.axvline(spot, color=GREY, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(spot, ax.get_ylim()[1]*0.95, " spot", color=GREY, fontsize=9)
    ax.set_xlabel("Strike price", color=GREY, fontsize=11)
    ax.set_ylabel("Implied volatility (%)", color=GREY, fontsize=11)
    ax.legend(facecolor=PANEL, edgecolor="#263B55", labelcolor=TEXT, fontsize=9)
    _style(ax, "BTC Implied Volatility Smile  (Deribit options chain)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=140, facecolor=NAVY_BG)
    plt.close(fig)
    return outfile


def plot_surface_3d(chain_iv: pd.DataFrame, outfile="vol_surface_3d.png"):
    """
    3-D volatility surface: IV as a function of (strike, time-to-expiry).
    Stretch deliverable. Uses matplotlib's 3D; interactive Plotly is an option
    on your machine.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(11, 8), facecolor=NAVY_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(NAVY_BG)
    
    chain_iv = _filter_liquid(_otm_only(chain_iv), 0.6, 1.5)
    x = chain_iv["log_moneyness"].values

    x = chain_iv["log_moneyness"].values
    y = (chain_iv["T"] * 365).values
    z = (chain_iv["iv"] * 100).values

    surf = ax.plot_trisurf(x, y, z, cmap="viridis", alpha=0.9,
                           edgecolor="none")
    ax.set_xlabel("log-moneyness  ln(K/S)", color=GREY, fontsize=9, labelpad=8)
    ax.set_ylabel("days to expiry", color=GREY, fontsize=9, labelpad=8)
    ax.set_zlabel("implied vol (%)", color=GREY, fontsize=9, labelpad=8)
    ax.set_title("BTC Volatility Surface", color=TEXT, fontsize=14,
                 fontweight="bold", pad=16)
    ax.tick_params(colors=GREY, labelsize=8)
    ax.xaxis.pane.set_facecolor(PANEL)
    ax.yaxis.pane.set_facecolor(PANEL)
    ax.zaxis.pane.set_facecolor(PANEL)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12)
    fig.savefig(outfile, dpi=140, facecolor=NAVY_BG)
    plt.close(fig)
    return outfile


if __name__ == "__main__":
    from .deribit import make_synthetic_chain
    chain = make_synthetic_chain()
    iv = compute_chain_iv(chain)
    print(f"Computed IV for {len(iv)} options")
    print(plot_smile(iv))
    print(plot_surface_3d(iv))
