"""
p3_risk/dashboard.py
--------------------
The P3 deliverable: a multi-panel institutional risk dashboard combining VaR,
the return distribution, the correlation matrix, and the efficient frontier.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from core.plotting import NAVY_BG, PANEL, TEAL, GOLD, RED, GREY, TEXT
from .returns import portfolio_returns, annualize_vol, annualize_return
from .var_es import var_historical, expected_shortfall_historical, risk_report
from .correlation import correlation_matrix
from .frontier import random_portfolios, min_variance_portfolio, max_sharpe_portfolio


def _style(ax, title):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color("#263B55")
    ax.tick_params(colors=GREY, labelsize=8.5)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=8, loc="left")
    ax.grid(True, color="#1C2E44", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def build_dashboard(returns: pd.DataFrame, weights: np.ndarray | None = None,
                    capital: float = 10_000.0, outfile="risk_dashboard.png"):
    n_assets = returns.shape[1]
    if weights is None:
        weights = np.ones(n_assets) / n_assets
    port = portfolio_returns(returns, weights)

    fig = plt.figure(figsize=(14, 9), facecolor=NAVY_BG)
    fig.suptitle("Multi-Asset Portfolio Risk Dashboard",
                 color=TEXT, fontsize=16, fontweight="bold", y=0.98, x=0.09,
                 ha="left")
    gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.28,
                  left=0.06, right=0.97, top=0.88, bottom=0.08)

    # ---- Panel 1: return distribution with VaR/ES marked ----
    ax = fig.add_subplot(gs[0, 0])
    var = var_historical(port)
    es = expected_shortfall_historical(port)
    ax.hist(port*100, bins=50, color=TEAL, alpha=0.8, edgecolor=NAVY_BG)
    ax.axvline(-var*100, color=GOLD, linewidth=2, label=f"95% VaR {var*100:.1f}%")
    ax.axvline(-es*100, color=RED, linewidth=2, label=f"ES {es*100:.1f}%")
    ax.set_xlabel("Daily return (%)", color=GREY, fontsize=9)
    ax.legend(facecolor=PANEL, edgecolor="#263B55", labelcolor=TEXT, fontsize=8)
    _style(ax, "1  Return distribution + risk")

    # ---- Panel 2: correlation heatmap ----
    ax = fig.add_subplot(gs[0, 1])
    corr = correlation_matrix(returns)
    im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_assets)); ax.set_xticklabels(returns.columns, fontsize=9)
    ax.set_yticks(range(n_assets)); ax.set_yticklabels(returns.columns, fontsize=9)
    for i in range(n_assets):
        for j in range(n_assets):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    color="#0E1826", fontsize=9, fontweight="bold")
    _style(ax, "2  Correlation matrix")
    ax.grid(False)

    # ---- Panel 3: per-asset volatility ----
    ax = fig.add_subplot(gs[0, 2])
    vols = [annualize_vol(returns[a])*100 for a in returns.columns]
    ax.bar(returns.columns, vols, color=GOLD, alpha=0.85)
    for i, v in enumerate(vols):
        ax.text(i, v+1, f"{v:.0f}%", ha="center", color=TEXT, fontsize=8.5)
    ax.set_ylabel("Annualized vol (%)", color=GREY, fontsize=9)
    _style(ax, "3  Volatility by asset")

    # ---- Panel 4 (wide): efficient frontier ----
    ax = fig.add_subplot(gs[1, :2])
    cloud = random_portfolios(returns, n_portfolios=8000)
    sc = ax.scatter(cloud["vol"]*100, cloud["return"]*100, c=cloud["sharpe"],
                    cmap="viridis", s=6, alpha=0.5)
    mv = min_variance_portfolio(returns)
    ms = max_sharpe_portfolio(cloud, returns)
    ax.scatter(mv["vol"]*100, mv["return"]*100, color=TEAL, s=140, marker="*",
               edgecolor="white", linewidth=0.6, label="Min variance", zorder=5)
    ax.scatter(ms["vol"]*100, ms["return"]*100, color=RED, s=140, marker="*",
               edgecolor="white", linewidth=0.6, label="Max Sharpe", zorder=5)
    ax.set_xlabel("Risk — annualized volatility (%)", color=GREY, fontsize=9)
    ax.set_ylabel("Return (%)", color=GREY, fontsize=9)
    ax.legend(facecolor=PANEL, edgecolor="#263B55", labelcolor=TEXT, fontsize=9)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8); cb.set_label("Sharpe", color=GREY)
    cb.ax.tick_params(colors=GREY)
    _style(ax, "4  Efficient frontier (each dot = a portfolio)")

    # ---- Panel 5: risk summary text ----
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.set_facecolor(PANEL)
    rep = risk_report(port, capital=capital)
    ann_r = annualize_return(port)*100
    ann_v = annualize_vol(port)*100
    lines = [
        ("PORTFOLIO (equal weight)", TEXT, 13, "bold"),
        ("", GREY, 8, "normal"),
        (f"Annual return    {ann_r:+.1f}%", GREY, 11, "normal"),
        (f"Annual vol       {ann_v:.1f}%", GREY, 11, "normal"),
        (f"Sharpe           {ann_r/ann_v:.2f}", GREY, 11, "normal"),
        ("", GREY, 8, "normal"),
        (f"95% 1-day VaR    ${rep['var_historical_$']:,.0f}", GOLD, 11, "bold"),
        (f"95% 1-day ES     ${rep['es_historical_$']:,.0f}", RED, 11, "bold"),
        ("", GREY, 8, "normal"),
        (f"on ${capital:,.0f} capital", GREY, 9, "normal"),
    ]
    y = 0.95
    for text, color, size, weight in lines:
        ax.text(0.05, y, text, color=color, fontsize=size,
                fontweight=weight, transform=ax.transAxes, va="top")
        y -= 0.095

    fig.savefig(outfile, dpi=140, facecolor=NAVY_BG)
    plt.close(fig)
    return outfile


if __name__ == "__main__":
    from p3_risk.returns import make_synthetic_prices, build_returns_table
    rets = build_returns_table(make_synthetic_prices())
    print(build_dashboard(rets))
