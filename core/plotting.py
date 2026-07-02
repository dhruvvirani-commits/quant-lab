"""
core/plotting.py
----------------
Shared plotting theme + the standard backtest chart (equity curve, benchmark,
and drawdown). Dark institutional-terminal aesthetic to match the quant-lab
brand. Saves a PNG you can drop straight into a README or a LinkedIn post.
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ---- theme ----
NAVY_BG = "#0F1B2A"
PANEL = "#16263A"
TEAL = "#2EC4B6"
GOLD = "#E0A458"
RED = "#E05263"
GREY = "#8A9BA8"
TEXT = "#D8E2EA"


def _style(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#2A3D52")
    ax.tick_params(colors=GREY, labelsize=8)
    ax.grid(True, color="#22344A", linewidth=0.6, alpha=0.7)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)


def plot_backtest(strategy_equity: pd.Series,
                  benchmark_equity: pd.Series,
                  title: str,
                  outfile: str = "equity_curve.png"):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}, facecolor=NAVY_BG,
    )

    # normalise both to 100 for fair comparison
    s = strategy_equity / strategy_equity.iloc[0] * 100
    b = benchmark_equity / benchmark_equity.iloc[0] * 100

    ax1.plot(s.index, s.values, color=TEAL, linewidth=1.6, label="A+ Sweep-Shift")
    ax1.plot(b.index, b.values, color=GOLD, linewidth=1.2, alpha=0.8,
             label="Buy & Hold (BTC)")
    ax1.axhline(100, color=GREY, linewidth=0.7, linestyle="--", alpha=0.5)
    ax1.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Equity (indexed to 100)")
    ax1.legend(facecolor=PANEL, edgecolor="#2A3D52", labelcolor=TEXT, fontsize=9)
    _style(ax1)

    # drawdown of the strategy
    dd = (s / s.cummax() - 1) * 100
    ax2.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.5)
    ax2.plot(dd.index, dd.values, color=RED, linewidth=0.8)
    ax2.set_ylabel("Drawdown %")
    _style(ax2)

    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(outfile, dpi=130, facecolor=NAVY_BG)
    plt.close(fig)
    return outfile
