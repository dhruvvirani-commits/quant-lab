"""p1_backtester/analysis.py - analysis figure + CSV export."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.data import load_binance, make_synthetic, validate
from core.plotting import NAVY_BG, PANEL, TEAL, GOLD, RED, GREY, TEXT
from p1_backtester.strategy_filtered import SweepShiftFiltered
from p1_backtester.strategy_regime import SweepShiftRegime
from p1_backtester.engine import run_backtest, BacktestConfig

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
YEARS = [("2021-01-01", "2022-01-01", "2021"), ("2022-01-01", "2023-01-01", "2022"), ("2023-01-01", "2024-01-01", "2023"), ("2024-01-01", "2025-01-01", "2024")]
STRATS = {"v2": SweepShiftFiltered, "v3": SweepShiftRegime}

def _style(ax, title):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values(): s.set_color("#2A3D52")
    ax.tick_params(colors=GREY, labelsize=8)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, color="#22344A", linewidth=0.5, alpha=0.6)

def collect_trades(source, strat_cls, interval="15m"):
    frames = []
    if source == "synthetic":
        for seed, label in [(1, "2021"), (2, "2022"), (3, "2023"), (4, "2024")]:
            for asset in ["SYN-A", "SYN-B"]:
                df = validate(make_synthetic(n=12000, seed=seed + hash(asset) % 5))
                _, _, tr = run_backtest(df, strat_cls(), BacktestConfig())
                if len(tr):
                    tr["asset"], tr["year"] = asset, label
                    frames.append(tr)
    else:
        for asset in ASSETS:
            for start, end, label in YEARS:
                try:
                    df = validate(load_binance(asset, interval, start, end))
                    if len(df) < 500: continue
                    _, _, tr = run_backtest(df, strat_cls(), BacktestConfig())
                    if len(tr):
                        tr["asset"], tr["year"] = asset, label
                        frames.append(tr)
                    print(f"  {asset} {label}: {len(tr)} trades")
                except Exception as e:
                    print(f"  (skip {asset} {label}: {e})")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def build_panel(trades, outfile="analysis_panel.png"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=NAVY_BG)
    fig.suptitle("A+ Sweep-Shift - Analysis of trades across 5 assets, 4 years", color=TEXT, fontsize=15, fontweight="bold", y=0.98)
    ax = axes[0, 0]
    r = trades["pnl_R"]
    ax.hist(r, bins=30, color=TEAL, alpha=0.8, edgecolor=NAVY_BG)
    ax.axvline(0, color=GOLD, linestyle="--", linewidth=1)
    ax.axvline(r.mean(), color=RED, linewidth=1.5, label=f"mean {r.mean():.2f}R")
    ax.legend(facecolor=PANEL, edgecolor="#2A3D52", labelcolor=TEXT, fontsize=8)
    ax.set_xlabel("Trade P&L (R-multiples)", color=GREY, fontsize=9)
    _style(ax, "1. Distribution of trade outcomes")
    ax = axes[0, 1]
    piv = trades.assign(win=(trades["pnl_R"] > 0)).pivot_table(index="asset", columns="year", values="win", aggfunc="mean") * 100
    ax.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=60, aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, color=GREY, fontsize=8)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, color=GREY, fontsize=8)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            if not np.isnan(v): ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="black", fontsize=8, fontweight="bold")
    _style(ax, "2. Win rate % by asset and year"); ax.grid(False)
    ax = axes[1, 0]
    for asset in sorted(trades["asset"].unique()):
        sub = trades[trades["asset"] == asset].reset_index()
        ax.plot(sub.index, sub["pnl_R"].cumsum(), linewidth=1.3, label=asset)
    ax.axhline(0, color=GREY, linestyle="--", linewidth=0.8)
    ax.legend(facecolor=PANEL, edgecolor="#2A3D52", labelcolor=TEXT, fontsize=8, ncol=2)
    ax.set_xlabel("Trade number", color=GREY, fontsize=9); ax.set_ylabel("Cumulative R", color=GREY, fontsize=9)
    _style(ax, "3. Cumulative R by asset (consistency check)")
    ax = axes[1, 1]
    by_year = trades.groupby("year")["pnl_R"].mean()
    colors = [TEAL if v > 0 else RED for v in by_year.values]
    ax.bar(by_year.index, by_year.values, color=colors, alpha=0.85)
    ax.axhline(0, color=GREY, linewidth=0.8)
    for i, v in enumerate(by_year.values): ax.text(i, v + (0.01 if v >= 0 else -0.03), f"{v:.3f}", ha="center", color=TEXT, fontsize=8)
    ax.set_ylabel("Expectancy (R)", color=GREY, fontsize=9)
    _style(ax, "4. Expectancy by year (regime dependence)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outfile, dpi=130, facecolor=NAVY_BG)
    plt.close(fig)
    return outfile

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["binance", "synthetic"], default="synthetic")
    ap.add_argument("--strategy", choices=list(STRATS.keys()), default="v2")
    ap.add_argument("--interval", default="15m")
    args = ap.parse_args()
    print(f"Collecting trades ({args.strategy}) ...")
    trades = collect_trades(args.source, STRATS[args.strategy], args.interval)
    if trades.empty:
        print("No trades collected."); return
    trades.to_csv("trades_export.csv", index=False)
    print(f"\n{len(trades)} trades -> trades_export.csv")
    print(f"Analysis figure -> {build_panel(trades)}")
    print(f"\n  Pooled win rate : {(trades['pnl_R']>0).mean()*100:.1f}%")
    print(f"  Pooled expectancy: {trades['pnl_R'].mean():.3f}R")

if __name__ == "__main__":
    main()