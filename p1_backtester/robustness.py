"""
p1_backtester/robustness.py
---------------------------
Multi-asset, multi-year robustness test for the v2 filtered strategy.

WHY THIS EXISTS
---------------
A single backtest (one asset, one year, 11 trades) tells you almost nothing —
the sample is too small to separate skill from luck. This harness runs the
EXACT SAME v2 rules across several assets and several years, then:

  1. reports each slice separately  -> shows CONSISTENCY (or lack of it)
  2. pools every trade together     -> gives a real sample size for stats

The rules never change per slice. That is the anti-overfitting guarantee:
we are measuring one fixed strategy against many independent datasets, not
tuning a strategy to fit the data.

USAGE (from quant-lab/):
    python -m p1_backtester.robustness --source binance
    python -m p1_backtester.robustness --source synthetic   # smoke test
"""

from __future__ import annotations
import argparse
import pandas as pd

from core.data import load_binance, make_synthetic, validate
from core.metrics import trade_stats, max_drawdown, full_report
from p1_backtester.strategy_filtered import SweepShiftFiltered
from p1_backtester.strategy_regime import SweepShiftRegime
from p1_backtester.engine import run_backtest, BacktestConfig


# assets to test — same rules applied to each. Breadth = more trades w/o
# lowering quality. Add/remove freely.
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

STRATEGIES = {
    "v2": SweepShiftFiltered,   # filtered
    "v3": SweepShiftRegime,     # filtered + regime gate
}

# yearly windows
YEARS = [
    ("2021-01-01", "2022-01-01", "2021"),
    ("2022-01-01", "2023-01-01", "2022"),
    ("2023-01-01", "2024-01-01", "2023"),
    ("2024-01-01", "2025-01-01", "2024"),
]


def run_slice(df: pd.DataFrame, cfg: BacktestConfig, strat_cls):
    strat = strat_cls()                   # SAME rules every time
    eq, r, trades = run_backtest(df, strat, cfg)
    return eq, r, trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["binance", "synthetic"], default="synthetic")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--strategy", choices=list(STRATEGIES.keys()), default="v3",
                    help="v2 = filtered, v3 = filtered + regime gate (default)")
    args = ap.parse_args()
    strat_cls = STRATEGIES[args.strategy]
    print(f"Strategy: {args.strategy}  ({strat_cls.__name__})\n")

    cfg = BacktestConfig()
    rows = []              # per-slice summary
    all_trades = []        # pooled trades for the big-sample stats

    if args.source == "synthetic":
        print("Smoke test on synthetic data (results meaningless) ...\n")
        for seed, label in [(1, "synthA"), (2, "synthB"), (3, "synthC")]:
            df = validate(make_synthetic(n=12000, seed=seed))
            eq, r, trades = run_slice(df, cfg, strat_cls)
            rep = full_report(eq, r, trades)
            rows.append(("SYNTH", label, rep))
            if len(trades):
                all_trades.append(trades)
    else:
        for symbol in ASSETS:
            for start, end, label in YEARS:
                try:
                    print(f"Downloading {symbol} {args.interval} {label} ...")
                    df = load_binance(symbol, args.interval, start, end)
                    df = validate(df)
                    if len(df) < 500:
                        print(f"  (skipped {symbol} {label}: too little data)")
                        continue
                    eq, r, trades = run_slice(df, cfg, strat_cls)
                    rep = full_report(eq, r, trades)
                    rows.append((symbol, label, rep))
                    if len(trades):
                        all_trades.append(trades)
                except Exception as e:
                    print(f"  (skipped {symbol} {label}: {e})")

    # ---- per-slice table ----
    print("\n" + "=" * 78)
    print(f"  {'Asset':<9}{'Year':<8}{'Trades':>7}{'Win%':>8}{'PF':>7}{'Exp.R':>8}{'Return%':>10}{'MaxDD%':>9}")
    print("=" * 78)
    for symbol, label, rep in rows:
        print(f"  {symbol:<9}{label:<8}{rep['n_trades']:>7}"
              f"{rep['win_rate']*100:>8.1f}{rep['profit_factor']:>7.2f}"
              f"{rep['expectancy_R']:>8.3f}{rep['total_return_pct']:>10.2f}"
              f"{rep['max_drawdown_pct']:>9.2f}")
    print("=" * 78)

    # ---- pooled big-sample stats ----
    if all_trades:
        pooled = pd.concat(all_trades, ignore_index=True)
        s = trade_stats(pooled)
        print(f"\n  POOLED ACROSS ALL SLICES  (the real sample)")
        print(f"  {'-'*54}")
        print(f"  Total trades         {s['n_trades']}")
        print(f"  Win rate             {s['win_rate']*100:.1f}%")
        print(f"  Profit factor        {s['profit_factor']:.2f}")
        print(f"  Expectancy (R)       {s['expectancy_R']:.3f}")
        print(f"  Avg win / loss (R)   {s['avg_win_R']:.2f} / {s['avg_loss_R']:.2f}")
        print(f"  {'-'*54}")

        # honest verdict on sample size
        n = s['n_trades']
        print(f"\n  Sample-size reality check:")
        if n < 30:
            print(f"  {n} trades is STILL too small to conclude anything. Noise.")
        elif n < 100:
            print(f"  {n} trades — suggestive, but not yet robust. Treat with caution.")
        else:
            print(f"  {n} trades — a meaningful sample. Now the numbers mean something.")

        # is the edge consistent, or driven by one lucky slice?
        pos = sum(1 for _, _, rep in rows if rep['expectancy_R'] > 0)
        print(f"  Positive-expectancy slices: {pos}/{len(rows)} "
              f"({'consistent' if pos > len(rows)*0.6 else 'inconsistent — likely regime-dependent'})")
    else:
        print("\n  No trades generated across any slice.")


if __name__ == "__main__":
    main()
