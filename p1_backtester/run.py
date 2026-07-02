"""
p1_backtester/run.py
--------------------
The entry point. Run this to backtest A+ Sweep-Shift and print the honest report.

USAGE
-----
From the quant-lab/ directory:

    # On YOUR machine, with real Binance data:
    python -m p1_backtester.run --source binance --start 2023-01-01 --end 2024-01-01

    # Anywhere, with synthetic data (for testing the engine):
    python -m p1_backtester.run --source synthetic

Add --exit trailing to test the trailing-stop variant instead of fixed 2.5R.
"""

from __future__ import annotations
import argparse
import pandas as pd

from core.data import load_binance, make_synthetic, validate
from core.metrics import full_report, print_report
from p1_backtester.strategy import SweepShiftStrategy
from p1_backtester.engine import run_backtest, buy_and_hold, BacktestConfig


def main():
    ap = argparse.ArgumentParser(description="A+ Sweep-Shift backtester")
    ap.add_argument("--source", choices=["binance", "synthetic"], default="synthetic")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--exit", choices=["fixed_rr", "trailing"], default="fixed_rr")
    ap.add_argument("--risk", type=float, default=0.01)
    args = ap.parse_args()

    # ---- load data ----
    if args.source == "binance":
        print(f"Downloading {args.symbol} {args.interval} from Binance ...")
        df = load_binance(args.symbol, args.interval, args.start, args.end)
    else:
        print("Using synthetic data (engine test mode) ...")
        df = make_synthetic(n=8000)
    df = validate(df)
    print(f"Loaded {len(df)} bars: {df.index[0]} -> {df.index[-1]}")

    # ---- run strategy ----
    strat = SweepShiftStrategy(n_pivot=3, atr_period=14, atr_mult=1.5, rr=2.5)
    cfg = BacktestConfig(risk_per_trade=args.risk, exit_mode=args.exit)
    equity, returns, trades = run_backtest(df, strat, cfg)

    # ---- benchmark ----
    bh_equity, bh_returns = buy_and_hold(df, cfg.initial_equity)

    # ---- report ----
    rep = full_report(equity, returns, trades)
    print_report(f"A+ SWEEP-SHIFT  ({args.exit})", rep)

    from core.metrics import sharpe_ratio, max_drawdown
    print(f"\n  Benchmark (Buy & Hold)")
    print(f"  {'-'*40}")
    print(f"  Total return         {(bh_equity.iloc[-1]/bh_equity.iloc[0]-1)*100:.2f}%")
    print(f"  Sharpe               {sharpe_ratio(bh_returns):.2f}")
    print(f"  Max drawdown         {max_drawdown(bh_equity)*100:.2f}%")

    if len(trades):
        print(f"\n  Exit reason breakdown:")
        print(trades["exit_reason"].value_counts().to_string())

    # ---- chart ----
    try:
        from core.plotting import plot_backtest
        out = plot_backtest(equity, bh_equity,
                            f"A+ Sweep-Shift vs Buy & Hold  ({args.symbol} {args.interval}, {args.exit})",
                            outfile="equity_curve.png")
        print(f"\n  Chart saved -> {out}")
    except Exception as e:
        print(f"\n  (plot skipped: {e})")

    return equity, returns, trades


if __name__ == "__main__":
    main()
