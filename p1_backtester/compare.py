"""
p1_backtester/compare.py
------------------------
Runs v1 (unfiltered) vs v2 (filtered) side by side on the same data, so the
effect of the trader's real quality filters is measurable and honest.

USAGE (from quant-lab/):
    python -m p1_backtester.compare --source binance --start 2023-01-01 --end 2024-01-01
    python -m p1_backtester.compare --source synthetic
"""

from __future__ import annotations
import argparse

from core.data import load_binance, make_synthetic, validate
from core.metrics import full_report
from p1_backtester.strategy import SweepShiftStrategy
from p1_backtester.strategy_filtered import SweepShiftFiltered
from p1_backtester.engine import run_backtest, buy_and_hold, BacktestConfig


def _fmt(rep):
    return {
        "Total return %": f"{rep['total_return_pct']:.2f}",
        "Sharpe": f"{rep['sharpe']:.2f}",
        "Max DD %": f"{rep['max_drawdown_pct']:.2f}",
        "Trades": f"{rep['n_trades']}",
        "Win rate %": f"{rep['win_rate']*100:.1f}",
        "Profit factor": f"{rep['profit_factor']:.2f}",
        "Expectancy R": f"{rep['expectancy_R']:.3f}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["binance", "synthetic"], default="synthetic")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    if args.source == "binance":
        print(f"Downloading {args.symbol} {args.interval} from Binance ...")
        df = load_binance(args.symbol, args.interval, args.start, args.end)
    else:
        print("Using synthetic data ...")
        df = make_synthetic(n=8000)
    df = validate(df)
    print(f"Loaded {len(df)} bars: {df.index[0]} -> {df.index[-1]}\n")

    cfg = BacktestConfig()

    # v1
    eq1, r1, t1 = run_backtest(df, SweepShiftStrategy(), cfg)
    rep1 = full_report(eq1, r1, t1)

    # v2
    eq2, r2, t2 = run_backtest(df, SweepShiftFiltered(), cfg)
    rep2 = full_report(eq2, r2, t2)

    # benchmark
    bh_eq, bh_r = buy_and_hold(df, cfg.initial_equity)
    bh_ret = (bh_eq.iloc[-1] / bh_eq.iloc[0] - 1) * 100

    f1, f2 = _fmt(rep1), _fmt(rep2)
    keys = list(f1.keys())
    w = max(len(k) for k in keys)

    print("=" * 60)
    print(f"  {'Metric':<{w}}   {'v1 (raw)':>12}   {'v2 (filtered)':>14}")
    print("=" * 60)
    for k in keys:
        print(f"  {k:<{w}}   {f1[k]:>12}   {f2[k]:>14}")
    print("=" * 60)
    print(f"  {'Buy & Hold return %':<{w}}   {bh_ret:>12.2f}")
    print("=" * 60)

    # chart both
    try:
        from core.plotting import plot_backtest
        plot_backtest(eq2, bh_eq,
                      f"A+ Sweep-Shift v2 (filtered) vs Buy & Hold  ({args.symbol} {args.interval})",
                      outfile="equity_curve_v2.png")
        print("\n  v2 chart saved -> equity_curve_v2.png")
    except Exception as e:
        print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    main()
