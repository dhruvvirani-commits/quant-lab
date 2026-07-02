"""
tests/test_engine.py
--------------------
Correctness tests. These prove the engine does what it claims — the difference
between "a script that runs" and "a backtester you can trust."

Run from quant-lab/:  python -m pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest

from core.data import make_synthetic, validate
from core.metrics import sharpe_ratio, max_drawdown, trade_stats
from p1_backtester.strategy import atr, swing_pivots, SweepShiftStrategy
from p1_backtester.engine import run_backtest, buy_and_hold, BacktestConfig, Costs


# ---------- data integrity ----------
def test_synthetic_is_valid():
    df = make_synthetic(n=500)
    validate(df)  # raises if malformed
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["close"]).all()


def test_synthetic_is_deterministic():
    a = make_synthetic(n=300, seed=1)
    b = make_synthetic(n=300, seed=1)
    pd.testing.assert_frame_equal(a, b)


# ---------- indicators ----------
def test_atr_positive_and_lagged():
    df = make_synthetic(n=200)
    a = atr(df, 14)
    assert a.iloc[:13].isna().all()          # first 13 undefined
    assert (a.dropna() > 0).all()            # ATR strictly positive


def test_swing_pivots_no_lookahead_shape():
    df = make_synthetic(n=300)
    hi, lo = swing_pivots(df, n=3)
    # pivots cannot exist in the first/last n bars (need both sides)
    assert not hi.iloc[:3].any() and not hi.iloc[-3:].any()
    assert not lo.iloc[:3].any() and not lo.iloc[-3:].any()


def test_swing_high_is_local_max():
    df = make_synthetic(n=400)
    hi, _ = swing_pivots(df, n=3)
    for idx in np.where(hi.values)[0]:
        window = df["high"].values[idx-3: idx+4]
        assert df["high"].values[idx] == window.max()


# ---------- metrics sanity ----------
def test_max_drawdown_bounds():
    eq = pd.Series([100, 120, 90, 130, 80, 140])
    mdd = max_drawdown(eq)
    assert -1.0 <= mdd <= 0.0
    # peak 130 -> trough 80 = -38.46%
    assert abs(mdd - (80/130 - 1)) < 1e-9


def test_sharpe_zero_on_flat():
    r = pd.Series(np.zeros(100))
    assert sharpe_ratio(r) == 0.0


def test_trade_stats_known_values():
    trades = pd.DataFrame({"pnl_R": [2.5, -1, -1, 2.5, -1],
                           "return_pct": [0]*5})
    s = trade_stats(trades)
    assert s["n_trades"] == 5
    assert abs(s["win_rate"] - 0.4) < 1e-9
    # gross win = 5.0, gross loss = 3.0 -> PF = 1.667
    assert abs(s["profit_factor"] - 5/3) < 1e-6


# ---------- engine behaviour ----------
def test_no_costs_vs_costs_direction():
    """With costs, net P&L must be <= P&L without costs (costs only subtract)."""
    df = make_synthetic(n=4000)
    strat = SweepShiftStrategy()

    free = BacktestConfig(costs=Costs(0, 0))
    paid = BacktestConfig(costs=Costs(4, 2))
    eq_free, _, _ = run_backtest(df, strat, free)
    eq_paid, _, _ = run_backtest(df, strat, paid)
    assert eq_paid.iloc[-1] <= eq_free.iloc[-1] + 1e-6


def test_risk_per_trade_respected():
    """A losing trade stopped out should lose about risk_per_trade of equity."""
    df = make_synthetic(n=6000)
    strat = SweepShiftStrategy()
    cfg = BacktestConfig(risk_per_trade=0.01, costs=Costs(0, 0))
    _, _, trades = run_backtest(df, strat, cfg)
    stopped = trades[trades["exit_reason"] == "stop"]
    if len(stopped):
        # stop loss should be about -1R (costs=0 here)
        assert stopped["pnl_R"].mean() < 0
        assert abs(stopped["pnl_R"].mean() + 1.0) < 0.25


def test_target_trades_are_positive_R():
    df = make_synthetic(n=6000)
    strat = SweepShiftStrategy(rr=2.5)
    cfg = BacktestConfig(costs=Costs(0, 0))
    _, _, trades = run_backtest(df, strat, cfg)
    tgt = trades[trades["exit_reason"] == "target"]
    if len(tgt):
        # fixed 2.5R target -> winners near +2.5R
        assert (tgt["pnl_R"] > 2.0).all()


def test_buy_and_hold_matches_price_move():
    df = make_synthetic(n=1000)
    eq, _ = buy_and_hold(df, 10_000)
    expected = 10_000 * (df["close"].iloc[-1] / df["close"].iloc[0])
    assert abs(eq.iloc[-1] - expected) < 1e-6


def test_equity_never_nan():
    df = make_synthetic(n=3000)
    eq, ret, _ = run_backtest(df, SweepShiftStrategy())
    assert not eq.isna().any()
    assert not ret.isna().any()


# ---------- v2 filtered strategy ----------
from p1_backtester.strategy_filtered import SweepShiftFiltered, htf_trend_bias


def test_filters_only_remove_trades():
    """v2 (filtered) must never produce MORE signals than v1 (raw)."""
    df = make_synthetic(n=8000)
    v1 = SweepShiftStrategy().generate(df)
    v2 = SweepShiftFiltered().generate(df)
    assert len(v2) <= len(v1)


def test_htf_bias_values_valid():
    """HTF bias is only -1, 0, or +1."""
    df = make_synthetic(n=3000)
    bias = htf_trend_bias(df)
    assert set(bias.unique()).issubset({-1.0, 0.0, 1.0})


def test_htf_bias_no_lookahead():
    """
    The bias series is built from CLOSED 4H bars (shifted by 1). The very first
    15m bars should therefore have bias 0 (no closed 4H bar yet).
    """
    df = make_synthetic(n=3000)
    bias = htf_trend_bias(df)
    assert bias.iloc[0] == 0.0


def test_v2_signals_respect_session():
    """Every v2 entry must fall inside the session window (12:00-16:00 UTC)."""
    df = make_synthetic(n=8000)
    strat = SweepShiftFiltered(session_utc=(12, 16))
    sigs = strat.generate(df)
    for s in sigs:
        hour = df.index[s.bar_index].hour
        assert 12 <= hour < 16


# ---------- v3 regime filter ----------
from p1_backtester.strategy_regime import SweepShiftRegime
from p1_backtester.strategy import adx


def test_adx_in_valid_range():
    """ADX must be between 0 and 100."""
    df = make_synthetic(n=2000)
    a = adx(df, 14).dropna()
    assert (a >= 0).all() and (a <= 100).all()


def test_regime_only_removes_trades():
    """v3 (regime) must never produce MORE signals than v2 (filtered)."""
    df = make_synthetic(n=12000, seed=2)
    v2 = SweepShiftFiltered().generate(df)
    v3 = SweepShiftRegime().generate(df)
    assert len(v3) <= len(v2)


def test_regime_signals_have_trend():
    """Every v3 entry must occur when ADX >= threshold."""
    df = make_synthetic(n=12000, seed=2)
    strat = SweepShiftRegime(adx_min=20.0)
    a = adx(df, strat.adx_period).values
    for s in strat.generate(df):
        assert a[s.bar_index] >= 20.0
