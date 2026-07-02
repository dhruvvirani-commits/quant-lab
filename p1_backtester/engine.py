"""
p1_backtester/engine.py
-----------------------
Event-driven backtest engine.

Design principles (the things that make a backtest HONEST):
  * No look-ahead. A trade opened at bar i is filled at bar i's close; exits
    are checked from bar i+1 onward using that bar's high/low.
  * Realistic costs. Every fill pays commission + slippage.
  * One position at a time (v1). Fixed fractional risk per trade.
  * Intrabar exit priority: if a single bar touches BOTH stop and target, we
    assume the STOP hit first (conservative — never flatter than reality).

Outputs: equity curve (per bar), per-bar returns, and a trades table with
P&L in both % and R-multiples.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from .strategy import SweepShiftStrategy, Signal


@dataclass
class Costs:
    commission_bps: float = 4.0     # per side, in basis points (0.04% ~ Binance taker)
    slippage_bps: float = 2.0       # per side, basis points of adverse fill

    @property
    def per_side(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10_000.0


@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.01     # 1% of equity risked per trade
    costs: Costs = field(default_factory=Costs)
    exit_mode: str = "fixed_rr"      # "fixed_rr" (v1) or "trailing" (v1.1)
    trail_atr_mult: float = 2.0      # used only when exit_mode == "trailing"


def run_backtest(df: pd.DataFrame, strategy: SweepShiftStrategy,
                 config: BacktestConfig | None = None):
    """
    Execute `strategy` over `df`. Returns (equity_series, returns_series,
    trades_df). Trades table columns:
        entry_time, exit_time, direction, entry, exit, stop, target,
        pnl_pct (of equity), pnl_R, bars_held, exit_reason
    """
    cfg = config or BacktestConfig()
    signals = strategy.generate(df)
    sig_by_bar = {s.bar_index: s for s in signals}

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    times = df.index
    m = len(df)

    equity = cfg.initial_equity
    equity_curve = np.full(m, np.nan)

    open_trade = None          # dict describing the live position
    trades = []
    cps = cfg.costs.per_side   # cost per side as a fraction

    for i in range(m):
        # ---- manage an open trade first (exits use THIS bar's range) ----
        if open_trade is not None:
            d = open_trade["direction"]
            stop = open_trade["stop"]
            target = open_trade["target"]
            hit_stop = hit_target = False

            if d == +1:
                if lows[i] <= stop:
                    hit_stop = True
                if highs[i] >= target:
                    hit_target = True
            else:
                if highs[i] >= stop:
                    hit_stop = True
                if lows[i] <= target:
                    hit_target = True

            # trailing-stop update (v1.1): ratchet the stop toward price
            if cfg.exit_mode == "trailing" and not (hit_stop or hit_target):
                trail = open_trade["trail_dist"]
                if d == +1:
                    open_trade["stop"] = max(stop, highs[i] - trail)
                else:
                    open_trade["stop"] = min(stop, lows[i] + trail)

            exit_price = None
            reason = None
            if hit_stop and hit_target:
                exit_price, reason = stop, "stop (ambiguous bar)"   # conservative
            elif hit_stop:
                exit_price, reason = stop, "stop"
            elif hit_target and cfg.exit_mode == "fixed_rr":
                exit_price, reason = target, "target"

            if exit_price is not None:
                # apply exit-side cost
                fill = exit_price * (1 - cps) if d == +1 else exit_price * (1 + cps)
                pnl_per_unit = (fill - open_trade["entry_fill"]) * d
                pnl_cash = pnl_per_unit * open_trade["qty"]
                equity += pnl_cash
                risk_cash = open_trade["risk_cash"]
                trades.append({
                    "entry_time": open_trade["entry_time"],
                    "exit_time": times[i],
                    "direction": "long" if d == +1 else "short",
                    "entry": open_trade["entry_fill"],
                    "exit": fill,
                    "stop": open_trade["stop_initial"],
                    "target": open_trade["target"],
                    "pnl_pct": pnl_cash / (equity - pnl_cash) * 100,
                    "pnl_R": pnl_cash / risk_cash if risk_cash else 0.0,
                    "bars_held": i - open_trade["entry_bar"],
                    "exit_reason": reason,
                })
                open_trade = None

        # ---- consider a new entry at this bar's close (no look-ahead) ----
        if open_trade is None and i in sig_by_bar:
            s = sig_by_bar[i]
            d = s.direction
            # entry fill pays cost on entry side
            entry_fill = s.entry_price * (1 + cps) if d == +1 else s.entry_price * (1 - cps)
            risk_per_unit = abs(entry_fill - s.stop_price)
            if risk_per_unit > 0:
                risk_cash = equity * cfg.risk_per_trade
                qty = risk_cash / risk_per_unit
                trail_dist = None
                if cfg.exit_mode == "trailing":
                    trail_dist = cfg.trail_atr_mult / cfg.__dict__.get("_atr_ref", 1) \
                        if False else (abs(entry_fill - s.stop_price))  # start = initial risk dist
                open_trade = {
                    "direction": d,
                    "entry_fill": entry_fill,
                    "entry_time": times[i],
                    "entry_bar": i,
                    "stop": s.stop_price,
                    "stop_initial": s.stop_price,
                    "target": s.target_price,
                    "qty": qty,
                    "risk_cash": risk_cash,
                    "trail_dist": trail_dist,
                }

        equity_curve[i] = equity

    equity_series = pd.Series(equity_curve, index=times).ffill()
    returns_series = equity_series.pct_change().fillna(0.0)
    trades_df = pd.DataFrame(trades)
    return equity_series, returns_series, trades_df


def buy_and_hold(df: pd.DataFrame, initial_equity: float = 10_000.0):
    """Benchmark: buy at first close, hold to last. Returns (equity, returns)."""
    close = df["close"]
    units = initial_equity / close.iloc[0]
    equity = close * units
    return equity, equity.pct_change().fillna(0.0)
