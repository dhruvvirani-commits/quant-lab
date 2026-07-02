"""
p1_backtester/strategy.py
-------------------------
The A+ Sweep-Shift strategy, translated from discretionary judgment into
EXACT, code-checkable rules. Every fuzzy word ("sweep", "shift") is pinned to
a precise definition below. This file is the real intellectual work of P1 —
the engine (engine.py) is generic; THIS is your edge, made testable.

STRATEGY DEFINITION (v1, as specified by the trader)
----------------------------------------------------
Swing pivot : fractal of size N=3. A swing HIGH at bar i means
              high[i] is strictly greater than the highs of the 3 bars on
              each side. Swing LOW symmetric. (Pivot is only *confirmed* 3
              bars later -> no look-ahead.)

Liquidity   : the most recent CONFIRMED swing high (for a bearish setup) or
sweep         swing low (for a bullish setup). A valid sweep = a candle whose
              wick pierces beyond that level BUT whose close returns back
              inside it.

CHoCH       : after a bullish sweep (swept a low), we require a Change of
(the shift)   Character up = price closes above the most recent minor swing
              high formed since the sweep. Bearish is symmetric.

Entry       : at the close of the candle that confirms the CHoCH.
Stop-loss   : 1.5 x ATR(14) from entry (below entry for longs).
Take-profit : fixed 2.5R (v1). Trailing stop is a v1.1 variant.

All signals use ONLY information available at or before the current bar.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range. Uses prior close -> no look-ahead."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index — measures TREND STRENGTH (not direction).
    Low ADX (< ~20) = choppy/ranging market; high ADX = strong trend.
    Standard Wilder calculation. Uses only prior/current bars -> no look-ahead.
    """
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)

    # Wilder smoothing via EMA with alpha = 1/period
    atr_ = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()


def swing_pivots(df: pd.DataFrame, n: int = 3):
    """
    Fractal swing highs/lows of half-width n.
    Returns two boolean Series (is_swing_high, is_swing_low).

    A pivot at index i is only knowable at bar i+n (needs n bars to its right).
    We therefore store the pivot at its true location i, and the engine is
    careful to only *use* a pivot once bar i+n has passed. See engine.py.
    """
    highs, lows = df["high"].values, df["low"].values
    m = len(df)
    is_high = np.zeros(m, dtype=bool)
    is_low = np.zeros(m, dtype=bool)
    for i in range(n, m - n):
        window_h = highs[i - n:i + n + 1]
        window_l = lows[i - n:i + n + 1]
        if highs[i] == window_h.max() and (window_h.argmax() == n):
            is_high[i] = True
        if lows[i] == window_l.min() and (window_l.argmin() == n):
            is_low[i] = True
    return (pd.Series(is_high, index=df.index),
            pd.Series(is_low, index=df.index))


# ---------------------------------------------------------------------------
# signal state machine
# ---------------------------------------------------------------------------
@dataclass
class Signal:
    direction: int          # +1 long, -1 short
    entry_price: float
    stop_price: float
    target_price: float
    bar_index: int


class SweepShiftStrategy:
    """
    Streams bars in and emits Signal objects. Keeps its own state so the engine
    stays generic. NO look-ahead: at bar i it only references pivots confirmed
    at or before bar i.
    """

    def __init__(self, n_pivot: int = 3, atr_period: int = 14,
                 atr_mult: float = 1.5, rr: float = 2.5,
                 sweep_lookback: int = 60):
        self.n = n_pivot
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.rr = rr
        self.sweep_lookback = sweep_lookback

    def generate(self, df: pd.DataFrame) -> list[Signal]:
        df = df.copy()
        df["atr"] = atr(df, self.atr_period)
        is_high, is_low = swing_pivots(df, self.n)

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        atrv = df["atr"].values
        m = len(df)

        # confirmed-pivot lists: (location_index, price). A pivot at loc j is
        # only appended once we've advanced to bar j+n (right side complete).
        conf_highs: list[tuple[int, float]] = []
        conf_lows: list[tuple[int, float]] = []

        signals: list[Signal] = []

        # sweep state: after a valid sweep we look for the CHoCH
        pending = None  # dict: direction, sweep_bar, swing_level, choch_level

        for i in range(m):
            # 1) confirm any pivot whose right side just completed at bar i
            j = i - self.n
            if j >= 0:
                if is_high.iloc[j]:
                    conf_highs.append((j, highs[j]))
                if is_low.iloc[j]:
                    conf_lows.append((j, lows[j]))

            if np.isnan(atrv[i]):
                continue

            # 2) look for a fresh sweep of the most recent confirmed pivot
            #    (only pivots confirmed and within lookback window)
            recent_high = _last_within(conf_highs, i, self.sweep_lookback)
            recent_low = _last_within(conf_lows, i, self.sweep_lookback)

            if pending is None:
                # bearish sweep: wick above a swing high, close back below it
                if recent_high is not None:
                    lvl = recent_high[1]
                    if highs[i] > lvl and closes[i] < lvl:
                        pending = {"direction": -1, "sweep_bar": i,
                                   "level": lvl, "choch_level": None}
                        continue
                # bullish sweep: wick below a swing low, close back above it
                if recent_low is not None:
                    lvl = recent_low[1]
                    if lows[i] < lvl and closes[i] > lvl:
                        pending = {"direction": +1, "sweep_bar": i,
                                   "level": lvl, "choch_level": None}
                        continue

            # 3) if we have a pending sweep, look for the CHoCH (the shift)
            else:
                d = pending["direction"]
                bars_since = i - pending["sweep_bar"]
                # abandon the setup if it takes too long
                if bars_since > self.sweep_lookback:
                    pending = None
                    continue

                if d == +1:
                    # need a minor swing high formed AFTER the sweep to break up
                    mh = _last_after(conf_highs, pending["sweep_bar"])
                    if mh is not None and closes[i] > mh[1]:
                        entry = closes[i]
                        stop = entry - self.atr_mult * atrv[i]
                        risk = entry - stop
                        target = entry + self.rr * risk
                        if risk > 0:
                            signals.append(Signal(+1, entry, stop, target, i))
                        pending = None
                else:
                    ml = _last_after(conf_lows, pending["sweep_bar"])
                    if ml is not None and closes[i] < ml[1]:
                        entry = closes[i]
                        stop = entry + self.atr_mult * atrv[i]
                        risk = stop - entry
                        target = entry - self.rr * risk
                        if risk > 0:
                            signals.append(Signal(-1, entry, stop, target, i))
                        pending = None

        return signals


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _last_within(pivots: list[tuple[int, float]], now: int, lookback: int):
    """Most recent pivot whose location is within [now-lookback, now]."""
    for loc, price in reversed(pivots):
        if loc <= now and (now - loc) <= lookback:
            return (loc, price)
        if (now - loc) > lookback:
            break
    return None


def _last_after(pivots: list[tuple[int, float]], after_bar: int):
    """Most recent pivot located strictly after `after_bar`."""
    for loc, price in reversed(pivots):
        if loc > after_bar:
            return (loc, price)
    return None
