"""
p1_backtester/strategy_filtered.py
----------------------------------
A+ Sweep-Shift **v2** — the same core logic as v1, plus the quality filters the
trader actually applies in live discretionary trading. v1 took EVERY sweep+CHoCH
(698 trades, -95%). The whole point of v2 is to test whether the DISCRETION —
not the base pattern — is where the edge lives.

IMPORTANT (anti-overfitting): every filter below is a rule the trader already
uses live. We are NOT inventing rules to fit 2023 data. That distinction is what
separates legitimate refinement from curve-fitting.

FILTERS ADDED IN v2
-------------------
1. HTF bias      : 4H 50-EMA. Longs only when the 4H trend is up; shorts only down.
2. CHoCH speed   : the shift must confirm within 10 bars of the sweep.
3. Session       : entry bar must be in 12:00-16:00 UTC (London/NY overlap).
4. Displacement  : the CHoCH candle body must exceed 1.0x the recent average body.
5. Sweep quality : the sweep wick must extend >= 0.25x ATR beyond the swept level.

All filters use only information available at or before the current bar.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .strategy import atr, swing_pivots, Signal, _last_within, _last_after


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def htf_trend_bias(df_15m: pd.DataFrame, ema_len: int = 50) -> pd.Series:
    """
    Resample 15m -> 4H, compute the 50-EMA trend, then map each 4H bias back
    onto the 15m index. Bias = +1 (up) if 4H close > EMA, -1 (down) if below.

    No look-ahead: a 15m bar is assigned the bias of the LAST FULLY CLOSED 4H
    candle before it (we shift the 4H series forward by one bar before mapping).
    """
    h4 = df_15m["close"].resample("4h").last().dropna()
    h4_ema = ema(h4, ema_len)
    bias_4h = np.sign(h4 - h4_ema)          # +1 / -1 / 0
    bias_4h = bias_4h.shift(1)              # only use CLOSED 4H bars -> no look-ahead
    # forward-fill the 4H bias onto every 15m timestamp
    bias_15m = bias_4h.reindex(df_15m.index, method="ffill")
    return bias_15m.fillna(0)


class SweepShiftFiltered:
    """v2 strategy: core sweep+CHoCH gated by the trader's real quality filters."""

    def __init__(self, n_pivot: int = 3, atr_period: int = 14,
                 atr_mult: float = 1.5, rr: float = 2.5,
                 sweep_lookback: int = 60,
                 # --- v2 filters ---
                 htf_ema: int = 50,
                 choch_max_bars: int = 10,
                 session_utc: tuple[int, int] = (12, 16),
                 displacement_mult: float = 1.0,
                 sweep_atr_frac: float = 0.25,
                 body_lookback: int = 20):
        self.n = n_pivot
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.rr = rr
        self.sweep_lookback = sweep_lookback
        self.htf_ema = htf_ema
        self.choch_max_bars = choch_max_bars
        self.session_utc = session_utc
        self.displacement_mult = displacement_mult
        self.sweep_atr_frac = sweep_atr_frac
        self.body_lookback = body_lookback

    def generate(self, df: pd.DataFrame) -> list[Signal]:
        df = df.copy()
        df["atr"] = atr(df, self.atr_period)
        is_high, is_low = swing_pivots(df, self.n)
        bias = htf_trend_bias(df, self.htf_ema).values

        # rolling average candle body (for displacement), lagged -> no look-ahead
        body = (df["close"] - df["open"]).abs()
        avg_body = body.rolling(self.body_lookback, min_periods=self.body_lookback).mean().shift(1).values

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        opens = df["open"].values
        atrv = df["atr"].values
        hours = df.index.hour.values
        m = len(df)

        conf_highs: list[tuple[int, float]] = []
        conf_lows: list[tuple[int, float]] = []
        signals: list[Signal] = []
        pending = None

        s_lo, s_hi = self.session_utc

        for i in range(m):
            j = i - self.n
            if j >= 0:
                if is_high.iloc[j]:
                    conf_highs.append((j, highs[j]))
                if is_low.iloc[j]:
                    conf_lows.append((j, lows[j]))

            if np.isnan(atrv[i]) or np.isnan(avg_body[i]):
                continue

            recent_high = _last_within(conf_highs, i, self.sweep_lookback)
            recent_low = _last_within(conf_lows, i, self.sweep_lookback)

            if pending is None:
                # bearish sweep (wick above swing high, close back below)
                if recent_high is not None:
                    lvl = recent_high[1]
                    wick_dist = highs[i] - lvl
                    if highs[i] > lvl and closes[i] < lvl \
                            and wick_dist >= self.sweep_atr_frac * atrv[i]:
                        pending = {"direction": -1, "sweep_bar": i, "level": lvl}
                        continue
                # bullish sweep (wick below swing low, close back above)
                if recent_low is not None:
                    lvl = recent_low[1]
                    wick_dist = lvl - lows[i]
                    if lows[i] < lvl and closes[i] > lvl \
                            and wick_dist >= self.sweep_atr_frac * atrv[i]:
                        pending = {"direction": +1, "sweep_bar": i, "level": lvl}
                        continue

            else:
                d = pending["direction"]
                bars_since = i - pending["sweep_bar"]
                # FILTER: CHoCH speed
                if bars_since > self.choch_max_bars:
                    pending = None
                    continue

                candle_body = abs(closes[i] - opens[i])
                displaced = candle_body >= self.displacement_mult * avg_body[i]  # FILTER: displacement
                in_session = s_lo <= hours[i] < s_hi                             # FILTER: session

                if d == +1:
                    mh = _last_after(conf_highs, pending["sweep_bar"])
                    if mh is not None and closes[i] > mh[1]:
                        # FILTER: HTF bias must be up for a long
                        if bias[i] >= 0 and displaced and in_session:
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
                        # FILTER: HTF bias must be down for a short
                        if bias[i] <= 0 and displaced and in_session:
                            entry = closes[i]
                            stop = entry + self.atr_mult * atrv[i]
                            risk = stop - entry
                            target = entry - self.rr * risk
                            if risk > 0:
                                signals.append(Signal(-1, entry, stop, target, i))
                        pending = None

        return signals
