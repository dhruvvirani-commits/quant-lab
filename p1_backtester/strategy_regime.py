"""
p1_backtester/strategy_regime.py
--------------------------------
A+ Sweep-Shift **v3** — everything in v2, plus a REGIME FILTER.

WHY
---
The multi-asset / multi-year test (robustness.py) revealed the v2 edge is
regime-dependent: it works in trending markets and bleeds in choppy, ranging
ones (2022 was a disaster; trending 2024 was near breakeven). This is the
single most important finding P1 produced.

v3 acts on that finding directly: it uses ADX (Average Directional Index) to
measure trend strength and STANDS ASIDE when the market is choppy.

    ADX  < adx_min  -> ranging/choppy  -> skip the setup
    ADX >= adx_min  -> trending        -> allow the setup

ANTI-OVERFITTING NOTE
---------------------
ADX is a standard, decades-old trend-strength indicator, and adx_min=20 is its
conventional threshold — NOT a value hand-tuned to make 2023 BTC look good. We
apply the same threshold to every asset and year, then re-run the SAME
robustness test to see if it helps CONSISTENTLY. If it only helps on the data
we already saw, that's a warning sign, not a win.
"""

from __future__ import annotations
import numpy as np
from .strategy import adx, Signal
from .strategy_filtered import SweepShiftFiltered


class SweepShiftRegime(SweepShiftFiltered):
    """v2 + an ADX regime gate. Only trades when a real trend is present."""

    def __init__(self, *args, adx_period: int = 14, adx_min: float = 20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.adx_period = adx_period
        self.adx_min = adx_min

    def generate(self, df):
        # compute the v2 signals first (all the existing filters)
        base_signals = super().generate(df)
        if not base_signals:
            return base_signals

        # regime series, aligned to the same index
        adx_series = adx(df, self.adx_period).values

        # keep only signals fired when ADX confirms a trend at the entry bar
        kept = []
        for s in base_signals:
            a = adx_series[s.bar_index]
            if not np.isnan(a) and a >= self.adx_min:
                kept.append(s)
        return kept
