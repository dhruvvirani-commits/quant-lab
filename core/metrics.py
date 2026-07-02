"""
core/metrics.py
---------------
Performance statistics computed from an equity curve and/or a list of trades.
These are the numbers a quant actually reads. Every formula is standard and
documented inline so you can defend each one in an interview.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# 15-minute bars -> this many bars per year (crypto trades 24/7/365)
BARS_PER_YEAR_15M = 365 * 24 * 4  # = 35040


def _annualisation_factor(bars_per_year: int) -> float:
    return np.sqrt(bars_per_year)


def sharpe_ratio(returns: pd.Series, bars_per_year: int = BARS_PER_YEAR_15M) -> float:
    """
    Annualised Sharpe on per-bar returns. Risk-free assumed 0 (standard for
    intraday strategy research). Sharpe = mean/std * sqrt(periods_per_year).
    """
    r = returns.dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return (r.mean() / r.std()) * _annualisation_factor(bars_per_year)


def sortino_ratio(returns: pd.Series, bars_per_year: int = BARS_PER_YEAR_15M) -> float:
    """Like Sharpe but penalises only downside volatility."""
    r = returns.dropna()
    downside = r[r < 0]
    if downside.std() == 0 or len(r) < 2:
        return 0.0
    return (r.mean() / downside.std()) * _annualisation_factor(bars_per_year)


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline of the equity curve, as a negative fraction."""
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def calmar_ratio(equity: pd.Series, returns: pd.Series,
                 bars_per_year: int = BARS_PER_YEAR_15M) -> float:
    """Annualised return divided by |max drawdown|."""
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(returns) / bars_per_year
    if years == 0:
        return 0.0
    ann_ret = (1 + total_ret) ** (1 / years) - 1
    return ann_ret / abs(mdd)


def trade_stats(trades: pd.DataFrame) -> dict:
    """
    Summary stats from a trades table with a 'pnl_R' column (P&L in R-multiples)
    and a 'return_pct' column (P&L as % of equity).
    """
    if len(trades) == 0:
        return {"n_trades": 0, "win_rate": 0.0, "avg_win_R": 0.0,
                "avg_loss_R": 0.0, "profit_factor": 0.0, "expectancy_R": 0.0}

    wins = trades[trades["pnl_R"] > 0]
    losses = trades[trades["pnl_R"] <= 0]

    gross_win = wins["pnl_R"].sum()
    gross_loss = abs(losses["pnl_R"].sum())

    return {
        "n_trades": int(len(trades)),
        "win_rate": len(wins) / len(trades),
        "avg_win_R": float(wins["pnl_R"].mean()) if len(wins) else 0.0,
        "avg_loss_R": float(losses["pnl_R"].mean()) if len(losses) else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "expectancy_R": float(trades["pnl_R"].mean()),
    }


def full_report(equity: pd.Series, returns: pd.Series, trades: pd.DataFrame,
                bars_per_year: int = BARS_PER_YEAR_15M) -> dict:
    """Everything, in one dict. This is what the backtest prints."""
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(returns) / bars_per_year
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0

    rep = {
        "total_return_pct": total_ret * 100,
        "annualised_return_pct": ann_ret * 100,
        "sharpe": sharpe_ratio(returns, bars_per_year),
        "sortino": sortino_ratio(returns, bars_per_year),
        "max_drawdown_pct": max_drawdown(equity) * 100,
        "calmar": calmar_ratio(equity, returns, bars_per_year),
    }
    rep.update(trade_stats(trades))
    return rep


def print_report(title: str, rep: dict) -> None:
    print(f"\n{'='*54}\n  {title}\n{'='*54}")
    order = [
        ("total_return_pct", "Total return", "%.2f%%"),
        ("annualised_return_pct", "Annualised return", "%.2f%%"),
        ("sharpe", "Sharpe", "%.2f"),
        ("sortino", "Sortino", "%.2f"),
        ("max_drawdown_pct", "Max drawdown", "%.2f%%"),
        ("calmar", "Calmar", "%.2f"),
        ("n_trades", "Trades", "%d"),
        ("win_rate", "Win rate", "%.1f%%"),
        ("avg_win_R", "Avg win (R)", "%.2f"),
        ("avg_loss_R", "Avg loss (R)", "%.2f"),
        ("profit_factor", "Profit factor", "%.2f"),
        ("expectancy_R", "Expectancy (R)", "%.3f"),
    ]
    for key, label, fmt in order:
        val = rep.get(key, 0.0)
        if key == "win_rate":
            val = val * 100
        print(f"  {label:<22} {fmt % val}")
    print("=" * 54)
