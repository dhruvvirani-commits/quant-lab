"""
tests/test_vol_surface.py
-------------------------
End-to-end test of the smile pipeline: build a synthetic chain with a KNOWN
imposed smile, invert every price to IV, and confirm we recover the smile.

Run from quant-lab/:  python -m pytest tests/test_vol_surface.py -v
"""
import numpy as np
from p2_volatility.deribit import make_synthetic_chain
from p2_volatility.vol_surface import compute_chain_iv


def test_chain_generates():
    chain = make_synthetic_chain()
    assert len(chain) > 20
    assert {"strike", "T", "mark_price_usd", "type"}.issubset(chain.columns)


def test_iv_recovery_is_exact():
    """Our solver must recover the exact IV we imposed on the synthetic chain."""
    chain = make_synthetic_chain()
    iv = compute_chain_iv(chain)
    err = np.abs(iv["iv"] - iv["true_iv"])
    assert err.max() < 1e-4


def test_smile_shape_recovered():
    """ATM implied vol must be lower than wing implied vol (a real smile)."""
    iv = compute_chain_iv(make_synthetic_chain())
    atm = iv[np.abs(iv["log_moneyness"]) < 0.05]["iv"].mean()
    wings = iv[np.abs(iv["log_moneyness"]) > 0.25]["iv"].mean()
    assert wings > atm


def test_multiple_expiries_present():
    """The chain should span several expiries (needed for a surface)."""
    iv = compute_chain_iv(make_synthetic_chain())
    assert iv["T"].nunique() >= 3
