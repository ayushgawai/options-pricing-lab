"""(c) Implied-vol recovery on a synthetic smile to ~0.1% relative error."""
import numpy as np
import pytest

from options_lab.implied_vol import implied_vol, implied_vol_with_error
from options_lab.pricer import bs_price


def smile_vol(K, S=100.0):
    k = np.log(K / S)
    return 0.20 - 0.05 * k + 0.10 * k ** 2


@pytest.mark.parametrize("K", [70., 85., 100., 115., 130., 150.])
@pytest.mark.parametrize("T", [0.25, 1.0])
def test_iv_recovery_on_smile(K, T):
    true_vol = float(smile_vol(K))
    price = bs_price(100, K, T, 0.05, true_vol)
    out = implied_vol_with_error(price, true_vol, 100, K, T, 0.05)
    assert out["rel_error"] < 1e-3  # 0.1%
    assert np.isfinite(out["implied_vol"]) and out["implied_vol"] > 0


def test_iv_never_negative_or_nan():
    # Prices at/outside no-arbitrage bounds, deep wings, tiny T.
    for price in (0.0, 1e-9, 0.01, 50.0, 99.0, 100.0):
        iv = implied_vol(price, 100, 100, 0.5, 0.05)
        assert np.isfinite(iv) and 0 < iv <= 5.0
    for K in (20., 300.):
        price = bs_price(100, K, 0.05, 0.05, 0.35)
        iv = implied_vol(price, 100, K, 0.05, 0.05)
        assert np.isfinite(iv) and iv > 0
