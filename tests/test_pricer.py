"""(a) Black-Scholes vs hand-checked values, moneyness sweep, edge cases."""
import numpy as np
import pytest

from options_lab.pricer import bs_greeks, bs_price, call_price, put_price

# Hand-checked (standard textbook values): S=K=100, T=1, r=5%, sigma=20%
ATM_CALL = 10.450584
ATM_PUT = 5.573526


def test_atm_hand_checked():
    assert call_price(100, 100, 1.0, 0.05, 0.20) == pytest.approx(ATM_CALL, abs=1e-5)
    assert put_price(100, 100, 1.0, 0.05, 0.20) == pytest.approx(ATM_PUT, abs=1e-5)


def test_put_call_parity():
    for K in (80, 100, 120):
        c = call_price(100, K, 1.0, 0.05, 0.25)
        p = put_price(100, K, 1.0, 0.05, 0.25)
        assert c - p == pytest.approx(100 - K * np.exp(-0.05), abs=1e-10)


def test_moneyness_sweep_monotonic_and_bounded():
    strikes = np.array([50., 80., 100., 120., 150., 200.])
    calls = bs_price(100, strikes, 1.0, 0.05, 0.20, option_type="call")
    puts = bs_price(100, strikes, 1.0, 0.05, 0.20, option_type="put")
    assert np.all(np.diff(calls) < 0)          # calls fall as strike rises
    assert np.all(np.diff(puts) > 0)
    assert np.all(calls >= 0) and np.all(calls <= 100)
    assert np.all(np.isfinite(calls)) and np.all(np.isfinite(puts))


def test_deep_itm_otm_no_nan():
    for S, K in ((100, 1), (100, 1000), (1, 100), (1000, 100)):
        for kind in ("call", "put"):
            v = bs_price(S, K, 1.0, 0.05, 0.20, option_type=kind)
            g = bs_greeks(S, K, 1.0, 0.05, 0.20, option_type=kind)
            assert np.isfinite(v) and v >= 0
            assert all(np.isfinite(x) for x in g.as_dict().values())


def test_expiry_and_zero_vol():
    assert bs_price(110, 100, 0.0, 0.05, 0.20, option_type="call") == 10.0
    assert bs_price(90, 100, 0.0, 0.05, 0.20, option_type="call") == 0.0
    assert bs_price(110, 100, 0.0, 0.05, 0.20, option_type="put") == 0.0
    g = bs_greeks(110, 100, 0.0, 0.05, 0.20, option_type="call")
    assert g.delta == 1.0 and g.gamma == 0.0 and g.vega == 0.0
    # T -> 0 limit stays finite
    for T in (1e-12, 1e-9, 1e-6):
        assert np.isfinite(bs_price(100, 100, T, 0.05, 0.20))
