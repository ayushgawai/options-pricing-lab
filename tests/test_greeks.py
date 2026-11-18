"""(b) Analytic Greeks vs central finite differences."""
import pytest

from options_lab.pricer import bs_greeks, bs_price

CASES = [(100, 100, 1.0, 0.20), (100, 90, 0.5, 0.30), (100, 110, 2.0, 0.15),
         (120, 100, 0.25, 0.40)]


@pytest.mark.parametrize("S,K,T,vol", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_finite_difference(S, K, T, vol, kind):
    r, kw = 0.05, dict(q=0.0, option_type=kind)
    g = bs_greeks(S, K, T, r, vol, **kw)
    hS, hV, hT, hR = 1e-4 * S, 1e-5, 1e-6, 1e-6
    delta = (bs_price(S + hS, K, T, r, vol, **kw) - bs_price(S - hS, K, T, r, vol, **kw)) / (2 * hS)
    gamma = (bs_price(S + hS, K, T, r, vol, **kw) - 2 * bs_price(S, K, T, r, vol, **kw)
             + bs_price(S - hS, K, T, r, vol, **kw)) / hS ** 2
    vega = (bs_price(S, K, T, r, vol + hV, **kw) - bs_price(S, K, T, r, vol - hV, **kw)) / (2 * hV)
    # Theta is dV/dT with a sign flip (decay as expiry approaches).
    theta = -(bs_price(S, K, T + hT, r, vol, **kw) - bs_price(S, K, T - hT, r, vol, **kw)) / (2 * hT)
    rho = (bs_price(S, K, T, r + hR, vol, **kw) - bs_price(S, K, T, r - hR, vol, **kw)) / (2 * hR)
    assert g.delta == pytest.approx(delta, abs=1e-5)
    assert g.gamma == pytest.approx(gamma, abs=1e-5)
    assert g.vega == pytest.approx(vega, abs=1e-4)
    assert g.theta == pytest.approx(theta, abs=1e-4)
    assert g.rho == pytest.approx(rho, abs=1e-4)
