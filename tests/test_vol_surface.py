"""(d) Surface arbitrage checks: clean SVI passes, broken skew is caught."""
import numpy as np

from options_lab.vol_surface import VolSurface, build_surface


def test_svi_surface_is_arbitrage_free():
    report = build_surface().check_arbitrage()
    assert report.ok, report


def test_broken_skew_is_caught():
    maturities = np.array([0.25, 0.5, 1.0])
    strikes = 100 * np.exp(np.linspace(-0.6, 0.6, 25))
    # Deliberately broken: a huge vol spike at the middle strike breaks
    # butterfly convexity (and the ATM skew bound nearby).
    vols = np.full((3, 25), 0.20)
    vols[:, 12] = 1.50
    surf = VolSurface.from_grid(100.0, maturities, strikes, vols)
    report = surf.check_arbitrage()
    assert not report.ok
    assert any("butterfly" in v or "skew" in v for v in report.violations)


def test_calendar_violation_is_caught():
    maturities = np.array([0.25, 1.0])
    strikes = np.array([90., 100., 110.])
    vols = np.array([[0.60, 0.60, 0.60],   # total var 0.09 at T=0.25
                     [0.20, 0.20, 0.20]])  # total var 0.04 at T=1 -> decreasing
    surf = VolSurface.from_grid(100.0, maturities, strikes, vols)
    report = surf.check_arbitrage()
    assert any("calendar" in v for v in report.violations)
