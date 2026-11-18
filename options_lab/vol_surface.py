"""Arbitrage-checked volatility surface via an SVI-style smile.

Each maturity slice uses raw SVI total variance in log-moneyness
``k = ln(K/F)``::

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + s^2) ),   vol = sqrt(w / T)

with ``a, b`` scaled linearly in ``T`` so total variance grows with
maturity (no calendar arbitrage by construction). Static-arbitrage
checks report violations of: positive vol/price, calendar monotonicity
(total variance non-decreasing in ``T`` for fixed moneyness), butterfly
arbitrage (call prices convex in strike), and a skew bound
(``|d vol / d k|`` at ATM below a generous threshold).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pricer import bs_price

SKEW_BOUND = 2.5  # max |d vol / d log-moneyness| tolerated at ATM


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float  # SVI curvature scale (not the implied vol)

    def total_variance(self, k):
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m)
                                  + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))


def _svi_for_maturity(T: float, atm_vol: float = 0.20, rho: float = -0.40) -> SVIParams:
    # Choose a, b so w(0) = atm_vol^2 * T and the smile is gentle.
    b = 0.10 * T
    s = 0.10
    a = atm_vol ** 2 * T - b * s  # w(0) with m=0: a + b*s
    return SVIParams(a=max(a, 1e-6), b=b, rho=rho, m=0.0, sigma=s)


@dataclass
class ArbitrageReport:
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def __str__(self) -> str:
        return "OK: no static arbitrage violations" if self.ok else \
            "Violations:\n" + "\n".join(f"  - {v}" for v in self.violations)


class VolSurface:
    """Implied-vol grid over (maturities x strikes) with arb checks."""

    def __init__(self, spot, maturities, strikes, vols, r=0.05, q=0.0):
        self.spot = float(spot)
        self.maturities = np.asarray(maturities, dtype=float)
        self.strikes = np.asarray(strikes, dtype=float)
        self.vols = np.asarray(vols, dtype=float)
        if self.vols.shape != (len(self.maturities), len(self.strikes)):
            raise ValueError("vols must have shape (n_maturities, n_strikes)")
        self.r, self.q = float(r), float(q)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_svi(cls, spot=100.0, maturities=(0.25, 0.5, 1.0, 2.0),
                 strikes=None, r=0.05, q=0.0, atm_vol=0.20, rho=-0.40):
        maturities = np.asarray(maturities, dtype=float)
        if strikes is None:
            strikes = spot * np.exp(np.linspace(-0.6, 0.6, 25))
        strikes = np.asarray(strikes, dtype=float)
        vols = np.empty((len(maturities), len(strikes)))
        for i, T in enumerate(maturities):
            F = spot * np.exp((r - q) * T)
            k = np.log(strikes / F)
            w = _svi_for_maturity(T, atm_vol, rho).total_variance(k)
            vols[i] = np.sqrt(np.maximum(w, 1e-12) / T)
        return cls(spot, maturities, strikes, vols, r=r, q=q)

    @classmethod
    def from_grid(cls, spot, maturities, strikes, vols, r=0.05, q=0.0):
        return cls(spot, maturities, strikes, vols, r=r, q=q)

    # -- queries ----------------------------------------------------------
    def implied_vol(self, T, K):
        """Bilinear interpolation in (T, log K); flat extrapolation."""
        T_c = np.clip(T, self.maturities[0], self.maturities[-1])
        logK = np.log(np.clip(K, self.strikes[0], self.strikes[-1]))
        log_strikes = np.log(self.strikes)
        i = np.searchsorted(self.maturities, T_c)
        i = int(np.clip(i, 1, len(self.maturities) - 1))
        j = int(np.clip(np.searchsorted(log_strikes, logK), 1, len(self.strikes) - 1))
        t0, t1 = self.maturities[i - 1], self.maturities[i]
        x0, x1 = log_strikes[j - 1], log_strikes[j]
        wt = 0.0 if t1 == t0 else (T_c - t0) / (t1 - t0)
        wk = 0.0 if x1 == x0 else (logK - x0) / (x1 - x0)
        v = ((1 - wt) * (1 - wk) * self.vols[i - 1, j - 1]
             + (1 - wt) * wk * self.vols[i - 1, j]
             + wt * (1 - wk) * self.vols[i, j - 1]
             + wt * wk * self.vols[i, j])
        return float(v)

    def price_grid(self, option_type="call") -> np.ndarray:
        prices = np.empty_like(self.vols)
        for i, T in enumerate(self.maturities):
            prices[i] = bs_price(self.spot, self.strikes, T, self.r,
                                 self.vols[i], q=self.q, option_type=option_type)
        return prices

    # -- arbitrage checks ---------------------------------------------------
    def check_arbitrage(self, tol=1e-9) -> ArbitrageReport:
        v = ArbitrageReport()
        if np.any(~np.isfinite(self.vols)) or np.any(self.vols <= 0):
            v.violations.append("non-positive or non-finite implied volatility")
        prices = self.price_grid("call")
        if np.any(~np.isfinite(prices)) or np.any(prices < -tol):
            v.violations.append("non-positive / non-finite call price")

        # Calendar: total variance w = vol^2 * T non-decreasing in T per strike.
        w = self.vols ** 2 * self.maturities[:, None]
        if np.any(np.diff(w, axis=0) < -1e-8):
            bad = int(np.sum(np.diff(w, axis=0) < -1e-8))
            v.violations.append(f"calendar arbitrage: total variance decreases "
                                f"with maturity at {bad} grid point(s)")

        # Butterfly: call price convex in K (second difference >= 0).
        # Strikes are log-spaced in from_svi; use general divided differences.
        for i, T in enumerate(self.maturities):
            K, C = self.strikes, prices[i]
            for j in range(1, len(K) - 1):
                h1, h2 = K[j] - K[j - 1], K[j + 1] - K[j]
                slope1 = (C[j] - C[j - 1]) / h1
                slope2 = (C[j + 1] - C[j]) / h2
                if slope2 - slope1 < -1e-7:
                    v.violations.append(
                        f"butterfly arbitrage at T={T:g}, K={K[j]:.2f}: "
                        f"call prices not convex in strike")
                    break

        # Skew bound at (nearest) ATM per maturity.
        for i, T in enumerate(self.maturities):
            F = self.spot * np.exp((self.r - self.q) * T)
            k = np.log(self.strikes / F)
            order = np.argsort(k)
            dk = np.gradient(self.vols[i][order], k[order])
            atm = dk[np.argmin(np.abs(k[order]))]
            if abs(atm) > SKEW_BOUND:
                v.violations.append(
                    f"skew bound violated at T={T:g}: |d vol/dk|={abs(atm):.2f} "
                    f"> {SKEW_BOUND}")
        return v


def build_surface(spot=100.0, maturities=(0.25, 0.5, 1.0, 2.0), **kw) -> VolSurface:
    return VolSurface.from_svi(spot=spot, maturities=maturities, **kw)
