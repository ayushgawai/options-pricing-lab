"""Implied-volatility recovery: Newton iteration with bisection fallback.

A volatility-smile guard supplies a moneyness-aware initial guess
(a Brenner-Subrahmanyam-style approximation clipped to a sane band and
blended toward ATM vol away from the money) so Newton starts close even
for wings. If Newton leaves the bracket, stalls, or vega vanishes, plain
bisection on ``[lo, hi]`` finishes the job. The result is always finite
and inside ``[1e-4, 5.0]`` -- never negative, never NaN: prices outside
no-arbitrage bounds are clipped to the nearest bound first.
"""
from __future__ import annotations

import numpy as np

from .pricer import bs_greeks, bs_price

VOL_LO, VOL_HI = 1e-4, 5.0


def _no_arb_bounds(S, K, T, r, q, option_type):
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    if option_type == "call":
        return max(S * df_q - K * df_r, 0.0), S * df_q
    return max(K * df_r - S * df_q, 0.0), K * df_r


def _initial_guess(price, S, K, T, r, q, option_type):
    """Smile-guarded guess: BS approximation, shrunk toward 0.20 in wings."""
    if T <= 0:
        return 0.20
    df_r = np.exp(-r * T)
    # Time value above discounted intrinsic drives the guess.
    lower, _ = _no_arb_bounds(S, K, T, r, q, option_type)
    time_value = max(price - lower, 1e-12)
    guess = np.sqrt(2 * np.pi / T) * time_value / max(S * np.exp(-q * T), 1e-12)
    log_m = np.log((S * np.exp((r - q) * T)) / K)  # log forward moneyness
    # Away from ATM the approximation degrades; blend toward a flat 0.20.
    weight = np.exp(-0.5 * (log_m / 0.25) ** 2)
    guess = weight * guess + (1 - weight) * 0.20
    return float(np.clip(guess if np.isfinite(guess) else 0.20, 0.05, 2.0))


def implied_vol(price, S, K, T, r, q=0.0, option_type: str = "call",
                tol: float = 1e-10, max_iter: int = 100) -> float:
    """Recover implied vol from a market price. Always in [1e-4, 5.0]."""
    if T <= 0 or not np.isfinite(price):
        return 0.20
    lower, upper = _no_arb_bounds(S, K, T, r, q, option_type)
    # Clip inside bounds (with a hair of room) so a root exists.
    eps = max(1e-12, 1e-10 * max(upper - lower, 1.0))
    target = float(np.clip(price, lower + eps, upper - eps)) if upper - lower > 2 * eps else lower + eps

    def diff(sigma):
        return bs_price(S, K, T, r, sigma, q=q, option_type=option_type) - target

    lo, hi = VOL_LO, VOL_HI
    if diff(lo) > 0:
        return VOL_LO
    if diff(hi) < 0:
        return VOL_HI

    sigma = _initial_guess(target, S, K, T, r, q, option_type)
    for _ in range(max_iter):
        d = diff(sigma)
        if abs(d) < tol:
            return float(np.clip(sigma, VOL_LO, VOL_HI))
        vega = float(bs_greeks(S, K, T, r, sigma, q=q, option_type=option_type).vega)
        # Maintain the bisection bracket from the sign of the residual.
        if d > 0:
            hi = min(hi, sigma)
        else:
            lo = max(lo, sigma)
        step = d / vega if vega > 1e-12 else np.nan
        candidate = sigma - step if np.isfinite(step) else np.nan
        if not np.isfinite(candidate) or not (lo < candidate < hi):
            candidate = 0.5 * (lo + hi)  # bisection fallback
        if abs(candidate - sigma) < 1e-12:
            sigma = candidate
            break
        sigma = candidate
    # Final polish with pure bisection (guaranteed convergence).
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if hi - lo < 1e-12 or abs(diff(mid)) < tol:
            return float(np.clip(mid, VOL_LO, VOL_HI))
        if diff(mid) > 0:
            hi = mid
        else:
            lo = mid
    return float(np.clip(0.5 * (lo + hi), VOL_LO, VOL_HI))


def implied_vol_with_error(price, true_vol, S, K, T, r, q=0.0,
                           option_type: str = "call") -> dict:
    """IV plus accuracy metrics against a planted true volatility."""
    iv = implied_vol(price, S, K, T, r, q=q, option_type=option_type)
    abs_err = abs(iv - true_vol)
    rel_err = abs_err / true_vol if true_vol else float(abs_err)
    return {"implied_vol": iv, "true_vol": float(true_vol),
            "abs_error": float(abs_err), "rel_error": float(rel_err)}


def implied_vols(prices, S, K, T, r, q=0.0, option_type: str = "call") -> np.ndarray:
    """Vectorised wrapper (loop over strikes; each solve is scalar)."""
    prices, Ks = np.broadcast_arrays(np.asarray(prices, float), np.asarray(K, float))
    out = [implied_vol(float(p), S, float(k), T, r, q=q, option_type=option_type)
           for p, k in zip(prices.ravel(), Ks.ravel())]
    return np.asarray(out).reshape(prices.shape)
