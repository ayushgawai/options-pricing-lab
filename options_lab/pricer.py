"""Black-Scholes pricing and analytic Greeks for European options.

Closed forms (with continuous dividend yield ``q``)::

    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)
    Call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    Put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

Edge cases: ``T <= 0`` returns discounted-free intrinsic value with step
delta and zero gamma/vega/theta/rho; ``sigma <= 0`` returns discounted
intrinsic value. Inputs are clipped so deep ITM/OTM never produce NaN.
All functions accept scalars or NumPy arrays (broadcast together).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr  # standard normal CDF

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def _norm_pdf(x):
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / _SQRT_2PI


def _as_float_or_array(x):
    arr = np.asarray(x, dtype=float)
    return float(arr) if arr.ndim == 0 else arr


def _validate(S, K, T, sigma):
    if np.any(np.asarray(S, dtype=float) <= 0) or np.any(np.asarray(K, dtype=float) <= 0):
        raise ValueError("S and K must be positive")
    if np.any(np.asarray(T, dtype=float) < 0):
        raise ValueError("T must be non-negative")
    if np.any(np.asarray(sigma, dtype=float) < 0):
        raise ValueError("sigma must be non-negative")


def _d1_d2(S, K, T, r, sigma, q):
    S, K, T, r, sigma, q = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (S, K, T, r, sigma, q)])
    vol_sqrt_t = sigma * np.sqrt(np.maximum(T, 1e-300))
    # Clip log-moneyness / denominator so extreme inputs stay finite.
    log_m = np.clip(np.log(S / K), -700.0, 700.0)
    d1 = (log_m + (r - q + 0.5 * sigma ** 2) * T) / np.maximum(vol_sqrt_t, 1e-300)
    d1 = np.clip(d1, -1e4, 1e4)
    return d1, d1 - vol_sqrt_t


@dataclass
class Greeks:
    """Analytic Black-Scholes Greeks (vega/rho per 1.0 vol/rate unit)."""
    delta: float | np.ndarray
    gamma: float | np.ndarray
    vega: float | np.ndarray
    theta: float | np.ndarray  # per year
    rho: float | np.ndarray

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("delta", "gamma", "vega", "theta", "rho")}


def bs_price(S, K, T, r, sigma, q=0.0, option_type: str = "call"):
    """European call/put price. Scalars in -> float out; arrays broadcast."""
    _validate(S, K, T, sigma)
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    scalar = all(np.ndim(v) == 0 for v in (S, K, T, r, sigma, q))
    Sa, Ka, Ta, ra, sa, qa = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (S, K, T, r, sigma, q)])
    df_r, df_q = np.exp(-ra * Ta), np.exp(-qa * Ta)
    intrinsic = np.maximum(Sa - Ka, 0.0) if option_type == "call" else np.maximum(Ka - Sa, 0.0)
    disc_intrinsic = (np.maximum(Sa * df_q - Ka * df_r, 0.0) if option_type == "call"
                      else np.maximum(Ka * df_r - Sa * df_q, 0.0))

    price = np.empty_like(Sa, dtype=float)
    expired = Ta <= 0
    no_vol = (~expired) & (sa <= 0)
    regular = ~(expired | no_vol)
    price[expired] = intrinsic[expired]
    price[no_vol] = disc_intrinsic[no_vol]
    if np.any(regular):
        d1, d2 = _d1_d2(Sa[regular], Ka[regular], Ta[regular],
                          ra[regular], sa[regular], qa[regular])
        if option_type == "call":
            price[regular] = Sa[regular] * df_q[regular] * ndtr(d1) - Ka[regular] * df_r[regular] * ndtr(d2)
        else:
            price[regular] = Ka[regular] * df_r[regular] * ndtr(-d2) - Sa[regular] * df_q[regular] * ndtr(-d1)
    price = np.maximum(np.nan_to_num(price, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    # No-arbitrage upper bound: call <= S e^{-qT}, put <= K e^{-rT}
    upper = Sa * df_q if option_type == "call" else Ka * df_r
    price = np.minimum(price, upper)
    return float(price) if scalar else price


def call_price(S, K, T, r, sigma, q=0.0):
    return bs_price(S, K, T, r, sigma, q=q, option_type="call")


def put_price(S, K, T, r, sigma, q=0.0):
    return bs_price(S, K, T, r, sigma, q=q, option_type="put")


def bs_greeks(S, K, T, r, sigma, q=0.0, option_type: str = "call") -> Greeks:
    """Full analytic Greeks. At expiry: step delta, all other Greeks 0."""
    _validate(S, K, T, sigma)
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    scalar = all(np.ndim(v) == 0 for v in (S, K, T, r, sigma, q))
    Sa, Ka, Ta, ra, sa, qa = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (S, K, T, r, sigma, q)])
    df_r, df_q = np.exp(-ra * Ta), np.exp(-qa * Ta)
    delta = np.zeros_like(Sa); gamma = np.zeros_like(Sa)
    vega = np.zeros_like(Sa); theta = np.zeros_like(Sa); rho = np.zeros_like(Sa)

    expired = Ta <= 0
    if np.any(expired):
        if option_type == "call":
            delta[expired] = np.where(Sa[expired] > Ka[expired], 1.0,
                                      np.where(Sa[expired] == Ka[expired], 0.5, 0.0))
        else:
            delta[expired] = np.where(Sa[expired] < Ka[expired], -1.0,
                                      np.where(Sa[expired] == Ka[expired], -0.5, 0.0))
    regular = (~expired) & (sa > 0)
    if np.any(regular):
        S_, K_, T_, r_, s_, q_ = (v[regular] for v in (Sa, Ka, Ta, ra, sa, qa))
        dfr, dfq = df_r[regular], df_q[regular]
        d1, d2 = _d1_d2(S_, K_, T_, r_, s_, q_)
        pdf = _norm_pdf(d1)
        sqrt_t = np.sqrt(T_)
        common_gamma = dfq * pdf / np.maximum(S_ * s_ * sqrt_t, 1e-300)
        gamma[regular] = np.nan_to_num(common_gamma)
        vega[regular] = S_ * dfq * pdf * sqrt_t
        if option_type == "call":
            delta[regular] = dfq * ndtr(d1)
            theta[regular] = (-S_ * dfq * pdf * s_ / (2 * sqrt_t)
                              - r_ * K_ * dfr * ndtr(d2) + q_ * S_ * dfq * ndtr(d1))
            rho[regular] = K_ * T_ * dfr * ndtr(d2)
        else:
            delta[regular] = -dfq * ndtr(-d1)
            theta[regular] = (-S_ * dfq * pdf * s_ / (2 * sqrt_t)
                              + r_ * K_ * dfr * ndtr(-d2) - q_ * S_ * dfq * ndtr(-d1))
            rho[regular] = -K_ * T_ * dfr * ndtr(-d2)
    # Degenerate sigma == 0, T > 0: delta is step on discounted forward.
    degen = (~expired) & (sa <= 0)
    if np.any(degen):
        fwd_itm = Sa[degen] * df_q[degen] > Ka[degen] * df_r[degen]
        step = np.where(fwd_itm, 1.0, 0.0)
        delta[degen] = step if option_type == "call" else step - 1.0
    out = [np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
           for v in (delta, gamma, vega, theta, rho)]
    if scalar:
        out = [float(v) for v in out]
    return Greeks(*out)
