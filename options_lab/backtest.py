"""Monte Carlo backtester: sweep market-making parameters over vol paths.

Each path is a synthetic underlying walk whose per-tick volatility
follows a mean-reverting random walk (a *simulated volatility path* --
not a stochastic-vol pricing model). The inventory-aware maker from
:mod:`options_lab.market_making` trades each path. Reports the P&L
distribution: mean, p5, p95, hit rate (fraction of paths with positive
final P&L) and worst max-drawdown of the mark-to-market P&L curve.
Fully deterministic for a fixed seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market_making import MMConfig, simulate


def _max_drawdown(curve: np.ndarray) -> float:
    if len(curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(curve)
    return float(np.max(peak - curve))


def _vol_path(rng, n_ticks, base_vol=0.15, vol_of_vol=0.02):
    vol = np.empty(n_ticks)
    v = base_vol
    for t in range(n_ticks):
        v += 0.05 * (base_vol - v) + rng.normal(0.0, vol_of_vol)
        vol[t] = float(np.clip(v, 0.02, 0.80))
    return vol


@dataclass
class BacktestResult:
    pnls: np.ndarray
    mean: float
    p5: float
    p95: float
    hit_rate: float
    max_drawdown: float
    params: dict

    def summary(self) -> dict:
        return {"mean": self.mean, "p5": self.p5, "p95": self.p95,
                "hit_rate": self.hit_rate, "max_drawdown": self.max_drawdown,
                **{f"param_{k}": v for k, v in self.params.items()}}


def run_backtest(n_paths=50, n_ticks=500, seed=42, cfg: MMConfig | None = None,
                 start_mid=100.0) -> BacktestResult:
    cfg = cfg or MMConfig()
    rng = np.random.default_rng(seed)
    pnls, worst_dd = [], 0.0
    for _ in range(n_paths):
        vol = _vol_path(rng, n_ticks)
        shocks = rng.normal(0.0, 1.0, size=n_ticks) * vol
        mid = start_mid * np.exp(np.cumsum(shocks) - 0.5 * vol ** 2)
        path_seed = int(rng.integers(0, 2 ** 31 - 1))
        res = simulate(mid_prices=mid, seed=path_seed, cfg=cfg)
        pnls.append(res.final_pnl)
        worst_dd = max(worst_dd, _max_drawdown(res.pnl))
    arr = np.asarray(pnls)
    return BacktestResult(
        pnls=arr, mean=float(np.mean(arr)),
        p5=float(np.percentile(arr, 5)), p95=float(np.percentile(arr, 95)),
        hit_rate=float(np.mean(arr > 0)), max_drawdown=float(worst_dd),
        params={"inventory_cap": cfg.inventory_cap,
                "base_half_spread": cfg.base_half_spread,
                "risk_aversion": cfg.risk_aversion,
                "arrival_rate": cfg.arrival_rate})


def sweep_parameters(base_cfg: MMConfig | None = None, n_paths=30, n_ticks=400,
                     seed=42) -> list[dict]:
    """Sweep a few (half-spread, risk-aversion) combos; same seed each row."""
    base = base_cfg or MMConfig()
    rows = []
    for half_spread in (0.03, 0.05, 0.10):
        for gamma in (0.05, 0.10, 0.30):
            cfg = MMConfig(inventory_cap=base.inventory_cap,
                           quote_size=base.quote_size,
                           base_half_spread=half_spread, risk_aversion=gamma,
                           sigma=base.sigma,
                           inventory_widening=base.inventory_widening,
                           arrival_rate=base.arrival_rate, tau=base.tau)
            rows.append(run_backtest(n_paths=n_paths, n_ticks=n_ticks,
                                     seed=seed, cfg=cfg).summary())
    return rows
