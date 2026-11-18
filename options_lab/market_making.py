"""Inventory-aware market making (Avellaneda-Stoikov style).

Reserve (indifference) price shifts the mid against inventory::

    reserve = mid - inventory * risk_aversion * sigma^2 * tau
    half_spread = base_half_spread + risk_aversion * sigma^2 * tau / 2
                  + inventory_widening * |inventory|

``tau`` is a fixed quoting horizon (one tick here -- no latency is
modelled). A side that would push ``|inventory|`` past
``inventory_cap`` is not quoted, so the cap is a hard bound. Per-tick
P&L is mark-to-market: ``cash + inventory * mid``. Realised spread
capture accumulates ``(trade_price - mid)`` signed in the maker's favour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MMConfig:
    inventory_cap: int = 10
    quote_size: int = 1
    base_half_spread: float = 0.05
    risk_aversion: float = 0.10   # gamma in Avellaneda-Stoikov
    sigma: float = 0.20           # per-tick vol scale used for quoting
    inventory_widening: float = 0.01
    arrival_rate: float = 0.45    # prob. of a trade on a quoted side per tick
    tau: float = 1.0


@dataclass
class MMResult:
    mid: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    inventory: np.ndarray
    cash: np.ndarray
    pnl: np.ndarray
    spread_capture: float
    n_trades: int
    final_pnl: float
    max_abs_inventory: int

    def summary(self) -> dict:
        return {"final_pnl": float(self.final_pnl),
                "spread_capture": float(self.spread_capture),
                "n_trades": int(self.n_trades),
                "max_abs_inventory": int(self.max_abs_inventory),
                "final_inventory": int(self.inventory[-1]) if len(self.inventory) else 0}


def quotes(mid, inventory, cfg: MMConfig):
    """(bid, ask) for one tick; a capped side returns NaN (not quoted)."""
    shift = inventory * cfg.risk_aversion * cfg.sigma ** 2 * cfg.tau
    reserve = mid - shift
    half = (cfg.base_half_spread
            + 0.5 * cfg.risk_aversion * cfg.sigma ** 2 * cfg.tau
            + cfg.inventory_widening * abs(inventory))
    bid = reserve - half if inventory + cfg.quote_size <= cfg.inventory_cap else np.nan
    ask = reserve + half if inventory - cfg.quote_size >= -cfg.inventory_cap else np.nan
    return float(bid), float(ask)


def simulate(mid_prices=None, n_ticks=1000, seed=0, cfg: MMConfig | None = None,
             start_mid=100.0, tick_vol=0.15) -> MMResult:
    """Run the maker over a mid path (synthetic GBM-ish walk if not given)."""
    cfg = cfg or MMConfig()
    rng = np.random.default_rng(seed)
    if mid_prices is None:
        steps = rng.normal(0.0, tick_vol, size=n_ticks)
        mid = start_mid * np.exp(np.cumsum(steps) - 0.5 * tick_vol ** 2)
        mid = np.concatenate([[start_mid], mid[:-1]]) if n_ticks else np.array([start_mid])
    else:
        mid = np.asarray(mid_prices, dtype=float)
        n_ticks = len(mid)

    bid = np.full(n_ticks, np.nan); ask = np.full(n_ticks, np.nan)
    inventory = np.zeros(n_ticks + 1, dtype=int)
    cash = np.zeros(n_ticks + 1)
    pnl = np.zeros(n_ticks)
    spread_capture, n_trades = 0.0, 0

    for t in range(n_ticks):
        inv = inventory[t]
        b, a = quotes(mid[t], inv, cfg)
        bid[t], ask[t] = b, a
        cash[t + 1] = cash[t]
        inventory[t + 1] = inv
        # At most one side trades per tick in this simple model.
        u = rng.random()
        if np.isfinite(b) and u < cfg.arrival_rate / 2:        # client sells -> maker buys
            inventory[t + 1] = inv + cfg.quote_size
            cash[t + 1] = cash[t] - b * cfg.quote_size
            spread_capture += (mid[t] - b) * cfg.quote_size
            n_trades += 1
        elif np.isfinite(a) and u < cfg.arrival_rate:          # client buys -> maker sells
            inventory[t + 1] = inv - cfg.quote_size
            cash[t + 1] = cash[t] + a * cfg.quote_size
            spread_capture += (a - mid[t]) * cfg.quote_size
            n_trades += 1
        assert abs(inventory[t + 1]) <= cfg.inventory_cap, "inventory cap breached"
        pnl[t] = cash[t + 1] + inventory[t + 1] * mid[t]

    return MMResult(mid=mid, bid=bid, ask=ask, inventory=inventory[:-1],
                    cash=cash[:-1], pnl=pnl, spread_capture=float(spread_capture),
                    n_trades=n_trades,
                    final_pnl=float(pnl[-1]) if n_ticks else 0.0,
                    max_abs_inventory=int(np.max(np.abs(inventory))) if n_ticks else 0)
