"""(e) Market making: inventory cap is a hard bound, quotes widen/shift."""
import numpy as np

from options_lab.market_making import MMConfig, quotes, simulate


def test_inventory_cap_never_breached():
    for seed in range(5):
        cfg = MMConfig(inventory_cap=5, arrival_rate=0.9)
        res = simulate(n_ticks=2000, seed=seed, cfg=cfg)
        assert res.max_abs_inventory <= 5
        assert np.all(np.abs(res.inventory) <= 5)


def test_quotes_widen_and_shift_with_inventory():
    cfg = MMConfig(inventory_cap=10)
    bid0, ask0 = quotes(100.0, 0, cfg)
    bid_long, ask_long = quotes(100.0, 8, cfg)
    assert (ask_long - bid_long) > (ask0 - bid0)          # wider when loaded
    assert (bid_long + ask_long) / 2 < (bid0 + ask0) / 2  # reserve price shifts down
    assert bid_long < bid0                               # bid drops (shift + widening)
    # At the cap the buying side is pulled.
    bid_cap, _ = quotes(100.0, 10, cfg)
    assert np.isnan(bid_cap)


def test_pnl_accounting_finite():
    res = simulate(n_ticks=500, seed=1, cfg=MMConfig(inventory_cap=10))
    assert np.all(np.isfinite(res.pnl))
    assert res.n_trades > 0
    assert res.spread_capture >= 0
