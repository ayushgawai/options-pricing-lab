"""Command-line interface: price | iv | surface | mm | bench | report."""
from __future__ import annotations

import argparse
import time

import numpy as np

from .backtest import run_backtest, sweep_parameters
from .implied_vol import implied_vol, implied_vol_with_error
from .market_making import MMConfig, simulate
from .pricer import bs_greeks, bs_price
from .vol_surface import build_surface

# Rough FLOP count for one vectorised Black-Scholes price (exp/log/normal
# CDF approximated as a fixed cost). Labelled as an estimate wherever used.
_FLOPS_PER_PRICE = 60


def _cmd_price(a):
    p = bs_price(a.spot, a.strike, a.maturity, a.rate, a.vol, q=a.dividend, option_type=a.kind)
    g = bs_greeks(a.spot, a.strike, a.maturity, a.rate, a.vol, q=a.dividend, option_type=a.kind)
    print(f"{a.kind} price: {p:.6f}")
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        print(f"  {name:5s}: {getattr(g, name): .6f}")


def _cmd_iv(a):
    planted_price = bs_price(a.spot, a.strike, a.maturity, a.rate, a.true_vol,
                             q=a.dividend, option_type=a.kind)
    price = a.price if a.price is not None else planted_price
    out = implied_vol_with_error(price, a.true_vol, a.spot, a.strike, a.maturity,
                                 a.rate, q=a.dividend, option_type=a.kind)
    print(f"market price : {price:.6f} (planted vol {a.true_vol})")
    print(f"implied vol  : {out['implied_vol']:.6f}")
    print(f"abs error    : {out['abs_error']:.2e}   rel error: {out['rel_error']:.2%}")


def _cmd_surface(a):
    surf = build_surface(spot=a.spot, r=a.rate)
    report = surf.check_arbitrage()
    print(f"surface: {len(surf.maturities)} maturities x {len(surf.strikes)} strikes, spot={a.spot}")
    print(f"ATM vol (T={surf.maturities[len(surf.maturities)//2]:g}): "
          f"{surf.implied_vol(surf.maturities[len(surf.maturities)//2], a.spot):.4f}")
    print(report)


def _cmd_mm(a):
    cfg = MMConfig(inventory_cap=a.cap)
    s = simulate(n_ticks=a.ticks, seed=a.seed, cfg=cfg).summary()
    for k, v in s.items():
        print(f"{k:18s}: {v}")


def bench(n_maturities=200, n_strikes=100, n_vols=50, repeats=3) -> dict:
    """Vectorised grid timing. Default 200x100x50 = 1,000,000 prices."""
    maturities = np.linspace(0.05, 2.0, n_maturities)[:, None, None]
    strikes = np.linspace(50.0, 150.0, n_strikes)[None, :, None]
    vols = np.linspace(0.05, 0.80, n_vols)[None, None, :]
    n = int(n_maturities * n_strikes * n_vols)
    best, last = np.inf, None
    for _ in range(max(repeats, 1)):
        t0 = time.perf_counter()
        last = bs_price(100.0, strikes, maturities, 0.05, vols, option_type="call")
        best = min(best, time.perf_counter() - t0)
    assert np.all(np.isfinite(last))
    throughput = n / best
    gflops = throughput * _FLOPS_PER_PRICE / 1e9
    rows = [("grid", f"{n_maturities}x{n_strikes}x{n_vols}"), ("prices", f"{n:,}"),
            ("best_time_s", f"{best:.4f}"), ("prices_per_s", f"{throughput:,.0f}"),
            ("est_gflops", f"{gflops:.3f} (at ~{_FLOPS_PER_PRICE} flops/price, estimated)"),
            ("ns_per_price", f"{best / n * 1e9:.1f}")]
    print("Benchmark: vectorised Black-Scholes grid")
    for k, v in rows:
        print(f"  {k:14s} {v}")
    print("\nLatency table (single vectorised call, best of 3):")
    print(f"  {'size':>10s} {'prices':>10s} {'time_ms':>10s} {'ns/price':>10s}")
    for m, k_, v_ in ((1, 1, 1), (1, 100, 1), (10, 100, 10), (50, 100, 20)):
        T = np.linspace(0.05, 2.0, m)[:, None, None]
        K = np.linspace(50.0, 150.0, k_)[None, :, None]
        sg = np.linspace(0.05, 0.80, v_)[None, None, :]
        cnt = m * k_ * v_
        t_best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            bs_price(100.0, K, T, 0.05, sg, option_type="call")
            t_best = min(t_best, time.perf_counter() - t0)
        print(f"  {f'{m}x{k_}x{v_}':>10s} {cnt:>10,} {t_best*1e3:>10.3f} {t_best/cnt*1e9:>10.1f}")
    return {"n_prices": n, "best_time_s": best, "prices_per_s": throughput,
            "est_gflops": gflops}


def _cmd_bench(a):
    bench(a.maturities, a.strikes, a.vols)


def _cmd_report(a):
    print("=== options-pricing-lab report ===\n")
    print("[price]")
    _cmd_price(argparse.Namespace(spot=100.0, strike=100.0, maturity=1.0, rate=0.05,
                                  vol=0.20, dividend=0.0, kind="call"))
    print("\n[iv]")
    _cmd_iv(argparse.Namespace(spot=100.0, strike=100.0, maturity=1.0, rate=0.05,
                               true_vol=0.22, dividend=0.0, kind="call", price=None))
    print("\n[surface]")
    _cmd_surface(argparse.Namespace(spot=100.0, rate=0.05))
    print("\n[mm]")
    _cmd_mm(argparse.Namespace(cap=10, ticks=a.ticks, seed=a.seed))
    print("\n[backtest sweep]")
    print(f"  {'half_spread':>11s} {'gamma':>6s} {'mean':>9s} {'p5':>9s} {'p95':>9s} {'hit':>6s} {'maxDD':>9s}")
    for row in sweep_parameters(n_paths=a.paths, n_ticks=min(a.ticks, 400), seed=a.seed):
        print(f"  {row['param_base_half_spread']:>11.3f} {row['param_risk_aversion']:>6.2f} "
              f"{row['mean']:>9.3f} {row['p5']:>9.3f} {row['p95']:>9.3f} "
              f"{row['hit_rate']:>6.2f} {row['max_drawdown']:>9.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="options-lab", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--spot", type=float, default=100.0)
        sp.add_argument("--strike", type=float, default=100.0)
        sp.add_argument("--maturity", type=float, default=1.0)
        sp.add_argument("--rate", type=float, default=0.05)
        sp.add_argument("--dividend", type=float, default=0.0)
        sp.add_argument("--kind", choices=("call", "put"), default="call")

    sp = sub.add_parser("price"); common(sp); sp.add_argument("--vol", type=float, default=0.20); sp.set_defaults(fn=_cmd_price)
    sp = sub.add_parser("iv"); common(sp)
    sp.add_argument("--true-vol", type=float, default=0.22); sp.add_argument("--price", type=float, default=None); sp.set_defaults(fn=_cmd_iv)
    sp = sub.add_parser("surface"); sp.add_argument("--spot", type=float, default=100.0); sp.add_argument("--rate", type=float, default=0.05); sp.set_defaults(fn=_cmd_surface)
    sp = sub.add_parser("mm"); sp.add_argument("--ticks", type=int, default=1000); sp.add_argument("--seed", type=int, default=0); sp.add_argument("--cap", type=int, default=10); sp.set_defaults(fn=_cmd_mm)
    sp = sub.add_parser("bench"); sp.add_argument("--maturities", type=int, default=200); sp.add_argument("--strikes", type=int, default=100); sp.add_argument("--vols", type=int, default=50); sp.set_defaults(fn=_cmd_bench)
    sp = sub.add_parser("report"); sp.add_argument("--ticks", type=int, default=1000); sp.add_argument("--paths", type=int, default=30); sp.add_argument("--seed", type=int, default=42); sp.set_defaults(fn=_cmd_report)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
