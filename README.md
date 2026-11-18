# options-pricing-lab

A small, self-contained options research toolkit that mirrors the core work of an
options market-making desk: price European options, recover implied volatility,
build an arbitrage-checked volatility surface, quote around fair value with
inventory awareness, and backtest the quoting strategy by Monte Carlo.

Python 3.10+, NumPy and SciPy only for the math core. No network, no live data —
everything is synthetic and reproducible on a laptop under a fixed seed.

## Design

```
                 +------------------+
   inputs -----> |     pricer       |  Black-Scholes prices + analytic Greeks
                 +--------+---------+
                          |
                          v
                 +------------------+
                 |   implied_vol    |  Newton + bisection fallback, smile-guarded
                 +--------+---------+
                          |
                          v
                 +------------------+
                 |   vol_surface    |  SVI-style smile, static-arbitrage checks
                 +--------+---------+
                          |
                          v
                 +------------------+
                 |  market_making   |  Avellaneda-Stoikov reserve price, capped inventory
                 +--------+---------+
                          |
                          v
                 +------------------+
                 |    backtest      |  Monte Carlo sweep -> P&L distribution
                 +------------------+

        cli.py wires all five together: price | iv | surface | mm | bench | report
```

| Module | Responsibility | Key API |
|---|---|---|
| `options_lab/pricer.py` | Black-Scholes European call/put prices and full analytic Greeks; safe at expiry and deep ITM/OTM | `bs_price(S, K, T, r, sigma, q=0, option_type)`, `bs_greeks(...) -> Greeks`, `call_price`, `put_price` |
| `options_lab/implied_vol.py` | Recover implied vol from a price; never negative/NaN | `implied_vol(price, S, K, T, r, ...)`, `implied_vol_with_error(...)`, `implied_vols(...)` |
| `options_lab/vol_surface.py` | SVI-style smile over maturities × strikes; static-arbitrage checks and violation reports | `VolSurface.from_svi(...)`, `build_surface(...)`, `VolSurface.implied_vol(T, K)`, `check_arbitrage() -> ArbitrageReport` |
| `options_lab/market_making.py` | Inventory-aware bid/ask quoting, per-tick mark-to-market P&L, hard inventory cap | `MMConfig`, `quotes(mid, inventory, cfg)`, `simulate(...) -> MMResult` |
| `options_lab/backtest.py` | Monte Carlo parameter sweep over simulated vol paths; P&L distribution stats | `run_backtest(...) -> BacktestResult`, `sweep_parameters(...)` |
| `options_lab/cli.py` | argparse front end and the vectorised benchmark | subcommands `price`, `iv`, `surface`, `mm`, `bench`, `report` |

## Install

```bash
pip install -e .          # or just run from the repo root, no install needed
pip install pytest        # for the test suite
```

## Run

```bash
python -m options_lab.cli price   --spot 100 --strike 100 --maturity 1 --vol 0.20 --kind call
python -m options_lab.cli iv      --spot 100 --strike 110 --maturity 1 --true-vol 0.22
python -m options_lab.cli surface --spot 100
python -m options_lab.cli mm      --ticks 1000 --seed 0 --cap 10
python -m options_lab.cli bench   # 200 x 100 x 50 grid by default
python -m options_lab.cli report  # everything end to end, plus the backtest sweep table
```

## Test

```bash
python -m pytest -q
```

41 tests cover: (a) Black-Scholes against hand-checked ATM values and a moneyness
sweep, (b) analytic Greeks vs finite differences, (c) implied-vol recovery on a
synthetic smile to ~0.1% relative error, (d) arbitrage checks catching a deliberately
broken skew (and a calendar violation), (e) P&L/inventory staying inside the bounded
cap, and (f) backtest determinism under a fixed seed, plus CLI smoke tests.

## Benchmark

`bench` times a fully vectorised Black-Scholes grid (maturities × strikes × vols)
with `time.perf_counter`, best of 3 runs, and prints throughput plus a latency
table. The GFLOPs figure is explicitly an *estimate* at ~60 flops per price
(exp/log/normal-CDF cost folded into that constant) — treat it as an order-of-magnitude
sanity number, not a hardware benchmark.

Sanity run (actual output of `python -m options_lab.cli bench` on the development
laptop, modest 200×100×50 = 1,000,000-price scale):

```
Benchmark: vectorised Black-Scholes grid
  grid           200x100x50
  prices         1,000,000
  best_time_s    0.1479
  prices_per_s   6,762,130
  est_gflops     0.406 (at ~60 flops/price, estimated)
  ns_per_price   147.9

Latency table (single vectorised call, best of 3):
        size     prices    time_ms   ns/price
       1x1x1          1      0.060    60000.0
     1x100x1        100      0.058      584.0
   10x100x10     10,000      0.358       35.8
   50x100x20    100,000      9.437       94.4
```

Timings vary by machine; small grids are dominated by call overhead, which is why
ns/price only settles at larger sizes. Scale the grid *down* (`--maturities/--strikes/--vols`)
if your laptop is slower — the default already runs in well under a second.

## Limitations

- **Synthetic data only.** No market data, no network, no calibration to real quotes.
  Every price, vol path, and trade arrival is generated in-process.
- **European options only.** Black-Scholes closed forms; no early exercise, so no
  American/Bermudan options, no dividends-as-cash-amounts (only a continuous yield `q`).
- **No jump or stochastic-vol dynamics.** The pricer is constant-vol Black-Scholes;
  the SVI surface is a static smile parameterisation, not a dynamic vol model
  (no Heston/SABR/jump-diffusion). The backtest's "vol path" is a simple
  mean-reverting random walk used to stress the quoter, not to price.
- **Single asset.** One underlying, one option chain; no cross-asset hedging,
  correlation, or multi-leg structures.
- **No latency modelling for the market-making tick.** Quotes are computed and
  (possibly) traded within the same tick; there is no queue position, no partial
  fills beyond one quote size per tick, no fees/rebates, and at most one side
  trades per tick.
- The inventory cap is a hard quoting bound, not a risk limit with liquidation
  logic — at the cap the maker simply stops quoting the side that would add risk.
- GFLOPs in `bench` is an estimated figure (fixed ~60 flops/price), not a measured
  hardware counter.

## License

MIT — see `LICENSE`. Copyright (c) 2026 Ayush Gawai.
