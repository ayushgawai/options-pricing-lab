"""(f) Backtest determinism under a fixed seed + CLI smoke tests."""
import numpy as np
import pytest

from options_lab.backtest import run_backtest, sweep_parameters
from options_lab.cli import main


def test_backtest_deterministic_same_seed():
    a = run_backtest(n_paths=10, n_ticks=200, seed=42)
    b = run_backtest(n_paths=10, n_ticks=200, seed=42)
    assert a.mean == b.mean and a.p5 == b.p5 and a.p95 == b.p95
    assert np.array_equal(a.pnls, b.pnls)


def test_backtest_differs_with_other_seed():
    a = run_backtest(n_paths=10, n_ticks=200, seed=42)
    b = run_backtest(n_paths=10, n_ticks=200, seed=7)
    assert not np.array_equal(a.pnls, b.pnls)


def test_backtest_summary_shape():
    r = run_backtest(n_paths=10, n_ticks=200, seed=1)
    assert r.p5 <= r.p95
    assert 0.0 <= r.hit_rate <= 1.0
    assert r.max_drawdown >= 0.0
    rows = sweep_parameters(n_paths=5, n_ticks=100, seed=1)
    assert len(rows) == 9


@pytest.mark.parametrize("argv", [["price"], ["iv"], ["surface"], ["mm", "--ticks", "50"],
                                  ["bench", "--maturities", "2", "--strikes", "3", "--vols", "2"],
                                  ["report", "--ticks", "50", "--paths", "3"]])
def test_cli_smoke(argv, capsys):
    main(argv)
    assert capsys.readouterr().out.strip()
