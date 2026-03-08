import math

import numpy as np
import pytest

from weightiz_module5_stats import (
    deflated_sharpe_ratio,
    effective_num_trials_from_corr,
    expected_max_z,
    model_confidence_set,
    norm_ppf,
    pbo_cscv,
    psr_against_threshold,
    spa_test,
    stationary_bootstrap_indices,
    run_full_stats,
    validate_returns_1d,
    validate_returns_2d,
    white_reality_check,
)


def test_determinism_bootstrap_wrc_spa_mcs():
    T, N = 320, 6
    rng = np.random.default_rng(123)
    bmk = rng.normal(0.0001, 0.01, size=T).astype(np.float64)
    ret = rng.normal(0.0002, 0.011, size=(T, N)).astype(np.float64)
    losses = (-ret + rng.normal(0.0, 0.001, size=(T, N))).astype(np.float64)

    idx_a = stationary_bootstrap_indices(T=T, B=128, avg_block_len=17, seed=77)
    idx_b = stationary_bootstrap_indices(T=T, B=128, avg_block_len=17, seed=77)
    idx_c = stationary_bootstrap_indices(T=T, B=128, avg_block_len=17, seed=78)
    assert np.array_equal(idx_a, idx_b)
    assert not np.array_equal(idx_a, idx_c)

    wrc_a = white_reality_check(ret, bmk, B=128, avg_block_len=15, seed=91)
    wrc_b = white_reality_check(ret, bmk, B=128, avg_block_len=15, seed=91)
    assert wrc_a["p_value"] == wrc_b["p_value"]
    assert np.array_equal(wrc_a["t_boot"], wrc_b["t_boot"])

    spa_a = spa_test(ret, bmk, B=128, avg_block_len=15, seed=92)
    spa_b = spa_test(ret, bmk, B=128, avg_block_len=15, seed=92)
    assert spa_a["p_value"] == spa_b["p_value"]
    assert np.array_equal(spa_a["t_boot"], spa_b["t_boot"])

    mcs_a = model_confidence_set(losses, alpha=0.10, B=128, avg_block_len=15, seed=93)
    mcs_b = model_confidence_set(losses, alpha=0.10, B=128, avg_block_len=15, seed=93)
    assert np.array_equal(mcs_a["survivors"], mcs_b["survivors"])
    assert mcs_a["elimination_order"] == mcs_b["elimination_order"]


def test_guards_validation_bootstrap_and_pbo():
    x_ok_1d = np.arange(5, dtype=np.float64)
    x_ok_2d = np.arange(12, dtype=np.float64).reshape(6, 2)

    with pytest.raises(RuntimeError):
        validate_returns_1d(np.nan * x_ok_1d)
    with pytest.raises(RuntimeError):
        x = x_ok_1d.copy()
        x[0] = np.inf
        validate_returns_1d(x)
    with pytest.raises(RuntimeError):
        validate_returns_1d(x_ok_2d)
    with pytest.raises(RuntimeError):
        validate_returns_1d(np.zeros(10_001, dtype=np.float64))

    with pytest.raises(RuntimeError):
        validate_returns_2d(np.nan * x_ok_2d)
    with pytest.raises(RuntimeError):
        x = x_ok_2d.copy()
        x[0, 0] = np.inf
        validate_returns_2d(x)
    with pytest.raises(RuntimeError):
        validate_returns_2d(x_ok_1d)
    with pytest.raises(RuntimeError):
        validate_returns_2d(np.zeros((10_001, 2), dtype=np.float64))

    with pytest.raises(RuntimeError):
        stationary_bootstrap_indices(T=9_000, B=3_000, avg_block_len=10, seed=1)

    tiny = np.zeros((20, 2), dtype=np.float64)
    with pytest.raises(RuntimeError):
        pbo_cscv(tiny, S=20, k=10)
    with pytest.raises(RuntimeError):
        pbo_cscv(tiny, S=300, k=2)



def test_expected_max_and_norm_ppf_policy():
    vals = np.array([expected_max_z(n) for n in range(2, 10_001)], dtype=np.float64)
    assert np.all(np.isfinite(vals))

    with pytest.raises(RuntimeError):
        norm_ppf(np.array([0.0], dtype=np.float64))
    with pytest.raises(RuntimeError):
        norm_ppf(np.array([1.0], dtype=np.float64))
    with pytest.raises(RuntimeError):
        norm_ppf(np.array([-1e-9], dtype=np.float64))


def test_dsr_effective_trials_participation_ratio():
    T, N = 900, 10
    rng = np.random.default_rng(202)

    base = rng.normal(0.0, 0.01, size=T).astype(np.float64)
    r_ident = np.tile(base[:, None], (1, N)).astype(np.float64)
    n_eff_ident = effective_num_trials_from_corr(r_ident)
    assert n_eff_ident <= 2

    r_uncorr = rng.normal(0.0, 0.01, size=(T, N)).astype(np.float64)
    n_eff_uncorr = effective_num_trials_from_corr(r_uncorr)
    assert n_eff_uncorr >= (N // 2)


def test_psr_denom_pathology_and_dsr_telemetry():
    sr = np.array([1.0, 0.2, -1.0], dtype=np.float64)
    sr_star = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    skew = np.array([10.0, 0.1, -10.0], dtype=np.float64)
    kurt_excess = np.array([-10.0, 0.0, -10.0], dtype=np.float64)
    n_obs = np.array([100.0, 100.0, 100.0], dtype=np.float64)

    psr = psr_against_threshold(
        sr_daily=sr,
        sr_star_daily=sr_star,
        skew=skew,
        kurt_excess=kurt_excess,
        n_obs=n_obs,
    )
    assert psr[0] == 0.0
    assert psr[2] == 0.0
    assert np.all(np.isfinite(psr))
    assert np.all((psr >= 0.0) & (psr <= 1.0))

    rng = np.random.default_rng(303)
    r = rng.normal(0.0001, 0.01, size=(600, 7)).astype(np.float64)
    dsr = deflated_sharpe_ratio(r)
    assert "psr_denom_sq_min" in dsr
    assert "psr_denom_bad_count" in dsr
    assert np.isfinite(float(dsr["psr_denom_sq_min"])) or math.isnan(float(dsr["psr_denom_sq_min"]))
    assert int(dsr["psr_denom_bad_count"]) >= 0
    dsr_vals = np.asarray(dsr["dsr"], dtype=np.float64)
    assert np.all(np.isfinite(dsr_vals))
    assert np.all((dsr_vals >= 0.0) & (dsr_vals <= 1.0))


def test_pbo_edge_cases_single_and_identical_candidates():
    rng = np.random.default_rng(404)

    one = rng.normal(0.0, 0.01, size=(300, 1)).astype(np.float64)
    out_one = pbo_cscv(one, S=10, k=5)
    assert out_one["insufficient_candidates"] is True
    assert math.isnan(float(out_one["pbo"]))

    base = rng.normal(0.0, 0.01, size=360).astype(np.float64)
    ident = np.tile(base[:, None], (1, 5)).astype(np.float64)
    out_a = pbo_cscv(ident, S=10, k=5)
    out_b = pbo_cscv(ident, S=10, k=5)
    assert np.isfinite(float(out_a["pbo"]))
    assert out_a["pbo"] == out_b["pbo"]
    assert np.array_equal(out_a["oos_rank_of_is_best"], out_b["oos_rank_of_is_best"])


def test_pbo_accepts_n_trials_effective_and_reports_used_value():
    rng = np.random.default_rng(606)
    r = rng.normal(0.0, 0.01, size=(300, 8)).astype(np.float64)

    out_default = pbo_cscv(r, S=10, k=5)
    out_neff = pbo_cscv(r, S=10, k=5, n_trials_effective=3)

    assert int(out_default["n_trials_effective_used"]) == 8
    assert int(out_neff["n_trials_effective_used"]) == 3
    assert np.isfinite(float(out_neff["pbo"]))


@pytest.mark.parametrize("n_trials_effective, expected", [(None, 8), (1, 1), (8, 8)])
def test_pbo_effective_trial_semantics(n_trials_effective, expected):
    rng = np.random.default_rng(1606)
    r = rng.normal(0.0, 0.01, size=(300, 8)).astype(np.float64)

    kwargs = {} if n_trials_effective is None else {"n_trials_effective": n_trials_effective}
    out = pbo_cscv(r, S=10, k=5, **kwargs)

    assert int(out["n_trials_effective_used"]) == expected
    assert np.isfinite(float(out["pbo"]))


@pytest.mark.parametrize("bad_value", [0, -1, 9, True, 2.5, np.array([2]), np.array([1, 2]), "3"])
def test_pbo_rejects_invalid_effective_trial_count(bad_value):
    rng = np.random.default_rng(1707)
    r = rng.normal(0.0, 0.01, size=(300, 8)).astype(np.float64)

    with pytest.raises(RuntimeError, match="n_trials_effective"):
        pbo_cscv(r, S=10, k=5, n_trials_effective=bad_value)


def test_run_full_stats_accepts_n_trials_effective():
    rng = np.random.default_rng(707)
    r = rng.normal(0.0002, 0.01, size=(360, 6)).astype(np.float64)
    bmk = rng.normal(0.0001, 0.009, size=360).astype(np.float64)
    out = run_full_stats(r, bmk, n_trials_effective=2)

    assert int(out["dsr"]["n_trials_effective"]) == 2
    assert int(out["pbo"]["n_trials_effective_used"]) == 2


def test_run_full_stats_rejects_invalid_effective_trial_count():
    rng = np.random.default_rng(808)
    r = rng.normal(0.0002, 0.01, size=(360, 6)).astype(np.float64)
    bmk = rng.normal(0.0001, 0.009, size=360).astype(np.float64)

    with pytest.raises(RuntimeError, match="n_trials_effective"):
        run_full_stats(r, bmk, n_trials_effective=7)


def test_run_full_stats_determinism_preserved_with_effective_trial_override():
    rng = np.random.default_rng(909)
    r = rng.normal(0.0002, 0.01, size=(360, 6)).astype(np.float64)
    bmk = rng.normal(0.0001, 0.009, size=360).astype(np.float64)

    out_a = run_full_stats(r, bmk, n_trials_effective=2)
    out_b = run_full_stats(r, bmk, n_trials_effective=2)

    assert int(out_a["dsr"]["n_trials_effective"]) == 2
    assert int(out_a["pbo"]["n_trials_effective_used"]) == 2
    assert np.array_equal(np.asarray(out_a["dsr"]["dsr"]), np.asarray(out_b["dsr"]["dsr"]))
    assert np.array_equal(np.asarray(out_a["pbo"]["lambda_logits"]), np.asarray(out_b["pbo"]["lambda_logits"]))
    assert float(out_a["pbo"]["pbo"]) == float(out_b["pbo"]["pbo"])


def test_wrc_spa_sanity_baseline_and_alpha_case():
    rng = np.random.default_rng(505)
    T, N = 700, 6
    bmk = rng.normal(0.0, 0.01, size=T).astype(np.float64)

    r_base = np.tile(bmk[:, None], (1, N)).astype(np.float64)
    wrc_base = white_reality_check(r_base, bmk, B=128, avg_block_len=20, seed=61)
    spa_base = spa_test(r_base, bmk, B=128, avg_block_len=20, seed=62)

    assert np.isfinite(float(wrc_base["p_value"]))
    assert np.isfinite(float(spa_base["p_value"]))
    assert float(wrc_base["p_value"]) >= 0.5
    assert float(spa_base["p_value"]) >= 0.5

    r_alpha = r_base.copy()
    r_alpha[:, 0] = bmk + 0.003

    wrc_alpha = white_reality_check(r_alpha, bmk, B=128, avg_block_len=20, seed=61)
    spa_alpha = spa_test(r_alpha, bmk, B=128, avg_block_len=20, seed=62)

    assert np.isfinite(float(wrc_alpha["p_value"]))
    assert np.isfinite(float(spa_alpha["p_value"]))
    assert float(wrc_alpha["p_value"]) < float(wrc_base["p_value"])
    assert float(spa_alpha["p_value"]) < float(spa_base["p_value"])
