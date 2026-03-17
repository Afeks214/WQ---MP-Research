from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from weightiz.module1.core import ProfileStatIdx, ScoreIdx
from weightiz.module4.strategy_funnel import Module4Config, Module4SignalOutput
from weightiz.module4.strategy_intent_engine import (
    FAMILY_F3,
    FAMILY_F5,
    FAMILY_F6,
    FAMILY_F5_OVERLAY,
    generate_strategy_intent,
)
from weightiz.module4.wave1_parity import evaluate_f6_parity


def _wave1_inputs(T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A = 1
    alpha = np.zeros((A, T, 7), dtype=np.float64)
    score = np.zeros((A, T, int(ScoreIdx.N_FIELDS)), dtype=np.float64)
    profile = np.zeros((A, T, int(ProfileStatIdx.N_FIELDS)), dtype=np.float64)
    regime = np.zeros((A, T), dtype=np.int8)
    confidence = np.ones((A, T), dtype=np.float64)
    tradable = np.ones((A, T), dtype=bool)
    alpha[:, :, 0] = 1.0  # rvol
    alpha[:, :, 1] = 100.0  # tod
    alpha[:, :, 2] = 0.0  # x_poc
    alpha[:, :, 3] = 0.0  # x_vah
    alpha[:, :, 4] = 0.0  # x_val
    alpha[:, :, 5] = 1.0  # va_width
    alpha[:, :, 6] = 1.0  # session_id
    return alpha, score, profile, regime, confidence, tradable


def _run_wave1(
    *,
    alpha: np.ndarray,
    score: np.ndarray,
    profile: np.ndarray,
    regime: np.ndarray,
    confidence: np.ndarray,
    tradable: np.ndarray,
    cfg: Module4Config,
):
    return generate_strategy_intent(
        alpha_signal_tensor=alpha,
        score_tensor=score,
        profile_stat_tensor=profile,
        regime_id=regime,
        regime_confidence=confidence,
        tradable_mask=tradable,
        cfg4=cfg,
    )


def test_f5_accepted_breakout_path() -> None:
    alpha, score, profile, regime, confidence, tradable = _wave1_inputs(T=1)
    cfg = Module4Config(strategy_type="institutional_wave1", family_id="F5")

    alpha[0, 0, 0] = 1.5
    alpha[0, 0, 3] = -0.25  # breakout up: VAH below close
    score[0, 0, int(ScoreIdx.SCORE_BO_LONG)] = 0.62
    score[0, 0, int(ScoreIdx.SCORE_BO_SHORT)] = 0.12
    score[0, 0, int(ScoreIdx.SCORE_REJECT)] = 0.18
    profile[0, 0, int(ProfileStatIdx.D)] = 1.6
    profile[0, 0, int(ProfileStatIdx.DCLIP)] = 1.6
    profile[0, 0, int(ProfileStatIdx.Z_DELTA)] = 1.3
    profile[0, 0, int(ProfileStatIdx.GBREAK)] = 0.75
    profile[0, 0, int(ProfileStatIdx.GREJECT)] = 0.2

    out = _run_wave1(alpha=alpha, score=score, profile=profile, regime=regime, confidence=confidence, tradable=tradable, cfg=cfg)
    assert bool(out.intent_long[0, 0]), "F5 should fire long on accepted breakout"
    assert not bool(out.intent_short[0, 0]), "F5 breakout should not emit short in this scenario"


def test_f3_responsive_rotation_path() -> None:
    alpha, score, profile, regime, confidence, tradable = _wave1_inputs(T=1)
    cfg = Module4Config(strategy_type="institutional_wave1", family_id="F3")

    alpha[0, 0, 0] = 1.1
    alpha[0, 0, 3] = 0.08  # near upper edge
    score[0, 0, int(ScoreIdx.SCORE_BO_LONG)] = 0.20
    score[0, 0, int(ScoreIdx.SCORE_BO_SHORT)] = 0.10
    score[0, 0, int(ScoreIdx.SCORE_REJECT)] = 0.72
    profile[0, 0, int(ProfileStatIdx.D)] = 1.1
    profile[0, 0, int(ProfileStatIdx.DCLIP)] = 1.1
    profile[0, 0, int(ProfileStatIdx.Z_DELTA)] = -1.3
    profile[0, 0, int(ProfileStatIdx.GBREAK)] = 0.25
    profile[0, 0, int(ProfileStatIdx.GREJECT)] = 0.72

    out = _run_wave1(alpha=alpha, score=score, profile=profile, regime=regime, confidence=confidence, tradable=tradable, cfg=cfg)
    assert bool(out.intent_short[0, 0]), "F3 should fire short at upper edge with rejection dominance"
    assert not bool(out.intent_long[0, 0]), "F3 short scenario should not emit long"


def test_f6_disabled_if_parity_required() -> None:
    alpha, score, profile, regime, confidence, tradable = _wave1_inputs(T=1)
    cfg = Module4Config(strategy_type="institutional_wave1", family_id="F6", f6_enabled=False, f6_parity_required=True)

    alpha[0, 0, 0] = 1.2
    alpha[0, 0, 3] = -0.25
    score[0, 0, int(ScoreIdx.SCORE_BO_LONG)] = 0.20
    score[0, 0, int(ScoreIdx.SCORE_REJECT)] = 0.60
    profile[0, 0, int(ProfileStatIdx.D)] = 1.6
    profile[0, 0, int(ProfileStatIdx.DCLIP)] = 1.6
    profile[0, 0, int(ProfileStatIdx.Z_DELTA)] = 0.2
    profile[0, 0, int(ProfileStatIdx.GBREAK)] = 0.30
    profile[0, 0, int(ProfileStatIdx.GREJECT)] = 0.60

    out = _run_wave1(alpha=alpha, score=score, profile=profile, regime=regime, confidence=confidence, tradable=tradable, cfg=cfg)
    assert bool(out.intent_flat[0, 0]), "F6 must fail closed when parity is required but f6_enabled=False"


def test_f6_enabled_when_gate_pass() -> None:
    alpha, score, profile, regime, confidence, tradable = _wave1_inputs(T=1)
    cfg = Module4Config(strategy_type="institutional_wave1", family_id="F6", f6_enabled=True, f6_parity_required=True)

    alpha[0, 0, 0] = 1.2
    alpha[0, 0, 3] = -0.20
    score[0, 0, int(ScoreIdx.SCORE_BO_LONG)] = 0.22
    score[0, 0, int(ScoreIdx.SCORE_BO_SHORT)] = 0.10
    score[0, 0, int(ScoreIdx.SCORE_REJECT)] = 0.62
    profile[0, 0, int(ProfileStatIdx.D)] = 1.6
    profile[0, 0, int(ProfileStatIdx.DCLIP)] = 1.6
    profile[0, 0, int(ProfileStatIdx.Z_DELTA)] = 0.1
    profile[0, 0, int(ProfileStatIdx.GBREAK)] = 0.30
    profile[0, 0, int(ProfileStatIdx.GREJECT)] = 0.62

    out = _run_wave1(alpha=alpha, score=score, profile=profile, regime=regime, confidence=confidence, tradable=tradable, cfg=cfg)
    assert bool(out.intent_short[0, 0]), "F6 should fade up-extension with weak breakout gate"


def test_f5_f6_dead_zone_no_trade() -> None:
    alpha, score, profile, regime, confidence, tradable = _wave1_inputs(T=1)
    cfg = Module4Config(strategy_type="institutional_wave1", family_id="AUTO")

    alpha[0, 0, 0] = 1.4
    alpha[0, 0, 3] = -0.2
    score[0, 0, int(ScoreIdx.SCORE_BO_LONG)] = 0.65
    score[0, 0, int(ScoreIdx.SCORE_BO_SHORT)] = 0.08
    score[0, 0, int(ScoreIdx.SCORE_REJECT)] = 0.30
    profile[0, 0, int(ProfileStatIdx.D)] = 1.4
    profile[0, 0, int(ProfileStatIdx.DCLIP)] = 1.4
    profile[0, 0, int(ProfileStatIdx.Z_DELTA)] = 1.1
    profile[0, 0, int(ProfileStatIdx.GBREAK)] = 0.48  # dead-zone
    profile[0, 0, int(ProfileStatIdx.GREJECT)] = 0.20

    out = _run_wave1(alpha=alpha, score=score, profile=profile, regime=regime, confidence=confidence, tradable=tradable, cfg=cfg)
    assert bool(out.intent_flat[0, 0]), "Breakout dead-zone must force no-trade"


def test_breakout_dead_zone_blocks_f3_fallback() -> None:
    alpha, score, profile, regime, confidence, tradable = _wave1_inputs(T=1)
    cfg = Module4Config(strategy_type="institutional_wave1", family_id="AUTO")

    # F3-short would otherwise fire: upper-edge + reject dominance + z_delta<0 + greject high.
    alpha[0, 0, 0] = 1.1
    alpha[0, 0, 3] = 0.08
    score[0, 0, int(ScoreIdx.SCORE_BO_LONG)] = 0.20
    score[0, 0, int(ScoreIdx.SCORE_BO_SHORT)] = 0.10
    score[0, 0, int(ScoreIdx.SCORE_REJECT)] = 0.75
    profile[0, 0, int(ProfileStatIdx.D)] = 1.3  # breakout context via |D|>=1.2
    profile[0, 0, int(ProfileStatIdx.DCLIP)] = 1.3
    profile[0, 0, int(ProfileStatIdx.Z_DELTA)] = -1.2
    profile[0, 0, int(ProfileStatIdx.GBREAK)] = 0.48  # dead-zone
    profile[0, 0, int(ProfileStatIdx.GREJECT)] = 0.70

    out = _run_wave1(alpha=alpha, score=score, profile=profile, regime=regime, confidence=confidence, tradable=tradable, cfg=cfg)
    assert bool(out.intent_flat[0, 0]), "Breakout dead-zone must block fallback F3 entries on the same bar"
    family_code = getattr(out, "family_code")
    assert family_code is not None
    assert int(family_code[0, 0]) == 0, "Dead-zone bars must emit no family selection"


def test_f3_f6_partition_by_excess_distance() -> None:
    alpha, score, profile, regime, confidence, tradable = _wave1_inputs(T=2)
    cfg = Module4Config(strategy_type="institutional_wave1", family_id="AUTO")

    alpha[0, :, 0] = [1.1, 1.1]
    alpha[0, :, 3] = [0.10, 0.10]  # upper edge for F3 short
    score[0, :, int(ScoreIdx.SCORE_BO_LONG)] = [0.30, 0.30]
    score[0, :, int(ScoreIdx.SCORE_BO_SHORT)] = [0.10, 0.10]
    score[0, :, int(ScoreIdx.SCORE_REJECT)] = [0.70, 0.70]
    profile[0, :, int(ProfileStatIdx.D)] = [1.5, 2.2]  # switch partition threshold
    profile[0, :, int(ProfileStatIdx.DCLIP)] = [2.0, 2.0]
    profile[0, :, int(ProfileStatIdx.Z_DELTA)] = [-1.2, -1.2]
    profile[0, :, int(ProfileStatIdx.GBREAK)] = [0.30, 0.30]
    profile[0, :, int(ProfileStatIdx.GREJECT)] = [0.62, 0.62]

    out = _run_wave1(alpha=alpha, score=score, profile=profile, regime=regime, confidence=confidence, tradable=tradable, cfg=cfg)
    family_code = getattr(out, "family_code")
    assert family_code is not None
    assert int(family_code[0, 0]) == int(FAMILY_F3), "Lower excess should resolve to F3 in F3/F6 collision"
    assert int(family_code[0, 1]) == int(FAMILY_F6), "Higher excess should resolve to F6 in F3/F6 collision"


def test_f5_overlay_extension_and_flatten() -> None:
    alpha, score, profile, regime, confidence, tradable = _wave1_inputs(T=4)
    cfg = Module4Config(strategy_type="institutional_wave1", family_id="F5", enable_f5_close_overlay=True)

    alpha[0, :, 0] = [1.5, 1.1, 1.1, 1.1]  # rvol
    alpha[0, :, 1] = [340.0, 350.0, 30.0, 40.0]  # tod
    alpha[0, :, 3] = [-0.20, 0.05, 0.05, 0.05]  # breakout only on t0
    alpha[0, :, 6] = [1.0, 1.0, 2.0, 2.0]  # session switch at t2

    score[0, :, int(ScoreIdx.SCORE_BO_LONG)] = [0.65, 0.10, 0.10, 0.10]
    score[0, :, int(ScoreIdx.SCORE_BO_SHORT)] = [0.05, 0.05, 0.05, 0.05]
    score[0, :, int(ScoreIdx.SCORE_REJECT)] = [0.10, 0.10, 0.10, 0.80]  # reject flip at t3

    profile[0, :, int(ProfileStatIdx.D)] = [1.6, 1.2, 1.0, 1.0]
    profile[0, :, int(ProfileStatIdx.DCLIP)] = [1.6, 1.2, 1.0, 1.0]
    profile[0, :, int(ProfileStatIdx.Z_DELTA)] = [1.9, 1.3, 1.2, 1.2]
    profile[0, :, int(ProfileStatIdx.GBREAK)] = [0.80, 0.70, 0.70, 0.70]
    profile[0, :, int(ProfileStatIdx.GREJECT)] = [0.20, 0.20, 0.20, 0.70]

    out = _run_wave1(alpha=alpha, score=score, profile=profile, regime=regime, confidence=confidence, tradable=tradable, cfg=cfg)
    assert bool(out.intent_long[0, 0]), "F5 should enter at close breakout"
    assert bool(out.intent_long[0, 1]), "Overlay should keep position on same session close window"
    assert bool(out.intent_long[0, 2]), "Overlay should continue into next session when reconfirmed"
    assert bool(out.intent_flat[0, 3]), "Overlay must flatten on rejection flip"
    family_code = getattr(out, "family_code")
    assert family_code is not None
    assert int(family_code[0, 1]) == int(FAMILY_F5_OVERLAY), "t1 should be labeled as overlay hold"


def test_wave1_intent_deterministic_hash() -> None:
    alpha, score, profile, regime, confidence, tradable = _wave1_inputs(T=3)
    cfg = Module4Config(strategy_type="institutional_wave1", family_id="AUTO")

    alpha[0, :, 0] = [1.3, 1.0, 1.1]
    alpha[0, :, 1] = [100.0, 101.0, 102.0]
    alpha[0, :, 3] = [-0.2, 0.08, 0.08]
    score[0, :, int(ScoreIdx.SCORE_BO_LONG)] = [0.6, 0.2, 0.2]
    score[0, :, int(ScoreIdx.SCORE_BO_SHORT)] = [0.1, 0.1, 0.1]
    score[0, :, int(ScoreIdx.SCORE_REJECT)] = [0.2, 0.7, 0.7]
    profile[0, :, int(ProfileStatIdx.D)] = [1.5, 1.2, 2.1]
    profile[0, :, int(ProfileStatIdx.DCLIP)] = [1.5, 2.0, 2.0]
    profile[0, :, int(ProfileStatIdx.Z_DELTA)] = [1.2, -1.2, -1.2]
    profile[0, :, int(ProfileStatIdx.GBREAK)] = [0.7, 0.3, 0.3]
    profile[0, :, int(ProfileStatIdx.GREJECT)] = [0.2, 0.65, 0.65]

    out1 = _run_wave1(alpha=alpha, score=score, profile=profile, regime=regime, confidence=confidence, tradable=tradable, cfg=cfg)
    out2 = _run_wave1(alpha=alpha, score=score, profile=profile, regime=regime, confidence=confidence, tradable=tradable, cfg=cfg)
    np.testing.assert_array_equal(out1.intent_long, out2.intent_long)
    np.testing.assert_array_equal(out1.intent_short, out2.intent_short)
    np.testing.assert_array_equal(getattr(out1, "family_code"), getattr(out2, "family_code"))


def test_f6_parity_pass_and_fail() -> None:
    rng = np.random.default_rng(17)
    shape = (64, 4)
    dclip = rng.normal(0.0, 1.2, size=shape)
    rvol = rng.uniform(0.8, 2.4, size=shape)
    gbreak = rng.uniform(0.15, 0.95, size=shape)
    base_l = 1.0 / (1.0 + np.exp(-(dclip - 1.0))) * rvol
    base_s = 1.0 / (1.0 + np.exp(-((-dclip) - 1.0))) * rvol
    sbo_l = base_l * gbreak
    sbo_s = base_s * gbreak

    pass_report = evaluate_f6_parity(
        dclip=dclip,
        rvol=rvol,
        gbreak=gbreak,
        score_bo_long=sbo_l,
        score_bo_short=sbo_s,
        numeric_tol=1.0e-4,
        behavioral_tol=0.02,
    )
    assert pass_report.passed, f"Expected parity pass, got {pass_report}"
    assert bool(pass_report.strict_behavior_checked) is False

    fail_report = evaluate_f6_parity(
        dclip=dclip,
        rvol=rvol,
        gbreak=gbreak,
        score_bo_long=sbo_l * 1.35,
        score_bo_short=sbo_s * 0.70,
        numeric_tol=1.0e-4,
        behavioral_tol=0.02,
    )
    assert not fail_report.passed, "Parity gate must fail on materially distorted score surfaces"

    # Strict behavioral parity: requires full decision-surface context.
    d_value = dclip.copy()
    score_reject = rng.uniform(0.05, 0.95, size=shape)
    greject = rng.uniform(0.05, 0.95, size=shape)
    z_delta = rng.normal(0.0, 1.4, size=shape)
    x_vah = rng.normal(0.0, 0.25, size=shape)
    x_val = rng.normal(0.0, 0.25, size=shape)

    strict_pass = evaluate_f6_parity(
        dclip=dclip,
        rvol=rvol,
        gbreak=gbreak,
        score_bo_long=sbo_l,
        score_bo_short=sbo_s,
        d_value=d_value,
        score_reject=score_reject,
        greject=greject,
        z_delta=z_delta,
        x_vah=x_vah,
        x_val=x_val,
        numeric_tol=1.0e-4,
        behavioral_tol=0.02,
        require_strict_behavior=True,
    )
    assert strict_pass.passed, f"Strict parity should pass under exact reconstructed surfaces, got {strict_pass}"
    assert bool(strict_pass.strict_behavior_checked) is True

    strict_missing = evaluate_f6_parity(
        dclip=dclip,
        rvol=rvol,
        gbreak=gbreak,
        score_bo_long=sbo_l,
        score_bo_short=sbo_s,
        numeric_tol=1.0e-4,
        behavioral_tol=0.02,
        require_strict_behavior=True,
    )
    assert not strict_missing.passed, "Strict behavioral parity must fail when decision-surface context is missing"
    assert str(strict_missing.reason) == "STRICT_BEHAVIOR_CONTEXT_MISSING"


def test_wave1_server_package_has_no_f7_family(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "generate_wave1_server_package.py"
    parity_input = tmp_path / "missing_parity_input.npz"
    subprocess.run([sys.executable, str(script), "--parity-input", str(parity_input)], check=True, cwd=repo)

    run_cfg_path = repo / "configs" / "server" / "wave1_server_campaign.yaml"
    manifest_path = repo / "configs" / "server" / "wave1_server_package_manifest.json"
    family_policy_path = repo / "configs" / "families" / "wave1_family_policy.yaml"
    assert run_cfg_path.exists(), "Run config must be generated"
    assert manifest_path.exists(), "Package manifest must be generated"
    assert family_policy_path.exists(), "Family policy file must be generated"

    import yaml

    run_cfg = yaml.safe_load(run_cfg_path.read_text(encoding="utf-8"))
    fam_ids = [str(c.get("family_id", "")) for c in run_cfg.get("module4_configs", [])]
    assert "F7" not in fam_ids, "F7 must not exist as standalone family in Wave-1 package"
    assert all(fid in {"F3", "F5", "F6"} for fid in fam_ids), "Only F3/F5/F6 are allowed family ids"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert int(manifest["family_counts"]["F7"]) == 0, "Manifest must declare zero F7 candidates"
    assert bool(manifest["f6_enabled"]) is False, "F6 must fail closed without canonical parity input"
    assert int(manifest["family_counts"]["F6"]) == 0, "F6 candidate budget must collapse to zero when parity gate fails"
    assert int(manifest["candidate_count"]) == 84, "Total candidate budget must be 84 when F6 is disabled"


def test_wave1_server_package_f6_disabled_on_incomplete_parity_context(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "generate_wave1_server_package.py"
    parity_input = tmp_path / "incomplete_parity_input.npz"
    rng = np.random.default_rng(3)
    shape = (8, 8)
    np.savez(
        parity_input,
        dclip=rng.normal(0.0, 1.0, size=shape),
        rvol=rng.uniform(0.8, 2.0, size=shape),
        gbreak=rng.uniform(0.1, 0.9, size=shape),
        score_bo_long=rng.uniform(0.0, 1.0, size=shape),
        score_bo_short=rng.uniform(0.0, 1.0, size=shape),
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--parity-input",
            str(parity_input),
        ],
        check=True,
        cwd=repo,
    )

    parity_report = json.loads((repo / "configs" / "server" / "wave1_f6_parity_report.json").read_text(encoding="utf-8"))
    assert bool(parity_report["passed"]) is False
    assert str(parity_report["reason"]).startswith("MISSING_KEYS:")

    manifest_path = repo / "configs" / "server" / "wave1_server_package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert bool(manifest["f6_enabled"]) is False, "Incomplete parity context must fail closed"
    assert int(manifest["family_counts"]["F6"]) == 0


def test_wave1_server_package_rejects_synthetic_override_flag() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "generate_wave1_server_package.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--allow-synthetic-fallback"],
        check=False,
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, "Synthetic fallback must not be accepted by production package generator"
    assert "unrecognized arguments: --allow-synthetic-fallback" in (proc.stderr or "")


def test_module4_signal_output_contract_remains_locked() -> None:
    from dataclasses import fields

    assert [f.name for f in fields(Module4SignalOutput)] == [
        "regime_primary_ta",
        "regime_confidence_ta",
        "intent_long_ta",
        "intent_short_ta",
        "target_qty_ta",
    ]
