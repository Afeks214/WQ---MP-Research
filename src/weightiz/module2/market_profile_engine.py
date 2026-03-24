from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable, Mapping

import numpy as np

from .market_profile_kernels import (
    InjectionResult,
    ScoreInputs,
    build_bar_mixture_params,
    inject_profile_mass,
)
from .tensor_builder import RollingMoments, apply_rolling_update, clear_rolling_state, init_rolling_profile_state


@dataclass(frozen=True)
class MarketProfileEngineConfig:
    storage_mode: str = "metrics_only"  # metrics_only | full_profile | forensic_windowed
    parallel_backend: str = "serial"  # serial | process_pool
    max_workers: int = 1
    seed: int = 17
    window: int = 60
    warmup: int = 60
    forensic_window_indices: tuple[int, ...] = ()

    @staticmethod
    def from_module2_config(cfg: Any) -> "MarketProfileEngineConfig":
        return MarketProfileEngineConfig(
            storage_mode=str(getattr(cfg, "storage_mode", "metrics_only")),
            parallel_backend=str(getattr(cfg, "parallel_backend", "serial")),
            max_workers=int(getattr(cfg, "max_workers", 1)),
            seed=int(getattr(cfg, "seed", 17)),
            window=int(getattr(cfg, "profile_window_bars", 60)),
            warmup=int(getattr(cfg, "profile_warmup_bars", 60)),
            forensic_window_indices=tuple(int(x) for x in getattr(cfg, "forensic_window_indices", ()) or ()),
        )


@dataclass(frozen=True)
class GoldenManifest:
    dataset_hash: str
    code_hash: str
    spec_version: str
    config_signature: str
    python_version: str
    numpy_version: str
    platform: str
    timezone: str
    seed: int
    schema: dict[str, str]
    generated_at_utc: str


@dataclass
class MarketProfileRunArtifacts:
    computed_mask: np.ndarray
    mixture_history: dict[str, np.ndarray]
    profile_history: dict[str, np.ndarray]
    metric_history: dict[str, np.ndarray]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_array(h: "hashlib._Hash", arr: np.ndarray) -> None:
    a = np.ascontiguousarray(np.asarray(arr))
    h.update(str(a.dtype).encode("utf-8"))
    h.update(str(a.shape).encode("utf-8"))
    h.update(a.tobytes(order="C"))


def build_dataset_hash(*, ts_ns: np.ndarray, symbols: tuple[str, ...], arrays: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    _hash_array(h, np.asarray(ts_ns, dtype=np.int64))
    h.update("|".join(symbols).encode("utf-8"))
    for key in sorted(arrays.keys()):
        h.update(key.encode("utf-8"))
        _hash_array(h, np.asarray(arrays[key]))
    return h.hexdigest()


def build_code_hash(source_files: list[str] | None = None) -> str:
    h = hashlib.sha256()
    if source_files:
        for f in sorted(source_files):
            path = Path(f)
            if not path.exists():
                continue
            h.update(str(path).encode("utf-8"))
            h.update(path.read_bytes())
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit = "nogit"
    h.update(commit.encode("utf-8"))
    return h.hexdigest()


def build_spec_version(*, spec_path: str | None = None, spec_id: str = "main-3") -> str:
    h = hashlib.sha256()
    h.update(str(spec_id).encode("utf-8"))
    if spec_path:
        p = Path(spec_path)
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def build_config_signature(cfg: Any) -> str:
    if hasattr(cfg, "__dataclass_fields__"):
        payload = asdict(cfg)
    elif isinstance(cfg, dict):
        payload = dict(cfg)
    else:
        payload = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_") and not callable(getattr(cfg, k))}
    txt = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def build_golden_manifest(
    *,
    dataset_hash: str,
    code_hash: str,
    spec_version: str,
    config_signature: str,
    timezone: str,
    seed: int,
    schema: Mapping[str, str],
) -> GoldenManifest:
    return GoldenManifest(
        dataset_hash=dataset_hash,
        code_hash=code_hash,
        spec_version=spec_version,
        config_signature=config_signature,
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        platform=platform.platform(),
        timezone=str(timezone),
        seed=int(seed),
        schema=dict(schema),
        generated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )


def write_golden_artifacts(
    *,
    manifest: GoldenManifest,
    payload: Mapping[str, np.ndarray],
    manifest_path: str | Path,
    payload_path: str | Path,
) -> None:
    m_path = Path(manifest_path)
    p_path = Path(payload_path)
    m_path.parent.mkdir(parents=True, exist_ok=True)
    p_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_payload = p_path.with_suffix(p_path.suffix + ".tmp")
    with tmp_payload.open("wb") as f:
        np.savez_compressed(f, **{k: np.asarray(v) for k, v in payload.items()})
    os.replace(tmp_payload, p_path)

    tmp_manifest = m_path.with_suffix(m_path.suffix + ".tmp")
    text = json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":"))
    tmp_manifest.write_text(text, encoding="utf-8")
    os.replace(tmp_manifest, m_path)


def load_golden_manifest(path: str | Path) -> GoldenManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return GoldenManifest(**payload)


def verify_golden_manifest(
    *,
    manifest: GoldenManifest,
    dataset_hash: str,
    code_hash: str,
    spec_version: str,
    config_signature: str,
) -> None:
    checks = {
        "dataset_hash": (manifest.dataset_hash, dataset_hash),
        "code_hash": (manifest.code_hash, code_hash),
        "spec_version": (manifest.spec_version, spec_version),
        "config_signature": (manifest.config_signature, config_signature),
    }
    for key, (got, exp) in checks.items():
        if str(got) != str(exp):
            raise RuntimeError(f"GOLDEN_REPLAY_HASH_MISMATCH: field={key} got={got} expected={exp}")


def benchmark_memory_layout(*, steps: int = 200, assets: int = 256, bins: int = 240, window: int = 60) -> dict[str, float]:
    """
    Deterministic micro-benchmark for ring-update access pattern.
    Returns elapsed seconds for C- and F-ordered ring tensors.
    """
    rng = np.random.default_rng(17)

    def _bench(order: str) -> float:
        ring = np.zeros((window, assets, bins), dtype=np.float64, order=order)
        agg = np.zeros((assets, bins), dtype=np.float64, order="C")
        start = dt.datetime.now(dt.timezone.utc).timestamp()
        for t in range(steps):
            slot = t % window
            incoming = rng.standard_normal((assets, bins), dtype=np.float64)
            agg += incoming - ring[slot]
            ring[slot] = incoming
            _ = np.argmax(agg, axis=1)
        end = dt.datetime.now(dt.timezone.utc).timestamp()
        return float(end - start)

    return {"C": _bench("C"), "F": _bench("F")}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x64))


def _window_median_mad_volume(vol_w: np.ndarray, valid_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Median/MAD over the decision-time working window.

    Semantics are equivalent to nan-median on `where(valid_w, vol_w, nan)`:
    invalid rows are excluded, and empty windows return zeros.
    """
    v = np.asarray(vol_w, dtype=np.float64)
    m = np.asarray(valid_w, dtype=bool)
    if v.shape != m.shape:
        raise RuntimeError(f"volume/valid window shape mismatch: {v.shape} vs {m.shape}")
    _, assets = v.shape

    # Hot-path: all rows valid, avoid nan/ma overhead.
    if bool(np.all(m)):
        med = np.median(v, axis=0)
        mad = np.median(np.abs(v - med[None, :]), axis=0)
        return np.asarray(med, dtype=np.float64), np.asarray(mad, dtype=np.float64)

    med = np.zeros(assets, dtype=np.float64)
    mad = np.zeros(assets, dtype=np.float64)
    for a in range(assets):
        col = v[:, a]
        mask = m[:, a]
        if not np.any(mask):
            continue
        vals = col[mask]
        med_a = float(np.median(vals))
        med[a] = med_a
        mad[a] = float(np.median(np.abs(vals - med_a)))
    return med, mad


def run_streaming_profile_engine(
    *,
    state: Any,
    cfg: Any,
    physics: Any,
    mode: str,
    open_use: np.ndarray,
    high_use: np.ndarray,
    low_use: np.ndarray,
    close_use: np.ndarray,
    vol_use: np.ndarray,
    valid: np.ndarray,
    build_poc_rank_fn: Callable[[np.ndarray], np.ndarray],
    compute_value_area_fn: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    rolling_median_mad_fn: Callable[[np.ndarray, int, int], tuple[np.ndarray, np.ndarray]],
    profile_stat_idx: Any,
    score_idx: Any,
    phase_enum: Any,
    collect_forensics: bool = False,
) -> MarketProfileRunArtifacts:
    mode_norm = str(mode).strip().lower()
    sealed_mode = mode_norm == "sealed"

    eng_cfg = MarketProfileEngineConfig.from_module2_config(cfg)
    T = int(state.cfg.T)
    A = int(state.cfg.A)
    B = int(state.cfg.B)
    W = int(eng_cfg.window)

    x = np.asarray(state.x_grid, dtype=np.float64)
    x2 = x * x
    sqrt_2pi = float(np.sqrt(2.0 * np.pi))
    dx = float(state.cfg.dx)
    idx_zero = int(np.argmin(np.abs(x)))
    eps_pdf = float(state.eps.eps_pdf)
    eps_div_a = np.asarray(state.eps.eps_div, dtype=np.float64)

    rolling = init_rolling_profile_state(window=W, assets=A, bins=B)
    poc_rank = build_poc_rank_fn(x)

    state.vp.fill(0.0)
    state.vp_delta.fill(0.0)
    state.profile_stats.fill(0.0)
    state.scores.fill(0.0)
    state.profile_stats[:, :, int(profile_stat_idx.IPOC)] = float(idx_zero)
    state.profile_stats[:, :, int(profile_stat_idx.IVAH)] = float(idx_zero)
    state.profile_stats[:, :, int(profile_stat_idx.IVAL)] = float(idx_zero)

    computed_mask = np.zeros((T, A), dtype=bool)

    mu_hist = np.zeros((T, A), dtype=np.float64) if collect_forensics else np.zeros((0, 0), dtype=np.float64)
    sigma1_hist = np.zeros((T, A), dtype=np.float64) if collect_forensics else np.zeros((0, 0), dtype=np.float64)
    sigma2_hist = np.zeros((T, A), dtype=np.float64) if collect_forensics else np.zeros((0, 0), dtype=np.float64)
    w1_hist = np.zeros((T, A), dtype=np.float64) if collect_forensics else np.zeros((0, 0), dtype=np.float64)
    w2_hist = np.zeros((T, A), dtype=np.float64) if collect_forensics else np.zeros((0, 0), dtype=np.float64)
    vprof_hist = np.zeros((T, A), dtype=np.float64) if collect_forensics else np.zeros((0, 0), dtype=np.float64)

    gap_reset = np.zeros(T, dtype=bool)
    if T > 1:
        gap_reset[1:] = np.asarray(state.gap_min[1:], dtype=np.float64) > 5.0
    session_start = 0

    for t in range(T):
        reset_boundary = (
            (t == 0)
            or (state.reset_flag[t] == 1)
            or gap_reset[t]
            or (t > 0 and state.session_id[t] != state.session_id[t - 1])
        )
        if reset_boundary:
            clear_rolling_state(rolling)
            session_start = t

        tradable_t = (
            state.bar_valid[t]
            & np.isfinite(close_use[t])
            & np.isfinite(physics.atr_eff[t])
            & (physics.atr_eff[t] > 0.0)
        )

        if sealed_mode:
            w_start = max(session_start, t - W + 1)
            open_w = np.asarray(open_use[w_start : t + 1], dtype=np.float64)
            high_w = np.asarray(high_use[w_start : t + 1], dtype=np.float64)
            low_w = np.asarray(low_use[w_start : t + 1], dtype=np.float64)
            close_w = np.asarray(close_use[w_start : t + 1], dtype=np.float64)
            vol_w = np.asarray(vol_use[w_start : t + 1], dtype=np.float64)
            valid_w = np.asarray(valid[w_start : t + 1], dtype=bool)
            clv_w = np.asarray(physics.clv[w_start : t + 1], dtype=np.float64)
            K = int(open_w.shape[0])

            close_t = np.asarray(close_use[t], dtype=np.float64)
            atr_t = np.asarray(physics.atr_eff[t], dtype=np.float64)
            rvol_t = np.maximum(np.asarray(physics.rvol[t], dtype=np.float64), 0.0)
            ret_norm_t = np.asarray(physics.ret_norm[t], dtype=np.float64)
            s_r_t = np.asarray(physics.s_r[t], dtype=np.float64)

            med_v, mad_v = _window_median_mad_volume(vol_w, valid_w)
            cap_base_t = med_v + 5.0 * mad_v
            cap_eff_t = cap_base_t * (1.0 + np.log(np.maximum(1.0, rvol_t)))
            cap_eff_t = np.where(np.isfinite(cap_eff_t), np.maximum(cap_eff_t, 0.0), 0.0)
            range_w = np.maximum(high_w - low_w, eps_div_a[None, :])
            body_w = np.abs(close_w - open_w)
            body_pct_w = np.clip(body_w / range_w, 0.0, 1.0)

            denom_mu = atr_t[None, :] + eps_div_a[None, :]
            mu_w = (0.5 * (open_w + close_w) - close_t[None, :]) / denom_mu

            w_rvol_t = rvol_t / (1.0 + rvol_t)
            range_eff_w = w_rvol_t[None, :] * range_w + (1.0 - w_rvol_t)[None, :] * atr_t[None, :]
            sigma_base_w = range_eff_w / (4.0 * (atr_t[None, :] + eps_div_a[None, :]))
            sigma1_w = np.maximum(sigma_base_w / (1.0 + np.log1p(rvol_t))[None, :], dx)
            sigma2_w = np.maximum(sigma_base_w, dx)

            s_eff_r_t = np.maximum(s_r_t, 0.5 * dx)
            p_sr_buy_t = _sigmoid(np.log(9.0) * ret_norm_t / (s_eff_r_t + eps_pdf))
            p_clv_buy_w = _sigmoid(6.0 * clv_w)
            p_buy_w = np.clip(body_pct_w * p_sr_buy_t[None, :] + (1.0 - body_pct_w) * p_clv_buy_w, 0.0, 1.0)
            delta_coeff_w = 2.0 * p_buy_w - 1.0

            vprof_w = np.minimum(np.maximum(vol_w, 0.0), cap_eff_t[None, :])
            vprof_w = np.where(np.isfinite(vprof_w), vprof_w, 0.0)

            z1 = (x[None, None, :] - mu_w[:, :, None]) / sigma1_w[:, :, None]
            z2 = (x[None, None, :] - mu_w[:, :, None]) / sigma2_w[:, :, None]
            pdf1 = np.exp(-0.5 * z1 * z1) / (sigma1_w[:, :, None] * sqrt_2pi)
            pdf2 = np.exp(-0.5 * z2 * z2) / (sigma2_w[:, :, None] * sqrt_2pi)

            w1_w = body_pct_w
            w2_w = 1.0 - w1_w
            mix_w = w1_w[:, :, None] * pdf1 + w2_w[:, :, None] * pdf2
            mix_w = np.where(np.isfinite(mix_w), mix_w, 0.0)
            norm_w = np.sum(mix_w, axis=2)
            mix_w = np.divide(
                mix_w,
                norm_w[:, :, None] + eps_pdf,
                out=np.zeros_like(mix_w),
                where=norm_w[:, :, None] > 0.0,
            )

            total_wab = vprof_w[:, :, None] * mix_w
            total_wab = np.where(valid_w[:, :, None], total_wab, 0.0)
            total_wab = np.where(np.isfinite(total_wab), total_wab, 0.0)
            delta_wab = total_wab * delta_coeff_w[:, :, None]
            delta_wab = np.where(np.isfinite(delta_wab), delta_wab, 0.0)

            vp_t = np.sum(total_wab, axis=0)
            vpd_t = np.sum(delta_wab, axis=0)

            if collect_forensics:
                mu_hist[t] = mu_w[-1]
                sigma1_hist[t] = sigma1_w[-1]
                sigma2_hist[t] = sigma2_w[-1]
                w1_hist[t] = w1_w[-1]
                w2_hist[t] = w2_w[-1]
                vprof_hist[t] = vprof_w[-1]
        else:
            score_inputs = ScoreInputs(
                ret_norm=np.asarray(physics.ret_norm[t], dtype=np.float64),
                s_r=np.asarray(physics.s_r[t], dtype=np.float64),
                clv=np.asarray(physics.clv[t], dtype=np.float64),
                body_pct=np.asarray(physics.body_pct[t], dtype=np.float64),
            )

            params = build_bar_mixture_params(
                open_a=np.asarray(open_use[t], dtype=np.float64),
                high_a=np.asarray(high_use[t], dtype=np.float64),
                low_a=np.asarray(low_use[t], dtype=np.float64),
                close_a=np.asarray(close_use[t], dtype=np.float64),
                atr_eff_a=np.asarray(physics.atr_eff[t], dtype=np.float64),
                rvol_a=np.asarray(physics.rvol[t], dtype=np.float64),
                clv_a=np.asarray(physics.clv[t], dtype=np.float64),
                body_pct_a=np.asarray(physics.body_pct[t], dtype=np.float64),
                sigma1_a=np.asarray(physics.sigma1[t], dtype=np.float64),
                sigma2_a=np.asarray(physics.sigma2[t], dtype=np.float64),
                w1_a=np.asarray(physics.w1[t], dtype=np.float64),
                w2_a=np.asarray(physics.w2[t], dtype=np.float64),
                volume_a=np.asarray(vol_use[t], dtype=np.float64),
                cap_v_eff_a=np.asarray(physics.cap_v_eff[t], dtype=np.float64),
                score_inputs=score_inputs,
                eps_div_a=np.asarray(state.eps.eps_div, dtype=np.float64),
                eps_pdf=float(state.eps.eps_pdf),
                dx=dx,
                sealed_mode=False,
                mu1_clv_shift=float(getattr(cfg, "mu1_clv_shift", 0.0)),
                mu2_clv_shift=float(getattr(cfg, "mu2_clv_shift", 0.35)),
            )
            inj = inject_profile_mass(
                params=params,
                x_grid=x,
                dx=dx,
                eps_pdf=float(state.eps.eps_pdf),
                valid_a=np.asarray(valid[t], dtype=bool),
            )

            apply_rolling_update(
                rolling,
                t_index=t,
                inj_total_an=inj.total_an,
                inj_delta_an=inj.delta_an,
                moments=RollingMoments(m0=inj.m0_a, m1=inj.m1_a, m2=inj.m2_a),
            )

            if collect_forensics:
                mu_hist[t] = params.w1 * params.mu1 + params.w2 * params.mu2
                sigma1_hist[t] = params.sigma1
                sigma2_hist[t] = params.sigma2
                w1_hist[t] = params.w1
                w2_hist[t] = params.w2
                vprof_hist[t] = params.vprof

            vp_t = np.asarray(rolling.vp_total_an, dtype=np.float64).copy()
            vpd_t = np.asarray(rolling.vp_delta_an, dtype=np.float64).copy()
        vp_t[~tradable_t] = 0.0
        vpd_t[~tradable_t] = 0.0

        state.vp[t] = vp_t
        state.vp_delta[t] = vpd_t

        total = np.sum(vp_t, axis=1)
        denom = total + float(state.eps.eps_vol)

        # Profile moments derived from the discrete VP grid.
        mu_prof = np.sum(vp_t * x[None, :], axis=1) / denom
        ex2_prof = np.sum(vp_t * x2[None, :], axis=1) / denom
        var_prof = ex2_prof - mu_prof * mu_prof
        var_prof = np.maximum(var_prof, 0.0)
        sigma_prof = np.sqrt(var_prof)

        sigma_eff = np.maximum(sigma_prof, 2.0 * dx)
        d_val = (-mu_prof) / (sigma_eff + float(state.eps.eps_pdf))
        d_clip = np.clip(d_val, -float(getattr(cfg, "d_clip", 6.0)), float(getattr(cfg, "d_clip", 6.0)))

        vp_max = np.max(vp_t, axis=1)
        a_aff = vp_t[:, idx_zero] / (vp_max + float(state.eps.eps_vol))

        max_mass = np.max(vp_t, axis=1, keepdims=True)
        is_max = np.isclose(vp_t, max_mass, atol=float(getattr(cfg, "poc_eq_atol", 0.0)), rtol=0.0)
        masked_rank = np.where(is_max, poc_rank[None, :], B + 1)
        ipoc = np.argmin(masked_rank, axis=1).astype(np.int64)
        ipoc = np.where(total > float(state.eps.eps_vol), ipoc, idx_zero)

        row = np.arange(A, dtype=np.int64)
        delta0 = vpd_t[:, idx_zero] / (vp_t[:, idx_zero] + float(state.eps.eps_vol))
        delta_poc = vpd_t[row, ipoc] / (vp_t[row, ipoc] + float(state.eps.eps_vol))

        wpoc = 1.0 - a_aff
        delta_eff = wpoc * delta_poc + (1.0 - wpoc) * delta0

        ipoc, ivah, ival = compute_value_area_fn(
            vp_ab=vp_t,
            ipoc_a=ipoc,
            x_grid=x,
            va_threshold=float(getattr(cfg, "va_threshold", 0.70)),
            eps_vol=float(state.eps.eps_vol),
        )

        a_idx = np.where(tradable_t)[0]
        if a_idx.size:
            state.profile_stats[t, a_idx, int(profile_stat_idx.MU_PROF)] = mu_prof[a_idx]
            state.profile_stats[t, a_idx, int(profile_stat_idx.SIGMA_PROF)] = sigma_prof[a_idx]
            state.profile_stats[t, a_idx, int(profile_stat_idx.SIGMA_EFF)] = sigma_eff[a_idx]
            state.profile_stats[t, a_idx, int(profile_stat_idx.D)] = d_val[a_idx]
            state.profile_stats[t, a_idx, int(profile_stat_idx.DCLIP)] = d_clip[a_idx]
            state.profile_stats[t, a_idx, int(profile_stat_idx.A_AFFINITY)] = a_aff[a_idx]
            state.profile_stats[t, a_idx, int(profile_stat_idx.DELTA0)] = delta0[a_idx]
            state.profile_stats[t, a_idx, int(profile_stat_idx.DELTA_POC)] = delta_poc[a_idx]
            state.profile_stats[t, a_idx, int(profile_stat_idx.DELTA_EFF)] = delta_eff[a_idx]
            state.profile_stats[t, a_idx, int(profile_stat_idx.IPOC)] = ipoc[a_idx].astype(np.float64)
            state.profile_stats[t, a_idx, int(profile_stat_idx.IVAH)] = ivah[a_idx].astype(np.float64)
            state.profile_stats[t, a_idx, int(profile_stat_idx.IVAL)] = ival[a_idx].astype(np.float64)
            computed_mask[t, a_idx] = True

    delta_eff_all = np.where(computed_mask, state.profile_stats[:, :, int(profile_stat_idx.DELTA_EFF)], np.nan)

    d_delta = np.full((T, A), np.nan, dtype=np.float64)
    for t in range(T):
        mask_t = computed_mask[t]
        if not np.any(mask_t):
            continue
        if t == 0 or state.reset_flag[t] == 1 or gap_reset[t] or state.session_id[t] != state.session_id[t - 1]:
            d_delta[t, mask_t] = 0.0
        else:
            prev = delta_eff_all[t - 1]
            curr = delta_eff_all[t]
            ok = mask_t & np.isfinite(prev) & np.isfinite(curr)
            d_delta[t, ok] = curr[ok] - prev[ok]
            d_delta[t, mask_t & ~ok] = 0.0

    sigma_delta = np.full((T, A), np.nan, dtype=np.float64)
    sid = np.asarray(state.session_id, dtype=np.int64)
    starts = np.where(np.r_[True, (sid[1:] != sid[:-1]) | (state.reset_flag[1:] == 1) | gap_reset[1:]])[0]
    ends = np.r_[starts[1:], T]

    for s, e in zip(starts.tolist(), ends.tolist()):
        seg_level = delta_eff_all[s:e]
        seg_chg = d_delta[s:e]
        _, mad_level_curr = rolling_median_mad_fn(
            seg_level,
            window=int(getattr(cfg, "delta_mad_lookback_bars", 180)),
            min_periods=int(getattr(cfg, "delta_mad_min_periods", 10)),
        )
        _, mad_chg_curr = rolling_median_mad_fn(
            seg_chg,
            window=int(getattr(cfg, "delta_mad_lookback_bars", 180)),
            min_periods=int(getattr(cfg, "delta_mad_min_periods", 10)),
        )
        mad_level = np.full_like(mad_level_curr, np.nan)
        mad_chg = np.full_like(mad_chg_curr, np.nan)
        if mad_level.shape[0] > 1:
            mad_level[1:] = mad_level_curr[:-1]
            mad_chg[1:] = mad_chg_curr[:-1]

        sig_seg = np.maximum(
            np.maximum(
                1.4826 * np.where(np.isfinite(mad_level), mad_level, 0.0),
                1.4826 * np.where(np.isfinite(mad_chg), mad_chg, 0.0),
            ),
            float(getattr(cfg, "sigma_delta_min", 0.05)),
        )
        sigma_delta[s:e] = sig_seg

    valid_post = computed_mask & np.isfinite(delta_eff_all) & np.isfinite(sigma_delta)
    z_delta = np.divide(
        delta_eff_all,
        sigma_delta + float(state.eps.eps_pdf),
        out=np.full((T, A), np.nan, dtype=np.float64),
        where=valid_post,
    )

    ln9 = float(np.log(9.0))
    gbreak = _sigmoid(ln9 * (z_delta - float(getattr(cfg, "delta_gate_threshold", 1.0))))
    greject = _sigmoid(ln9 * (-z_delta - float(getattr(cfg, "delta_gate_threshold", 1.0))))

    dclip_all = state.profile_stats[:, :, int(profile_stat_idx.DCLIP)]
    aff_all = state.profile_stats[:, :, int(profile_stat_idx.A_AFFINITY)]
    rvol_all = np.asarray(physics.rvol, dtype=np.float64)
    body_all = np.asarray(physics.body_pct, dtype=np.float64)

    sbase_bo_long = _sigmoid(dclip_all - float(getattr(cfg, "break_bias", 1.0))) * rvol_all
    sbase_bo_short = _sigmoid((-dclip_all) - float(getattr(cfg, "break_bias", 1.0))) * rvol_all
    rvoltrend = (
        (rvol_all > float(getattr(cfg, "rvol_trend_cutoff", 2.0)))
        & (body_all > float(getattr(cfg, "body_trend_cutoff", 0.60)))
    ).astype(np.float64)
    sbase_reject = (
        _sigmoid(float(getattr(cfg, "reject_center", 2.0)) - np.abs(dclip_all))
        * aff_all
        * (1.0 - rvoltrend)
    )

    score_bo_long = sbase_bo_long * gbreak
    score_bo_short = sbase_bo_short * gbreak
    score_reject = sbase_reject * greject
    score_rej_long = score_reject * _sigmoid(-dclip_all)
    score_rej_short = score_reject * _sigmoid(dclip_all)

    z_chan = state.profile_stats[:, :, int(profile_stat_idx.Z_DELTA)]
    gb_chan = state.profile_stats[:, :, int(profile_stat_idx.GBREAK)]
    gr_chan = state.profile_stats[:, :, int(profile_stat_idx.GREJECT)]
    z_chan[valid_post] = z_delta[valid_post]
    gb_chan[valid_post] = gbreak[valid_post]
    gr_chan[valid_post] = greject[valid_post]

    sc_bo_l = state.scores[:, :, int(score_idx.SCORE_BO_LONG)]
    sc_bo_s = state.scores[:, :, int(score_idx.SCORE_BO_SHORT)]
    sc_rej = state.scores[:, :, int(score_idx.SCORE_REJECT)]
    sc_rej_l = state.scores[:, :, int(score_idx.SCORE_REJ_LONG)]
    sc_rej_s = state.scores[:, :, int(score_idx.SCORE_REJ_SHORT)]

    sc_bo_l[valid_post] = score_bo_long[valid_post]
    sc_bo_s[valid_post] = score_bo_short[valid_post]
    sc_rej[valid_post] = score_reject[valid_post]
    sc_rej_l[valid_post] = score_rej_long[valid_post]
    sc_rej_s[valid_post] = score_rej_short[valid_post]

    warmup_rows = state.phase == np.int8(phase_enum.WARMUP)
    if np.any(warmup_rows):
        state.scores[warmup_rows] = 0.0

    rvol_policy = str(getattr(cfg, "rvol_policy", "neutral_one"))
    if sealed_mode and rvol_policy == "warmup_mask":
        raise RuntimeError("sealed mode forbids rvol_policy='warmup_mask'")
    if rvol_policy == "warmup_mask":
        bad_mask = ~np.asarray(physics.rvol_eligible, dtype=bool)
        state.profile_stats[bad_mask] = np.nan
        state.scores[bad_mask] = np.nan

    invalid_mask = ~state.bar_valid
    if np.any(invalid_mask):
        state.vp[invalid_mask] = 0.0
        state.vp_delta[invalid_mask] = 0.0
        state.scores[invalid_mask] = 0.0
        state.profile_stats[invalid_mask] = 0.0
        state.profile_stats[invalid_mask, int(profile_stat_idx.IPOC)] = float(idx_zero)
        state.profile_stats[invalid_mask, int(profile_stat_idx.IVAH)] = float(idx_zero)
        state.profile_stats[invalid_mask, int(profile_stat_idx.IVAL)] = float(idx_zero)

    if sealed_mode:
        for name, arr in [
            ("vp", state.vp),
            ("vp_delta", state.vp_delta),
            ("profile_stats", state.profile_stats),
            ("scores", state.scores),
        ]:
            if not np.all(np.isfinite(arr)):
                bad = np.argwhere(~np.isfinite(arr))[0]
                raise RuntimeError(f"sealed non-finite output in {name} at index {bad.tolist()}")

    # Enforce immutable output tensors after completion.
    state.vp.flags.writeable = False
    state.vp_delta.flags.writeable = False
    state.profile_stats.flags.writeable = False
    state.scores.flags.writeable = False

    mixture_history = {
        "mu": mu_hist,
        "sigma1": sigma1_hist,
        "sigma2": sigma2_hist,
        "w1": w1_hist,
        "w2": w2_hist,
        "vprof": vprof_hist,
    }
    profile_history = {
        "vp_total": np.asarray(state.vp, dtype=np.float64),
        "vp_delta": np.asarray(state.vp_delta, dtype=np.float64),
    }
    metric_history = {
        "ipoc": np.asarray(state.profile_stats[:, :, int(profile_stat_idx.IPOC)], dtype=np.float64),
        "ival": np.asarray(state.profile_stats[:, :, int(profile_stat_idx.IVAL)], dtype=np.float64),
        "ivah": np.asarray(state.profile_stats[:, :, int(profile_stat_idx.IVAH)], dtype=np.float64),
        "d": np.asarray(state.profile_stats[:, :, int(profile_stat_idx.D)], dtype=np.float64),
        "affinity": np.asarray(state.profile_stats[:, :, int(profile_stat_idx.A_AFFINITY)], dtype=np.float64),
        "delta_eff": np.asarray(state.profile_stats[:, :, int(profile_stat_idx.DELTA_EFF)], dtype=np.float64),
        "breakout": np.asarray(state.scores[:, :, int(score_idx.SCORE_BO_LONG)], dtype=np.float64),
        "rejection": np.asarray(state.scores[:, :, int(score_idx.SCORE_REJECT)], dtype=np.float64),
    }
    return MarketProfileRunArtifacts(
        computed_mask=computed_mask,
        mixture_history=mixture_history,
        profile_history=profile_history,
        metric_history=metric_history,
    )
