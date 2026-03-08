from __future__ import annotations

import itertools
from typing import Any

import numpy as np


def session_bounds(session_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sid = np.asarray(session_id, dtype=np.int64)
    T = int(sid.shape[0])
    starts = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]]).astype(np.int64)
    ends = np.r_[starts[1:], T].astype(np.int64)
    sessions = sid[starts]
    return starts, ends, sessions


def sessions_to_idx(session_id: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    mask = np.isin(session_id, sessions)
    return np.flatnonzero(mask).astype(np.int64)


def contiguous_segments(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    cut = np.flatnonzero(np.diff(idx) > 1) + 1
    starts = np.r_[idx[0], idx[cut]]
    ends = np.r_[idx[cut - 1] + 1, idx[-1] + 1]
    return starts.astype(np.int64), ends.astype(np.int64)


def apply_purge_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    T: int,
    purge_bars: int,
    embargo_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tr = np.unique(np.asarray(train_idx, dtype=np.int64))
    te = np.unique(np.asarray(test_idx, dtype=np.int64))

    seg_starts, _seg_ends = contiguous_segments(te)

    purge_mask = np.zeros(T, dtype=bool)
    for t0 in seg_starts.tolist():
        lo = max(0, int(t0) - int(purge_bars))
        hi = int(t0)
        if hi > lo:
            purge_mask[lo:hi] = True

    embargo_mask = np.zeros(T, dtype=bool)
    for t0 in seg_starts.tolist():
        lo = int(t0)
        hi = min(T, int(t0) + int(embargo_bars))
        if hi > lo:
            embargo_mask[lo:hi] = True

    purge_idx = np.flatnonzero(purge_mask & np.isin(np.arange(T, dtype=np.int64), tr)).astype(np.int64)
    embargo_idx = np.flatnonzero(embargo_mask & np.isin(np.arange(T, dtype=np.int64), te)).astype(np.int64)

    tr2 = tr[~np.isin(tr, purge_idx)]
    te2 = te[~np.isin(te, embargo_idx)]

    return tr2.astype(np.int64), te2.astype(np.int64), purge_idx, embargo_idx


def generate_wf_splits(state: Any, cfg: Any, *, split_spec_cls: Any) -> list[Any]:
    starts, ends, sessions = session_bounds(state.session_id)
    n_s = int(sessions.size)

    train_n = int(cfg.wf_train_sessions)
    test_n = int(cfg.wf_test_sessions)
    step_n = int(cfg.wf_step_sessions)

    out: list[Any] = []
    if n_s < train_n + test_n:
        return out

    sid = state.session_id.astype(np.int64)
    fold = 0
    s0 = 0
    while s0 + train_n + test_n <= n_s:
        tr_s = sessions[s0 : s0 + train_n]
        te_s = sessions[s0 + train_n : s0 + train_n + test_n]

        tr_idx = sessions_to_idx(sid, tr_s)
        te_idx = sessions_to_idx(sid, te_s)

        tr_idx, te_idx, purge_idx, embargo_idx = apply_purge_embargo(
            tr_idx,
            te_idx,
            state.cfg.T,
            cfg.purge_bars,
            cfg.embargo_bars,
        )

        out.append(
            split_spec_cls(
                split_id=f"wf_{fold:03d}",
                mode="wf",
                train_idx=tr_idx,
                test_idx=te_idx,
                purge_idx=purge_idx,
                embargo_idx=embargo_idx,
                session_train_bounds=(int(tr_s[0]), int(tr_s[-1])),
                session_test_bounds=(int(te_s[0]), int(te_s[-1])),
                purge_bars=int(cfg.purge_bars),
                embargo_bars=int(cfg.embargo_bars),
                total_bars=int(state.cfg.T),
            )
        )
        fold += 1
        s0 += max(1, step_n)

    return out


def generate_cpcv_splits(state: Any, cfg: Any, *, split_spec_cls: Any) -> list[Any]:
    starts, ends, sessions = session_bounds(state.session_id)
    n_s = int(sessions.size)
    S = int(cfg.cpcv_slices)
    k = int(cfg.cpcv_k_test)

    if S < 2 or k < 1 or k >= S:
        raise RuntimeError(f"Invalid CPCV params: slices={S}, k_test={k}")
    if n_s < S:
        return []

    groups = np.array_split(np.arange(n_s, dtype=np.int64), S)
    out: list[Any] = []
    sid = state.session_id.astype(np.int64)

    comb_iter = itertools.combinations(range(S), k)
    for i, test_grp_idx in enumerate(comb_iter):
        test_loc = np.concatenate([groups[g] for g in test_grp_idx if groups[g].size > 0]).astype(np.int64)
        if test_loc.size == 0:
            continue
        train_loc_mask = np.ones(n_s, dtype=bool)
        train_loc_mask[test_loc] = False
        train_loc = np.where(train_loc_mask)[0].astype(np.int64)
        if train_loc.size == 0:
            continue

        tr_s = sessions[train_loc]
        te_s = sessions[test_loc]

        tr_idx = sessions_to_idx(sid, tr_s)
        te_idx = sessions_to_idx(sid, te_s)

        tr_idx, te_idx, purge_idx, embargo_idx = apply_purge_embargo(
            tr_idx,
            te_idx,
            state.cfg.T,
            cfg.purge_bars,
            cfg.embargo_bars,
        )

        out.append(
            split_spec_cls(
                split_id=f"cpcv_{i:03d}",
                mode="cpcv",
                train_idx=tr_idx,
                test_idx=te_idx,
                purge_idx=purge_idx,
                embargo_idx=embargo_idx,
                session_train_bounds=(int(np.min(tr_s)), int(np.max(tr_s))),
                session_test_bounds=(int(np.min(te_s)), int(np.max(te_s))),
                purge_bars=int(cfg.purge_bars),
                embargo_bars=int(cfg.embargo_bars),
                total_bars=int(state.cfg.T),
            )
        )

    return out


def generate_quick_fallback_split(state: Any, cfg: Any, *, split_spec_cls: Any) -> list[Any]:
    T = int(state.cfg.T)
    if T < 2:
        return []
    cut = int(max(1, min(T - 1, T // 2)))
    tr_idx = np.arange(0, cut, dtype=np.int64)
    te_idx = np.arange(cut, T, dtype=np.int64)
    tr_idx, te_idx, purge_idx, embargo_idx = apply_purge_embargo(
        tr_idx,
        te_idx,
        T,
        int(max(0, cfg.purge_bars)),
        int(max(0, cfg.embargo_bars)),
    )
    if tr_idx.size == 0 or te_idx.size == 0:
        return []
    sid = state.session_id.astype(np.int64)
    return [
        split_spec_cls(
            split_id="wf_quick_000",
            mode="wf",
            train_idx=tr_idx,
            test_idx=te_idx,
            purge_idx=purge_idx,
            embargo_idx=embargo_idx,
            session_train_bounds=(int(np.min(sid[tr_idx])), int(np.max(sid[tr_idx]))),
            session_test_bounds=(int(np.min(sid[te_idx])), int(np.max(sid[te_idx]))),
            purge_bars=int(max(0, cfg.purge_bars)),
            embargo_bars=int(max(0, cfg.embargo_bars)),
            total_bars=T,
        )
    ]


def validate_split(spec: Any, enforce_guard: bool, *, contiguous_segments_fn: Any) -> None:
    tr = np.asarray(spec.train_idx, dtype=np.int64)
    te = np.asarray(spec.test_idx, dtype=np.int64)
    purge = np.asarray(spec.purge_idx, dtype=np.int64)
    embargo = np.asarray(spec.embargo_idx, dtype=np.int64)

    if tr.size == 0 or te.size == 0:
        raise RuntimeError(f"Split {spec.split_id} has empty train or test index set")
    if np.any(np.diff(tr) < 0) or np.any(np.diff(te) < 0):
        raise RuntimeError(f"Split {spec.split_id} indices must be sorted")
    if tr.size != np.unique(tr).size or te.size != np.unique(te).size:
        raise RuntimeError(f"Split {spec.split_id} indices must be unique")
    inter = np.intersect1d(tr, te)
    if inter.size > 0:
        raise RuntimeError(f"Split {spec.split_id} leakage: train/test overlap exists")

    if enforce_guard:
        if tr.size != np.unique(tr).size:
            raise RuntimeError(f"Split {spec.split_id} train_idx must be unique")
        if te.size != np.unique(te).size:
            raise RuntimeError(f"Split {spec.split_id} test_idx must be unique")
        if purge.size != np.unique(purge).size:
            raise RuntimeError(f"Split {spec.split_id} purge_idx must be unique")
        if embargo.size != np.unique(embargo).size:
            raise RuntimeError(f"Split {spec.split_id} embargo_idx must be unique")

        all_idx = np.r_[tr, te, purge, embargo].astype(np.int64, copy=False)
        if all_idx.size == 0:
            raise RuntimeError(f"Split {spec.split_id} has no indices to validate")
        total_bars = int(spec.total_bars) if int(spec.total_bars) > 0 else int(np.max(all_idx) + 1)
        if np.any(all_idx < 0) or np.any(all_idx >= total_bars):
            bad = int(all_idx[(all_idx < 0) | (all_idx >= total_bars)][0])
            raise RuntimeError(
                f"Split {spec.split_id} has out-of-range index {bad} for total_bars={total_bars}"
            )

        leak_tp = np.intersect1d(tr, purge)
        if leak_tp.size > 0:
            i = int(leak_tp[0])
            raise RuntimeError(
                f"Split {spec.split_id} purge leakage: train_idx intersects purge_idx, "
                f"first_offending_index={i}"
            )

        leak_te = np.intersect1d(te, embargo)
        if leak_te.size > 0:
            i = int(leak_te[0])
            raise RuntimeError(
                f"Split {spec.split_id} embargo leakage: test_idx intersects embargo_idx, "
                f"first_offending_index={i}"
            )

        purge_bars = int(max(0, spec.purge_bars))
        embargo_bars = int(max(0, spec.embargo_bars))

        train_full = np.unique(np.r_[tr, purge]).astype(np.int64)
        test_full = np.unique(np.r_[te, embargo]).astype(np.int64)
        seg_starts, _seg_ends = contiguous_segments_fn(test_full)

        for t0 in seg_starts.tolist():
            t0_i = int(t0)

            lo_p = max(0, t0_i - purge_bars)
            hi_p = t0_i
            if hi_p > lo_p:
                bad_train = tr[(tr >= lo_p) & (tr < hi_p)]
                if bad_train.size > 0:
                    i = int(bad_train[0])
                    raise RuntimeError(
                        f"Split {spec.split_id} purge guard violated: "
                        f"forbidden_train_range=[{lo_p},{hi_p}) first_offending_index={i}"
                    )
                expected_purge = train_full[(train_full >= lo_p) & (train_full < hi_p)]
                if expected_purge.size > 0:
                    missing = expected_purge[~np.isin(expected_purge, purge)]
                    if missing.size > 0:
                        i = int(missing[0])
                        raise RuntimeError(
                            f"Split {spec.split_id} purge carve incomplete: "
                            f"required_range=[{lo_p},{hi_p}) first_offending_index={i}"
                        )

            lo_e = t0_i
            hi_e = min(total_bars, t0_i + embargo_bars)
            if hi_e > lo_e:
                bad_test = te[(te >= lo_e) & (te < hi_e)]
                if bad_test.size > 0:
                    i = int(bad_test[0])
                    raise RuntimeError(
                        f"Split {spec.split_id} embargo guard violated: "
                        f"forbidden_test_range=[{lo_e},{hi_e}) first_offending_index={i}"
                    )
                expected_embargo = test_full[(test_full >= lo_e) & (test_full < hi_e)]
                if expected_embargo.size > 0:
                    missing = expected_embargo[~np.isin(expected_embargo, embargo)]
                    if missing.size > 0:
                        i = int(missing[0])
                        raise RuntimeError(
                            f"Split {spec.split_id} embargo carve incomplete: "
                            f"required_range=[{lo_e},{hi_e}) first_offending_index={i}"
                        )


def default_stress_scenarios(cfg: Any, *, stress_scenario_cls: Any) -> list[Any]:
    if cfg.stress_profile == "baseline_mild_severe":
        return [
            stress_scenario_cls(
                scenario_id="baseline",
                name="baseline",
                missing_burst_prob=0.0,
                missing_burst_min=0,
                missing_burst_max=0,
                jitter_sigma_bps=0.0,
                slippage_mult=1.0,
                enabled=True,
            ),
            stress_scenario_cls(
                scenario_id="mild",
                name="mild",
                missing_burst_prob=0.0005,
                missing_burst_min=2,
                missing_burst_max=5,
                jitter_sigma_bps=1.5,
                slippage_mult=1.5,
                enabled=True,
            ),
            stress_scenario_cls(
                scenario_id="severe",
                name="severe",
                missing_burst_prob=0.0020,
                missing_burst_min=5,
                missing_burst_max=20,
                jitter_sigma_bps=4.0,
                slippage_mult=3.0,
                enabled=True,
            ),
        ]
    raise RuntimeError(f"Unsupported stress_profile: {cfg.stress_profile}")


def build_candidate_specs_default(
    A: int,
    m2_configs: list[Any],
    m3_configs: list[Any],
    m4_configs: list[Any],
    *,
    candidate_spec_cls: Any,
) -> list[Any]:
    all_on = np.ones(A, dtype=bool)
    out: list[Any] = []
    cid = 0
    for i2 in range(len(m2_configs)):
        for i3 in range(len(m3_configs)):
            for i4 in range(len(m4_configs)):
                out.append(
                    candidate_spec_cls(
                        candidate_id=f"cand_{cid:04d}_m2{i2}_m3{i3}_m4{i4}",
                        m2_idx=i2,
                        m3_idx=i3,
                        m4_idx=i4,
                        enabled_assets_mask=all_on.copy(),
                        tags=(),
                    )
                )
                cid += 1
    return out


def normalize_candidate_specs(
    specs: list[Any],
    keep_idx: np.ndarray,
    A_filtered: int,
    A_input: int,
    *,
    candidate_spec_cls: Any,
) -> list[Any]:
    out: list[Any] = []
    for spec in specs:
        m = np.asarray(spec.enabled_assets_mask, dtype=bool)
        if m.shape == (A_input,):
            m2 = m[keep_idx]
        elif m.shape == (A_filtered,):
            m2 = m.copy()
        else:
            raise RuntimeError(
                f"Candidate {spec.candidate_id} enabled_assets_mask has invalid shape {m.shape}; "
                f"expected {(A_input,)} or {(A_filtered,)}"
            )
        out.append(
            candidate_spec_cls(
                candidate_id=spec.candidate_id,
                m2_idx=int(spec.m2_idx),
                m3_idx=int(spec.m3_idx),
                m4_idx=int(spec.m4_idx),
                enabled_assets_mask=m2.astype(bool, copy=True),
                tags=tuple(spec.tags),
            )
        )
    return out


def build_group_tasks(candidates: list[Any], splits: list[Any], scenarios: list[Any], *, group_task_cls: Any) -> list[Any]:
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for ci, c in enumerate(candidates):
        for si, _sp in enumerate(splits):
            for xi, _sc in enumerate(scenarios):
                key = (si, xi, int(c.m2_idx), int(c.m3_idx))
                groups.setdefault(key, []).append(int(ci))
    out: list[Any] = []
    for gi, ((si, xi, i2, i3), cand_idx) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
        out.append(
            group_task_cls(
                group_id=f"g_{gi:05d}",
                split_idx=int(si),
                scenario_idx=int(xi),
                m2_idx=int(i2),
                m3_idx=int(i3),
                candidate_indices=tuple(int(x) for x in cand_idx),
            )
        )
    return out
