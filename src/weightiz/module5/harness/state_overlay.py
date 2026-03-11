from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from weightiz.module3.types import Module3Output
from weightiz.module1.core import OrderIdx, ProfileStatIdx, ScoreIdx, TensorState


def _readonly_view(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    try:
        out.setflags(write=False)
    except Exception:
        pass
    return out


def _tensor_nbytes_total(obj: Any, _seen: set[int] | None = None) -> int:
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return 0
    _seen.add(oid)
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return int(sum(_tensor_nbytes_total(v, _seen) for v in obj.values()))
    if isinstance(obj, (list, tuple)):
        return int(sum(_tensor_nbytes_total(v, _seen) for v in obj))
    if hasattr(obj, "__dict__"):
        return int(sum(_tensor_nbytes_total(v, _seen) for v in vars(obj).values()))
    return 0


def measure_module3_output_bytes(m3: Module3Output | None) -> int:
    if m3 is None:
        return 0
    return int(_tensor_nbytes_total(m3))


def freeze_module3_output(m3: Module3Output | None) -> None:
    if m3 is None:
        return
    for value in vars(m3).values():
        if isinstance(value, np.ndarray):
            try:
                value.setflags(write=False)
            except Exception:
                pass


@dataclass(frozen=True)
class BaseTensorState:
    cfg: Any
    symbols: tuple[str, ...]
    eps: Any
    ts_ns: np.ndarray
    minute_of_day: np.ndarray
    tod: np.ndarray
    session_id: np.ndarray
    gap_min: np.ndarray
    reset_flag: np.ndarray
    phase: np.ndarray
    x_grid: np.ndarray
    open_px: np.ndarray
    high_px: np.ndarray
    low_px: np.ndarray
    close_px: np.ndarray
    volume: np.ndarray
    bar_valid: np.ndarray
    orders: np.ndarray
    order_side: np.ndarray
    order_flags: np.ndarray
    position_qty: np.ndarray
    overnight_mask: np.ndarray
    available_cash: np.ndarray
    equity: np.ndarray
    margin_used: np.ndarray
    buying_power: np.ndarray
    realized_pnl: np.ndarray
    unrealized_pnl: np.ndarray
    daily_loss: np.ndarray
    daily_loss_breach_flag: np.ndarray
    leverage_limit: np.ndarray
    dqs_day_ta: np.ndarray | None = None
    shared_feature_tensor: np.ndarray | None = None

    @classmethod
    def from_tensor_state(cls, state: TensorState) -> "BaseTensorState":
        arrays = {
            "ts_ns": _readonly_view(state.ts_ns),
            "minute_of_day": _readonly_view(state.minute_of_day),
            "tod": _readonly_view(state.tod),
            "session_id": _readonly_view(state.session_id),
            "gap_min": _readonly_view(state.gap_min),
            "reset_flag": _readonly_view(state.reset_flag),
            "phase": _readonly_view(state.phase),
            "x_grid": _readonly_view(state.x_grid),
            "open_px": _readonly_view(state.open_px),
            "high_px": _readonly_view(state.high_px),
            "low_px": _readonly_view(state.low_px),
            "close_px": _readonly_view(state.close_px),
            "volume": _readonly_view(state.volume),
            "bar_valid": _readonly_view(state.bar_valid),
            "orders": _readonly_view(state.orders),
            "order_side": _readonly_view(state.order_side),
            "order_flags": _readonly_view(state.order_flags),
            "position_qty": _readonly_view(state.position_qty),
            "overnight_mask": _readonly_view(state.overnight_mask),
            "available_cash": _readonly_view(state.available_cash),
            "equity": _readonly_view(state.equity),
            "margin_used": _readonly_view(state.margin_used),
            "buying_power": _readonly_view(state.buying_power),
            "realized_pnl": _readonly_view(state.realized_pnl),
            "unrealized_pnl": _readonly_view(state.unrealized_pnl),
            "daily_loss": _readonly_view(state.daily_loss),
            "daily_loss_breach_flag": _readonly_view(state.daily_loss_breach_flag),
            "leverage_limit": _readonly_view(state.leverage_limit),
        }
        dqs = getattr(state, "dqs_day_ta", None)
        shared = getattr(state, "shared_feature_tensor", None)
        return cls(
            cfg=state.cfg,
            symbols=tuple(str(s) for s in state.symbols),
            eps=state.eps,
            dqs_day_ta=None if dqs is None else _readonly_view(np.asarray(dqs, dtype=np.float64)),
            shared_feature_tensor=None if shared is None else _readonly_view(np.asarray(shared, dtype=np.float64)),
            **arrays,
        )

    @property
    def base_bytes(self) -> int:
        return _tensor_nbytes_total(self)


@dataclass
class MarketOverlay:
    open_px: np.ndarray
    high_px: np.ndarray
    low_px: np.ndarray
    close_px: np.ndarray
    volume: np.ndarray
    bar_valid: np.ndarray
    active_t: np.ndarray

    @classmethod
    def from_base(cls, base: BaseTensorState) -> "MarketOverlay":
        t_count, a_count = base.open_px.shape
        return cls(
            open_px=np.array(base.open_px, copy=True),
            high_px=np.array(base.high_px, copy=True),
            low_px=np.array(base.low_px, copy=True),
            close_px=np.array(base.close_px, copy=True),
            volume=np.array(base.volume, copy=True),
            bar_valid=np.array(base.bar_valid, copy=True),
            active_t=np.zeros(t_count, dtype=bool),
        )

    @property
    def nbytes(self) -> int:
        return _tensor_nbytes_total(self)


@dataclass
class FeatureOverlay:
    rvol: np.ndarray
    atr_floor: np.ndarray
    vp: np.ndarray
    vp_delta: np.ndarray
    profile_stats: np.ndarray
    scores: np.ndarray
    module3_output: Module3Output | None = None
    group_invariant_reason_codes: tuple[str, ...] = ()
    module3_group_bytes: int = 0
    module3_output_mode: str = "full_legacy"

    @classmethod
    def allocate(cls, base: BaseTensorState, *, module3_output_mode: str = "full_legacy") -> "FeatureOverlay":
        t_count, a_count = base.open_px.shape
        b_count = int(base.cfg.B)
        return cls(
            rvol=np.full((t_count, a_count), np.nan, dtype=np.float64),
            atr_floor=np.full((t_count, a_count), np.nan, dtype=np.float64),
            vp=np.zeros((t_count, a_count, b_count), dtype=np.float64),
            vp_delta=np.zeros((t_count, a_count, b_count), dtype=np.float64),
            profile_stats=np.zeros((t_count, a_count, int(ProfileStatIdx.N_FIELDS)), dtype=np.float64),
            scores=np.zeros((t_count, a_count, int(ScoreIdx.N_FIELDS)), dtype=np.float64),
            module3_output_mode=str(module3_output_mode),
        )

    @property
    def feature_bytes(self) -> int:
        return int(
            self.rvol.nbytes
            + self.atr_floor.nbytes
            + self.vp.nbytes
            + self.vp_delta.nbytes
            + self.profile_stats.nbytes
            + self.scores.nbytes
        )

    @property
    def nbytes(self) -> int:
        return int(self.feature_bytes + int(self.module3_group_bytes))

    def freeze_for_candidate_loop(self) -> None:
        for arr in (self.rvol, self.atr_floor, self.vp, self.vp_delta, self.profile_stats, self.scores):
            try:
                arr.setflags(write=False)
            except Exception:
                pass
        freeze_module3_output(self.module3_output)


@dataclass
class CandidateScratch:
    mode: str
    risk_res: Any | None = None
    m4_sig: Any | None = None
    orders: np.ndarray | None = None
    order_side: np.ndarray | None = None
    order_flags: np.ndarray | None = None
    position_qty: np.ndarray | None = None
    overnight_mask: np.ndarray | None = None
    available_cash: np.ndarray | None = None
    equity: np.ndarray | None = None
    margin_used: np.ndarray | None = None
    buying_power: np.ndarray | None = None
    realized_pnl: np.ndarray | None = None
    unrealized_pnl: np.ndarray | None = None
    daily_loss: np.ndarray | None = None
    daily_loss_breach_flag: np.ndarray | None = None
    leverage_limit: np.ndarray | None = None

    @classmethod
    def template_from_base(cls, base: BaseTensorState) -> "CandidateScratch":
        # Group-build validation still expects the historical execution tensor
        # surface to exist structurally. These read-only views provide that
        # surface without granting candidate-local mutability before the
        # candidate loop attaches a real scratch instance.
        return cls(
            mode="template",
            orders=base.orders,
            order_side=base.order_side,
            order_flags=base.order_flags,
            position_qty=base.position_qty,
            overnight_mask=base.overnight_mask,
            available_cash=base.available_cash,
            equity=base.equity,
            margin_used=base.margin_used,
            buying_power=base.buying_power,
            realized_pnl=base.realized_pnl,
            unrealized_pnl=base.unrealized_pnl,
            daily_loss=base.daily_loss,
            daily_loss_breach_flag=base.daily_loss_breach_flag,
            leverage_limit=base.leverage_limit,
        )

    @classmethod
    def allocate(cls, base: BaseTensorState, mode: str) -> "CandidateScratch":
        selected = str(mode).strip().lower()
        t_count, a_count = base.open_px.shape
        if selected == "compact":
            return cls(
                mode="compact",
                order_side=np.zeros((t_count, a_count), dtype=np.int8),
                order_flags=np.zeros((t_count, a_count), dtype=np.uint16),
                position_qty=np.zeros((t_count, a_count), dtype=np.float64),
                equity=np.full(t_count, float(base.cfg.initial_cash), dtype=np.float64),
                margin_used=np.zeros(t_count, dtype=np.float64),
                buying_power=np.full(t_count, float(base.cfg.initial_cash), dtype=np.float64),
                daily_loss=np.zeros(t_count, dtype=np.float64),
                daily_loss_breach_flag=np.zeros(t_count, dtype=np.int8),
            )
        if selected != "full":
            raise RuntimeError(f"Unsupported CandidateScratch mode={mode!r}")
        return cls(
            mode="full",
            orders=np.array(base.orders, copy=True),
            order_side=np.array(base.order_side, copy=True),
            order_flags=np.array(base.order_flags, copy=True),
            position_qty=np.array(base.position_qty, copy=True),
            overnight_mask=np.array(base.overnight_mask, copy=True),
            available_cash=np.array(base.available_cash, copy=True),
            equity=np.array(base.equity, copy=True),
            margin_used=np.array(base.margin_used, copy=True),
            buying_power=np.array(base.buying_power, copy=True),
            realized_pnl=np.array(base.realized_pnl, copy=True),
            unrealized_pnl=np.array(base.unrealized_pnl, copy=True),
            daily_loss=np.array(base.daily_loss, copy=True),
            daily_loss_breach_flag=np.array(base.daily_loss_breach_flag, copy=True),
            leverage_limit=np.array(base.leverage_limit, copy=True),
        )

    def reset_from_base(self, base: BaseTensorState) -> None:
        self.risk_res = None
        self.m4_sig = None
        if self.mode == "template":
            raise RuntimeError("Template candidate scratch cannot be reset for candidate execution")
        if self.mode == "compact":
            assert self.order_side is not None
            assert self.order_flags is not None
            assert self.position_qty is not None
            assert self.equity is not None
            assert self.margin_used is not None
            assert self.buying_power is not None
            assert self.daily_loss is not None
            assert self.daily_loss_breach_flag is not None
            self.order_side[:, :] = 0
            self.order_flags[:, :] = np.uint16(0)
            self.position_qty[:, :] = 0.0
            self.equity[:] = float(base.cfg.initial_cash)
            self.margin_used[:] = 0.0
            self.buying_power[:] = float(base.cfg.initial_cash)
            self.daily_loss[:] = 0.0
            self.daily_loss_breach_flag[:] = 0
            return
        assert self.orders is not None
        assert self.order_side is not None
        assert self.order_flags is not None
        assert self.position_qty is not None
        assert self.overnight_mask is not None
        assert self.available_cash is not None
        assert self.equity is not None
        assert self.margin_used is not None
        assert self.buying_power is not None
        assert self.realized_pnl is not None
        assert self.unrealized_pnl is not None
        assert self.daily_loss is not None
        assert self.daily_loss_breach_flag is not None
        assert self.leverage_limit is not None
        self.orders[:, :, :] = base.orders
        self.order_side[:, :] = 0
        self.order_flags[:, :] = np.uint16(0)
        self.position_qty[:, :] = 0.0
        self.overnight_mask[:, :] = 0
        self.available_cash[:] = base.available_cash
        self.equity[:] = base.equity
        self.margin_used[:] = base.margin_used
        self.buying_power[:] = base.buying_power
        self.realized_pnl[:] = base.realized_pnl
        self.unrealized_pnl[:] = base.unrealized_pnl
        self.daily_loss[:] = base.daily_loss
        self.daily_loss_breach_flag[:] = base.daily_loss_breach_flag
        self.leverage_limit[:] = base.leverage_limit

    @property
    def nbytes(self) -> int:
        if self.mode == "template":
            return 0
        return _tensor_nbytes_total(self)


class CombinedStateView:
    __slots__ = ("base", "market_overlay", "feature_overlay", "candidate_scratch", "asset_enabled_mask")
    # Explicit ownership surface for the worker hot path.
    #
    # BaseTensorState (read-only):
    #   cfg, symbols, eps, ts_ns, minute_of_day, tod, session_id, gap_min,
    #   reset_flag, phase, x_grid, dqs_day_ta, shared_feature_tensor
    #
    # MarketOverlay (group-local writable):
    #   open_px, high_px, low_px, close_px, volume, bar_valid, active_t
    #
    # FeatureOverlay (group-local writable until freeze_for_candidate_loop()):
    #   rvol, atr_floor, vp, vp_delta, profile_stats, scores
    #
    # CandidateScratch (candidate-local writable, fail-closed if unavailable in
    # the active scratch mode):
    #   orders, order_side, order_flags, position_qty, overnight_mask,
    #   available_cash, equity, margin_used, buying_power, realized_pnl,
    #   unrealized_pnl, daily_loss, daily_loss_breach_flag, leverage_limit
    #
    # CandidateScratch(mode="template") is the only allowed non-candidate
    # variant. It provides read-only structural templates during group build so
    # Module2 validation can inspect the historical TensorState surface without
    # granting pre-candidate mutability.
    #
    # No broad TensorState emulation is allowed here. If a field is not exposed
    # explicitly below, the worker runtime must not assume it exists.

    def __init__(
        self,
        base: BaseTensorState,
        market_overlay: MarketOverlay,
        feature_overlay: FeatureOverlay,
        candidate_scratch: CandidateScratch | None,
        *,
        asset_enabled_mask: np.ndarray | None = None,
    ) -> None:
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "market_overlay", market_overlay)
        object.__setattr__(self, "feature_overlay", feature_overlay)
        object.__setattr__(self, "candidate_scratch", candidate_scratch)
        if asset_enabled_mask is None:
            asset_enabled_mask = np.ones(int(base.cfg.A), dtype=bool)
        object.__setattr__(self, "asset_enabled_mask", np.asarray(asset_enabled_mask, dtype=bool))

    def _require_scratch_field(self, name: str) -> np.ndarray:
        scratch = self.candidate_scratch
        if scratch is None:
            raise RuntimeError(f"Candidate scratch is not attached; missing writable field {name}")
        value = getattr(scratch, name, None)
        if value is None:
            raise RuntimeError(
                f"Candidate scratch mode={scratch.mode!r} does not expose required writable field {name}"
            )
        return value

    def _set_market_field(self, name: str, value: Any) -> None:
        arr = getattr(self.market_overlay, name)
        np.asarray(arr)[...] = value

    def _set_feature_field(self, name: str, value: Any) -> None:
        arr = getattr(self.feature_overlay, name)
        np.asarray(arr)[...] = value

    def _set_scratch_field(self, name: str, value: Any) -> None:
        scratch = self._require_scratch_field(name)
        scratch[...] = value

    @property
    def cfg(self) -> Any:
        return self.base.cfg

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.base.symbols

    @property
    def eps(self) -> Any:
        return self.base.eps

    @property
    def ts_ns(self) -> np.ndarray:
        return self.base.ts_ns

    @property
    def minute_of_day(self) -> np.ndarray:
        return self.base.minute_of_day

    @property
    def tod(self) -> np.ndarray:
        return self.base.tod

    @property
    def session_id(self) -> np.ndarray:
        return self.base.session_id

    @property
    def gap_min(self) -> np.ndarray:
        return self.base.gap_min

    @property
    def reset_flag(self) -> np.ndarray:
        return self.base.reset_flag

    @property
    def phase(self) -> np.ndarray:
        return self.base.phase

    @property
    def x_grid(self) -> np.ndarray:
        return self.base.x_grid

    @property
    def dqs_day_ta(self) -> np.ndarray | None:
        return self.base.dqs_day_ta

    @property
    def shared_feature_tensor(self) -> np.ndarray | None:
        return self.base.shared_feature_tensor

    @property
    def open_px(self) -> np.ndarray:
        return self.market_overlay.open_px

    @open_px.setter
    def open_px(self, value: Any) -> None:
        self._set_market_field("open_px", value)

    @property
    def high_px(self) -> np.ndarray:
        return self.market_overlay.high_px

    @high_px.setter
    def high_px(self, value: Any) -> None:
        self._set_market_field("high_px", value)

    @property
    def low_px(self) -> np.ndarray:
        return self.market_overlay.low_px

    @low_px.setter
    def low_px(self, value: Any) -> None:
        self._set_market_field("low_px", value)

    @property
    def close_px(self) -> np.ndarray:
        return self.market_overlay.close_px

    @close_px.setter
    def close_px(self, value: Any) -> None:
        self._set_market_field("close_px", value)

    @property
    def volume(self) -> np.ndarray:
        return self.market_overlay.volume

    @volume.setter
    def volume(self, value: Any) -> None:
        self._set_market_field("volume", value)

    @property
    def bar_valid(self) -> np.ndarray:
        return self.market_overlay.bar_valid

    @bar_valid.setter
    def bar_valid(self, value: Any) -> None:
        self._set_market_field("bar_valid", value)

    @property
    def active_t(self) -> np.ndarray:
        return self.market_overlay.active_t

    @active_t.setter
    def active_t(self, value: Any) -> None:
        self._set_market_field("active_t", value)

    @property
    def rvol(self) -> np.ndarray:
        return self.feature_overlay.rvol

    @rvol.setter
    def rvol(self, value: Any) -> None:
        self._set_feature_field("rvol", value)

    @property
    def atr_floor(self) -> np.ndarray:
        return self.feature_overlay.atr_floor

    @atr_floor.setter
    def atr_floor(self, value: Any) -> None:
        self._set_feature_field("atr_floor", value)

    @property
    def vp(self) -> np.ndarray:
        return self.feature_overlay.vp

    @vp.setter
    def vp(self, value: Any) -> None:
        self._set_feature_field("vp", value)

    @property
    def vp_delta(self) -> np.ndarray:
        return self.feature_overlay.vp_delta

    @vp_delta.setter
    def vp_delta(self, value: Any) -> None:
        self._set_feature_field("vp_delta", value)

    @property
    def profile_stats(self) -> np.ndarray:
        return self.feature_overlay.profile_stats

    @profile_stats.setter
    def profile_stats(self, value: Any) -> None:
        self._set_feature_field("profile_stats", value)

    @property
    def scores(self) -> np.ndarray:
        return self.feature_overlay.scores

    @scores.setter
    def scores(self, value: Any) -> None:
        self._set_feature_field("scores", value)

    @property
    def orders(self) -> np.ndarray:
        return self._require_scratch_field("orders")

    @orders.setter
    def orders(self, value: Any) -> None:
        self._set_scratch_field("orders", value)

    @property
    def order_side(self) -> np.ndarray:
        return self._require_scratch_field("order_side")

    @order_side.setter
    def order_side(self, value: Any) -> None:
        self._set_scratch_field("order_side", value)

    @property
    def order_flags(self) -> np.ndarray:
        return self._require_scratch_field("order_flags")

    @order_flags.setter
    def order_flags(self, value: Any) -> None:
        self._set_scratch_field("order_flags", value)

    @property
    def position_qty(self) -> np.ndarray:
        return self._require_scratch_field("position_qty")

    @position_qty.setter
    def position_qty(self, value: Any) -> None:
        self._set_scratch_field("position_qty", value)

    @property
    def overnight_mask(self) -> np.ndarray:
        return self._require_scratch_field("overnight_mask")

    @overnight_mask.setter
    def overnight_mask(self, value: Any) -> None:
        self._set_scratch_field("overnight_mask", value)

    @property
    def available_cash(self) -> np.ndarray:
        return self._require_scratch_field("available_cash")

    @available_cash.setter
    def available_cash(self, value: Any) -> None:
        self._set_scratch_field("available_cash", value)

    @property
    def equity(self) -> np.ndarray:
        return self._require_scratch_field("equity")

    @equity.setter
    def equity(self, value: Any) -> None:
        self._set_scratch_field("equity", value)

    @property
    def margin_used(self) -> np.ndarray:
        return self._require_scratch_field("margin_used")

    @margin_used.setter
    def margin_used(self, value: Any) -> None:
        self._set_scratch_field("margin_used", value)

    @property
    def buying_power(self) -> np.ndarray:
        return self._require_scratch_field("buying_power")

    @buying_power.setter
    def buying_power(self, value: Any) -> None:
        self._set_scratch_field("buying_power", value)

    @property
    def realized_pnl(self) -> np.ndarray:
        return self._require_scratch_field("realized_pnl")

    @realized_pnl.setter
    def realized_pnl(self, value: Any) -> None:
        self._set_scratch_field("realized_pnl", value)

    @property
    def unrealized_pnl(self) -> np.ndarray:
        return self._require_scratch_field("unrealized_pnl")

    @unrealized_pnl.setter
    def unrealized_pnl(self, value: Any) -> None:
        self._set_scratch_field("unrealized_pnl", value)

    @property
    def daily_loss(self) -> np.ndarray:
        return self._require_scratch_field("daily_loss")

    @daily_loss.setter
    def daily_loss(self, value: Any) -> None:
        self._set_scratch_field("daily_loss", value)

    @property
    def daily_loss_breach_flag(self) -> np.ndarray:
        return self._require_scratch_field("daily_loss_breach_flag")

    @daily_loss_breach_flag.setter
    def daily_loss_breach_flag(self, value: Any) -> None:
        self._set_scratch_field("daily_loss_breach_flag", value)

    @property
    def leverage_limit(self) -> np.ndarray:
        return self._require_scratch_field("leverage_limit")

    @leverage_limit.setter
    def leverage_limit(self, value: Any) -> None:
        self._set_scratch_field("leverage_limit", value)


def validate_candidate_execution_view(state: CombinedStateView) -> None:
    cfg = state.cfg
    t_count = int(cfg.T)
    a_count = int(cfg.A)

    def _shape(name: str, arr: np.ndarray, expected: tuple[int, ...]) -> None:
        if tuple(np.asarray(arr).shape) != tuple(expected):
            raise RuntimeError(f"{name} shape mismatch: got {np.asarray(arr).shape}, expected {expected}")

    _shape("position_qty", np.asarray(state.position_qty, dtype=np.float64), (t_count, a_count))
    _shape("equity", np.asarray(state.equity, dtype=np.float64), (t_count,))
    _shape("margin_used", np.asarray(state.margin_used, dtype=np.float64), (t_count,))
    _shape("buying_power", np.asarray(state.buying_power, dtype=np.float64), (t_count,))
    _shape("daily_loss", np.asarray(state.daily_loss, dtype=np.float64), (t_count,))
    _shape("order_side", np.asarray(state.order_side, dtype=np.int8), (t_count, a_count))
    _shape("order_flags", np.asarray(state.order_flags, dtype=np.uint16), (t_count, a_count))
    _shape(
        "daily_loss_breach_flag",
        np.asarray(state.daily_loss_breach_flag, dtype=np.int8),
        (t_count,),
    )
    for name in ("position_qty", "equity", "margin_used", "buying_power", "daily_loss"):
        arr = np.asarray(getattr(state, name), dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{name} must be finite in compact candidate execution view")
