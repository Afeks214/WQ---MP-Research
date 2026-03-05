import pytest

from risk_engine import CostConfig, REASON_COST_MODEL_VIOLATION, _trade_costs


def test_commission_is_share_based() -> None:
    cfg = CostConfig()
    shares = 123
    c = _trade_costs(shares=shares, side=1, is_short_entry=False, cost_cfg=cfg)
    assert abs(c - (shares * cfg.commission_per_share)) < 1e-12


def test_negative_shares_fail_closed() -> None:
    cfg = CostConfig()
    with pytest.raises(RuntimeError, match=REASON_COST_MODEL_VIOLATION):
        _trade_costs(shares=-1, side=1, is_short_entry=False, cost_cfg=cfg)
