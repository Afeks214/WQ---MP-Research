from risk_engine import CostConfig, _trade_costs


def test_short_entry_locate_fee_applied() -> None:
    cfg = CostConfig()
    shares = 100
    c = _trade_costs(shares=shares, side=-1, is_short_entry=True, cost_cfg=cfg)
    expected = (
        shares * cfg.commission_per_share
        + shares * cfg.reg_fee_per_share_sell
        + shares * cfg.locate_fee_per_share_short_entry
    )
    assert abs(c - expected) < 1e-12
