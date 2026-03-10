# SPEC-COMPLIANCE

This patch adds a strict `engine.mode = "sealed"` kernel path while preserving legacy tunable behavior under `engine.mode = "research"`.

## Sealed-mode mappings (PDF sections)

- **§12.2 Value Area expansion**
  - Implemented `compute_value_area_greedy(...)` in Module 2.
  - Rule: expand from IPOC by choosing larger neighboring VP mass each step.
  - Boundary rule: if one side is OOB, choose in-bounds side.
  - Tie-break: smaller `|x|`, then left.

- **§3.1 Epsilon/unit system**
  - `eps_pdf=1e-12`, `eps_vol=1e-12` remain scalar.
  - `eps_div=tick`, `eps_range=tick` remain typed by tick.
  - In sealed mode, sigma floor uses **DX** (`cfg.dx`) instead of tick.

- **§4.2 ATR floor**
  - Sealed mode now uses:
    - `ATR_floor = max(ATR, 4*tick, 0.0002*close)`

- **§5 RVOL baseline**
  - Sealed mode uses causal same-ToD median over prior sessions only, `K=20`, excluding today.
  - Implemented memory-safe ring-buffer baseline builder (no `S x 1440 x A` cube in sealed path).

- **§8.1 Mean reprojection**
  - Sealed mode uses candle mid-body:
    - `m_k = (O_k + C_k)/2`
    - `mu_k→t = (m_k - C_t)/(ATR_floor_t + eps_div)`

- **§8.2.1 Range/sigma base**
  - `wRVOL = RVOL/(1+RVOL)`
  - `range_eff = wRVOL*range_k + (1-wRVOL)*ATR_floor_t`
  - `sigma_base = range_eff / (4*(ATR_floor_t + eps_div))`

- **§8.3 Sigma split**
  - `sigma1 = sigma_base/(1+log(1+RVOL_t))`
  - `sigma2 = sigma_base`
  - `sigma >= DX`

- **§8.3.2 Mixture weights**
  - `w1 = clip(body_pct, 0, 1)`, `w2 = 1-w1`
  - Means identical in sealed path (`mu1=mu2=mu_k→t`).

- **§9.2 / §9.2.1 Volume cap**
  - Sealed mode locks:
    - `W=60`, `λV=5`
    - `cap(t)=median(V)+λV*MAD(V)`
    - `cap_eff = cap*(1+ln(max(1,RVOL_t)))`

- **§10 Hybrid delta**
  - CLV:
    - `((C-L)-(H-C))/(H-L+eps_div)`
  - `r_k = (C_k - C_{k-1})/(ATR_floor_t+eps_div)`
  - `sr(t)=1.4826*MAD(r window)`, `s_eff_r=max(sr, DX/2)`
  - `k_r=ln(9)/(s_eff_r+eps_pdf)`
  - `pSRbuy=sigmoid(k_r*r)`, `pCLVbuy=sigmoid(6*CLV)`
  - `wtrend=clip(body_pct,0,1)`
  - `pbuy=wtrend*pSRbuy + (1-wtrend)*pCLVbuy`
  - `signed_blend = 2*pbuy - 1`

- **Time representation / DST**
  - Module 1 clock conversion now uses `zoneinfo` (DST-safe).
  - Module 5 ingestion builds clock arrays from UTC with zoneinfo and injects them via `clock_override`.

## Research mode differences

- Keeps existing tunable coefficients and prior behavior for:
  - sigma models (`sigma1_base`, `sigma1_body_coeff`, etc.)
  - weight model (`w1_base`, etc.)
  - RVOL lookback and cap tuning from `Module2Config`.
- Retains legacy flexibility for experimentation and ablation studies.
