# Sweep Math Spec (Code-Derived, No Invented Logic)

This spec is derived from current repository code only.
All anchors below refer to concrete functions in:
- `weightiz_module1_core.py`
- `weightiz_module2_core.py`
- `weightiz_module4_strategy_funnel.py`
- `weightiz_module5_harness.py`

## 1) Session/Gap/Warmup State Machine

### 1.1 Gap minutes and reset flag
Anchor: `weightiz_module1_core.py::build_session_clock_vectorized`

$$
\mathrm{gap\_min}_t = \begin{cases}
0, & t=0\\
\dfrac{ts\_ns[t]-ts\_ns[t-1]}{\mathrm{NS\_PER\_MIN}}, & t>0
\end{cases}
$$

$$
\mathrm{reset\_flag}_t = \mathbf{1}\left[(\mathrm{gap\_min}_t > \mathrm{gap\_reset\_minutes}) \lor \mathrm{session\_change}_t\right], \quad \mathrm{reset\_flag}_0=1
$$

### 1.2 Phase assignment
Anchor: `weightiz_module1_core.py::_compute_phase`

Let $tod_t = \mathrm{minute\_of\_day}_t - \mathrm{rth\_open\_minute}$.

- WARMUP by default
- LIVE if $tod_t \ge \mathrm{warmup\_minutes}$ and minute-of-day $< \mathrm{flat\_time\_minute}$
- OVERNIGHT_SELECT if minute-of-day equals `flat_time_minute` and warmup complete
- FLATTEN if minute-of-day `> flat_time_minute`

### 1.3 Warmup neutralization
Anchor: `weightiz_module2_core.py` block near `warmup_rows`

$$
\mathrm{scores}[\mathrm{phase}=\mathrm{WARMUP}] \leftarrow 0
$$

Internal state updates still run; output scores are neutralized during warmup rows.

## 2) Module 2 (Value Area / Profile)

### 2.1 Value area expansion
Anchor: `weightiz_module2_core.py::compute_value_area_greedy`

For each asset row $a$:

$$
\mathrm{target}_a = \mathrm{va\_threshold} \cdot \sum_b vp_{a,b}
$$

Initialize at POC index $ipoc_a$, then grow left/right one bin at a time by higher adjacent mass. Ties are broken deterministically by:
1. smaller $|x|$,
2. then left.

### 2.2 Delta and blend terms (as coded)
Anchor: `weightiz_module2_core.py` near `delta0`, `delta_poc`, `delta_eff`

$$
\delta0_a = \frac{vpd_{a,idx0}}{vp_{a,idx0} + \epsilon_{vol}}, \quad
\delta^{poc}_a = \frac{vpd_{a,ipoc_a}}{vp_{a,ipoc_a} + \epsilon_{vol}}
$$

$$
\delta^{eff}_a = w^{poc}_a \cdot \delta^{poc}_a + (1-w^{poc}_a)\cdot \delta0_a
$$

## 3) Module 4 (Signals/Gates/Sizing)

### 3.1 Regime rules
Anchor: `weightiz_module4_strategy_funnel.py` trend/neutral block

Trend-up condition:
$$
\mathrm{trend\_up} = \mathrm{tradable} \land (tgs \ge \theta_{spread}) \land (poc\_drift \ge \theta_{trend}) \land (poc\_vs\_prev\_va > 1)
$$

Trend-down is symmetric with negative drift and $poc\_vs\_prev\_va < -1$.

Neutral condition:
$$
\mathrm{neutral} = \mathrm{tradable} \land (|poc\_drift| \le \theta_{neutral}) \land (|poc\_vs\_prev\_va| \le 1) \land (valid\_ratio \ge 0.70)
$$

with:
- $\theta_{trend}=\texttt{trend\_poc\_drift\_min\_abs}$
- $\theta_{neutral}=\texttt{neutral\_poc\_drift\_max\_abs}$

### 3.2 Entry gating
Anchor: `weightiz_module4_strategy_funnel.py` intent block

Example long breakout gate (as coded):
$$
\mathrm{intent\_bo\_long} = \mathrm{tradable} \land (\mathrm{trend\_up} \lor \mathrm{p\_shape} \lor \mathrm{double\_up}) \land (bo\_l^{eff} > \theta_{entry}) \land (g_{break}>0.5)
$$

where $\theta_{entry}=\texttt{entry\_threshold}$.

### 3.3 Intraday top-k sizing
Anchor: `weightiz_module4_strategy_funnel.py` Top-K block

Top-k selection:
$$
k = \min\left(\max(\texttt{top\_k\_intraday},0),\#\{candidates\}\right)
$$

Inverse-ATR style weighting (current code):
$$
atr_{sel} = \max(atr_{eff}, atr_{floor}),\quad w_{raw}=\frac{1}{atr_{sel}},\quad w=\min(w_{raw}, w_{cap})
$$
then normalized by $\sum w$ when finite and positive.

Notional allocation / quantity:
$$
alloc = \min(gross\_budget\cdot w,\; equity\cdot max\_asset\_cap\_frac)
$$
$$
qty = sign \cdot \frac{alloc}{px + \epsilon}
$$
with turnover cap applied by scaling $\Delta qty$ when total delta notional exceeds `max_turnover_frac_per_bar * equity`.

### 3.4 Clipping form used in code
Multiple places use `np.clip`, equivalent to:
$$
X_{clipped} = \max(\min(X, U), L)
$$

## 4) Metrics (as implemented)
Anchor: `weightiz_module5_harness.py` candidate metrics block

Given baseline return series `ret_series`:

$$
\mathrm{win\_rate} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[r_i>0]
$$

$$
\mathrm{avg\_trade} = \mathrm{mean}(\{r_i: |r_i|>10^{-15}\})
$$

$$
\mathrm{profit\_factor} = \frac{\sum_{r_i>0} r_i}{\max\left(|\sum_{r_i<0} r_i|, 10^{-12}\right)}
$$

`max_drawdown` is computed by `_max_drawdown_from_returns(...)`.

Robustness score uses code-declared formula string in `candidate_metrics["robustness"]["formula"]` and `ROBUSTNESS_CAPS`.

## 5) Epsilons / Constants (code anchors)
- Engine: `run_research.py::EngineConfigModel`
  - `eps_pdf=1e-12`, `eps_vol=1e-12`
- Module 3: `run_research.py::Module3ConfigModel.eps=1e-12`
- Module 4: `run_research.py::Module4ConfigModel.eps=1e-12`
- Profit factor denominator floor: `weightiz_module5_harness.py` uses `max(abs(neg_sum), 1e-12)`

## 6) Sanity Checklist (Guard Rails)

| Failure mode | Guard in code | Anchor |
|---|---|---|
| Non-monotonic timestamps | hard validation + exceptions | `weightiz_module1_core.py` clock builders |
| Invalid reset flag values | binary + first-row assertions | `weightiz_module1_core.py` validation blocks |
| Non-finite Module2 outputs | fail on non-finite output branch | `weightiz_module2_core.py` Stage F checks |
| Non-finite Module4 outputs | `_assert_finite_masked(...)` at output validation | `weightiz_module4_strategy_funnel.py` |
| Unknown YAML fields | `extra="forbid"` models | `run_research.py` Pydantic models |
| Family worker oversubscription | fatal error if workers > 14 | `run_research.py::main` |
| Family result non-finite fields | strict integrity check before parquet write | `run_research.py::_assert_results_integrity` |

## 7) NOT FOUND IN CODE
- A closed-form single-equation strategy objective for the full funnel is **NOT FOUND IN CODE** (logic is procedural across Module 2 + Module 4).
- A standalone symbolic formula for `_max_drawdown_from_returns` beyond its implementation function body is **NOT FOUND IN CODE**.
