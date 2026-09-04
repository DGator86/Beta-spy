# Market Mechanics V2 — Four-Quadrant Launch / Braking Protocol

Status: frozen experimental research protocol  
Parent baseline: `MARKET_MECHANICS_MVP_V1` — rejected, preserved unchanged  
Primary instrument: SPY  
Primary objective: test whether resistance to **opposing** pressure predicts trend persistence

## 1. Why V2 exists

V1 asked whether bullish and bearish pressure had different unconditional acceleration-response
coefficients. On 11 archived SPY sessions, the resulting inverse-response variables failed the frozen
walk-forward test and degraded out-of-sample probability quality.

V2 is a new hypothesis. V1 is not modified, retuned, or reinterpreted.

The V2 question is narrower:

> Once SPY is already moving, does the market's measured resistance to opposing pressure contain
> incremental information about whether that motion persists?

This separates launch/continuation response from braking response.

## 2. Frozen force

V2 reuses the V1 force **without any weight change**:

```text
F_t =
    0.60 * aggressor order-flow imbalance
  + 0.25 * top-of-book quote imbalance
  + 0.15 * liquidity-support asymmetry
```

Each component remains clipped to `[-1,+1]` by the existing V1 implementation.

Changing force weights after seeing V2 results is prohibited. A different force definition requires a
new protocol/version.

## 3. Motion variables

```text
x_t = log(SPY close_t)
v_t = 10,000 * (x_t - x_{t-1})
a_t = v_t - v_{t-1}
```

Only regular one-minute transitions create acceleration-response rows. Session boundaries and material
timestamp gaps reset/skip response construction exactly as in V1.

## 4. Four causal response quadrants

Each response row uses lagged force and lagged velocity to explain current acceleration:

```text
a_t = alpha_q + beta_q * F_{t-1} - gamma_q * v_{t-1} + error_t
```

The quadrant is selected from the sign of `v_{t-1}` and `F_{t-1}`:

```text
beta_pp: v >= 0, F > 0   uptrend launch / continuation response
beta_pm: v >= 0, F < 0   uptrend braking response
beta_mp: v <  0, F > 0   downtrend braking response
beta_mm: v <  0, F < 0   downtrend launch / continuation response
```

Force remains signed in all regressions. Therefore the V2 hypothesis expects a valid `beta_q` to be
positive in all four quadrants. Negative or near-zero fitted response is reported unavailable.

No absolute value, coefficient sign flip, clipping-to-positive, or post-hoc relabeling is allowed.

## 5. V2 inertia definitions

```text
M_launch_up   = 1 / beta_pp
M_brake_up    = 1 / beta_pm
M_brake_down  = 1 / beta_mp
M_launch_down = 1 / beta_mm
```

For the current direction of motion:

```text
if v_t >= 0:
    active_launch_inertia  = M_launch_up
    active_braking_inertia = M_brake_up
else:
    active_launch_inertia  = M_launch_down
    active_braking_inertia = M_brake_down
```

Diagnostic ratio:

```text
brake_launch_ratio = active_braking_inertia / active_launch_inertia
```

The ratio is not required for the primary hypothesis.

## 6. Frozen estimator settings

Default response windows remain:

- 60 minutes
- 120 minutes
- 240 minutes

Default V2 minimum observations per quadrant:

```text
min_quadrant_samples = 12
```

Default ridge penalty:

```text
ridge = 0.25
```

A quadrant with fewer samples or effectively constant force is unavailable rather than imputed.

## 7. Primary V2 target — trend persistence

The primary five-minute target is conditional on the current direction of motion:

```text
future_return_5m = log(P_{t+5} / P_t)
current_direction = sign(v_t)
continuation_5m = 1[future_return_5m * current_direction > 0]
```

Rows with undefined current velocity or missing exact `t+5m` price are excluded.

No row-offset labels are permitted; the exact future timestamp must exist.

## 8. Primary baseline

The baseline already receives the raw information that could explain persistence:

```text
abs_velocity_bps
aligned_acceleration_bps
aligned_force
opposing_force_magnitude
aligned_ofi
aligned_quote_imbalance
```

where `aligned_*` multiplies the signed variable by current motion direction, so positive values support
current motion and negative values oppose it.

This is deliberately difficult to beat.

## 9. Primary V2 augmented model

The primary V2 model adds **one variable only**:

```text
active_braking_inertia
```

The primary hypothesis passes only if this transformation adds incremental out-of-sample information
beyond the raw motion/pressure baseline.

A secondary diagnostic model may additionally include:

```text
active_launch_inertia
brake_launch_ratio
```

Secondary diagnostics cannot rescue failure of the primary braking-only test.

## 10. Walk-forward design

Use the same strict deployment-like structure as V1:

- five sequential time-series folds;
- five-minute purge gap or greater;
- logistic probability model;
- feature scaling fitted only on the training fold;
- test observations always occur after training observations.

Random splitting is prohibited.

## 11. Primary scores

Primary:

```text
out-of-sample log loss
```

Secondary:

```text
Brier score
```

For each fold:

```text
relative improvement =
    (baseline log loss - V2 log loss)
    / baseline log loss
```

## 12. Candidate criterion

A response window is a V2 candidate only if all hold:

1. at least 300 exact-label rows have `active_braking_inertia` available;
2. median relative log-loss improvement >= +1%;
3. V2 improves log loss in at least 4 of 5 folds;
4. median Brier improvement is positive.

V2 is a robust research candidate only if at least **two** of the frozen 60/120/240-minute windows pass.

This authorizes more research only. It does not authorize Delta integration or trading.

## 13. Coverage / estimability diagnostics

For every window report:

- `beta_pp`, `beta_pm`, `beta_mp`, `beta_mm` valid fractions;
- uptrend braking inertia coverage;
- downtrend braking inertia coverage;
- active braking inertia coverage;
- full four-quadrant coverage;
- quadrant sample-count distributions.

The primary test requires only the braking inertia relevant to current motion; it does **not** require both
braking directions to be simultaneously estimable.

This avoids repeating V1's unnecessary two-sided coverage bottleneck while remaining faithful to the
trend-persistence hypothesis.

## 14. Direct persistence diagnostic

As a non-promotional diagnostic, divide model-ready observations into braking-inertia terciles within each
response window and report five-minute continuation rates.

The V2 hypothesis expects, directionally:

```text
continuation_rate(high braking inertia)
    > continuation_rate(low braking inertia)
```

This diagnostic does not replace the walk-forward incremental-prediction criterion.

## 15. Launch-vs-brake diagnostic

Test whether braking and launch response are empirically distinguishable.

If `beta_pm` and `beta_pp` (or `beta_mp` and `beta_mm`) are effectively interchangeable out of sample,
then the launch/braking distinction has not earned its complexity.

## 16. Raw-force substitution kill test

The baseline contains aligned raw force and opposing-force magnitude.

If `active_braking_inertia` does not improve the primary probability model, the default conclusion is:

> Braking inertia is a repackaging of current motion and opposing flow pressure.

Do not add impulse, options, futures, macro, or learned force weights to rescue V2.

## 17. Smoothing artifact kill test

Any positive V2 result must be challenged against equivalently windowed/smoothed baseline flow and motion
variables. If the advantage disappears, attribute the result to smoothing rather than inertia.

## 18. Reverse-causality diagnostic

Compare the forward response relationship:

```text
F_(t-1) -> a_t
```

against:

```text
a_(t-1) -> F_t
```

If reverse predictability is comparable, remove causal language even if the state remains predictively
useful.

## 19. Session and regime robustness

Report results by session and, if sample size permits, by realized-volatility tercile and time of day.

A pooled win dominated by one session or one opening/closing regime is not robust evidence.

## 20. Explicitly prohibited V2 rescue actions

Do not:

- alter the V1 force weights;
- take `abs(beta)`;
- flip negative coefficient signs;
- change the 5-minute primary target after seeing results;
- search many response windows beyond 60/120/240 and report the best;
- tune `min_quadrant_samples` on the evaluation set;
- add options, ES, constituents, macro, or news after a failed result;
- optimize a trading rule, P&L, threshold, or position size.

## 21. Decision rule

V1 is already rejected.

V2 asks one new question:

> Does active braking inertia improve causal five-minute trend-persistence inference beyond the raw
> velocity, acceleration, and pressure variables used to estimate it?

If no, reject V2 and preserve the result.

If yes across at least two frozen windows, proceed to deeper braking-inertia robustness and only then
consider multi-horizon Market Mechanics research.
