# Market Mechanics MVP — Research Protocol v1

Status: experimental, non-execution research  
Primary instrument: SPY  
Primary data substrate: Beta-spy point-in-time minute bars + SPY minute flow  
Primary objective: attempt to falsify the effective-inertia hypothesis before any Delta integration

## 1. Research question

Does a causal estimate of SPY's price-response resistance to directional market pressure contain
incremental information about subsequent SPY behavior beyond the raw motion and flow variables from
which the estimate is built?

The MVP treats "inertia" as an empirical inverse response coefficient. It is not physical mass.

## 2. Pre-registered MVP equation

For minute t:

```text
x_t = log(SPY close_t)
v_t = 10,000 * (x_t - x_{t-1})
a_t = v_t - v_{t-1}
```

The pressure variable is a fixed, bounded composite:

```text
F_t =
    0.60 * aggressor order-flow imbalance
  + 0.25 * top-of-book quote imbalance
  + 0.15 * liquidity-support asymmetry
```

Each component is clipped to [-1, +1]. The MVP weights are not fitted.

The causal response model is:

```text
a_t =
    alpha
  + beta_up   * max(F_{t-1}, 0)
  - beta_down * max(-F_{t-1}, 0)
  - gamma     * v_{t-1}
  + error_t
```

The model is estimated on a trailing window with ridge regularization.

When the response coefficient has the expected sign:

```text
M_up   = 1 / beta_up
M_down = 1 / beta_down
```

If a fitted response coefficient is non-positive, inertia for that direction is reported as unavailable
instead of taking an absolute value.

Directional inertial bias is:

```text
IB = (M_down - M_up) / (M_down + M_up)
```

Mechanical momentum in the MVP is:

```text
p_t = M_direction * v_t
```

where direction is selected from the sign of current velocity.

Impulse is a causal exponentially decayed pressure accumulation:

```text
J_t = decay * J_{t-1} + F_t
```

## 3. What is deliberately excluded

The first test does not include:

- options flow or dealer gamma,
- ES futures pressure,
- S&P constituent pressure,
- macro/news data,
- four-quadrant launch/braking inertia,
- equilibrium/restoring force,
- learned force weights,
- deep learning,
- broker orders,
- trade sizing,
- P&L optimization.

Those are prohibited until the simpler hypothesis survives.

## 4. Data requirements

Minimum acceptable research substrate:

1. SPY one-minute close observations.
2. SPY minute-level aggressor flow derived causally from tape/quote information.
3. SPY top-of-book quote imbalance when available.
4. Replenishment/withdrawal features when available.
5. Strict timestamp ordering.
6. No repaired value may use data from a later timestamp.

Bars without flow may remain in the motion series but will contribute zero observed force. Coverage must
therefore be reported explicitly.

## 5. Primary prediction target

Initial horizon:

```text
future_return_5m = log(P_{t+5} / P_t)
future_up_5m = 1[future_return_5m > 0]
```

Additional horizons may be evaluated after the 5-minute protocol is frozen.

Labels are permitted only in offline evaluation. They never enter the live state estimator.

## 6. Baseline model

The baseline receives:

- velocity,
- acceleration,
- composite force,
- order-flow component,
- quote-imbalance component.

This is intentionally difficult to beat because it contains the raw ingredients used to create inertia.

## 7. Augmented mechanics model

The augmented model receives the baseline variables plus:

- upside inertia,
- downside inertia,
- inertial bias,
- mechanical momentum,
- impulse.

A gain by the augmented model is therefore evidence that the dynamic response transformation adds
information beyond its ingredients.

## 8. Walk-forward protocol

Random train/test splitting is prohibited.

Use expanding time-series folds with a purge gap equal to at least the prediction horizon.

Default MVP:

- 5 sequential folds,
- 5-minute prediction horizon,
- purge gap >= 5 minutes,
- logistic direction model,
- feature standardization fitted only on training observations.

The test fold must always occur after its training fold.

## 9. Primary score

Primary score:

```text
out-of-sample log loss
```

Secondary score:

```text
Brier score
```

Accuracy is diagnostic only and is not a promotion criterion.

For each fold calculate:

```text
relative improvement =
    (baseline log loss - mechanics log loss)
    / baseline log loss
```

## 10. Candidate promotion criterion

The MVP is only a candidate for deeper study when all of the following hold:

1. median relative log-loss improvement >= 1%;
2. augmented mechanics improves log loss in at least 4 of 5 valid folds;
3. median Brier-score improvement is positive;
4. the result is not confined to one response-estimation window;
5. there is sufficient model-ready coverage to make the result operationally meaningful.

The repository runner declares `robust_candidate=true` only when at least two tested windows meet the
first three statistical conditions.

This is not authorization for trading or production deployment.

## 11. Estimability kill test

Measure the fraction of post-warmup observations for which both directional response coefficients have
the expected sign and can produce finite inertia estimates.

Failure conditions:

- valid two-sided inertia is rare,
- directional response signs flip chaotically,
- response coefficients are dominated by numerical regularization.

A construct that cannot be estimated reliably should not be promoted.

## 12. Persistence kill test

For model-ready observations calculate lag-1 rank persistence separately for:

- upside inertia,
- downside inertia.

If inertia is almost white noise, the interpretation of a market "state" is weakened.

Persistence alone is not sufficient for success because a stable but useless state is still useless.

## 13. Raw-feature substitution kill test

The baseline already includes the raw motion/flow ingredients.

If the augmented mechanics model does not improve out-of-sample probability quality, the default
interpretation is:

> effective inertia is a repackaging of information already present in velocity and order flow.

Do not add more mechanics variables to rescue the model before diagnosing this failure.

## 14. Window robustness kill test

Default response windows:

- 60 minutes,
- 120 minutes,
- 240 minutes.

A useful effect should not depend on one arbitrary window.

If only one window succeeds, label the result unstable and investigate before promotion.

## 15. Directional-asymmetry kill test

Compare the two-sided model with a restricted symmetric response model:

```text
beta_up = beta_down
```

This comparison is a Phase II requirement.

Directional inertia is retained only if the unrestricted model improves out-of-sample likelihood or
materially improves the downstream target.

## 16. Momentum kill test

Mechanical momentum:

```text
p = M * v
```

must be compared with velocity alone and with equivalently smoothed velocity.

If p adds no predictive information, remove mechanical momentum even if inertia itself survives.

## 17. Impulse kill test

Impulse must be compared with:

- raw current force,
- rolling average force,
- cumulative OFI.

If exponentially decayed impulse offers no incremental information, remove it.

The stronger "unexpressed impulse" construct is not admitted into the MVP.

## 18. Liquidity-substitution kill test

In Phase II, compare inferred inertia against directly observed:

- spread,
- top-of-book size,
- depth where available,
- price-impact proxy,
- absorption proxy.

If direct liquidity variables fully explain the effect, keep the simpler variables unless the mechanics
representation provides demonstrable interpretability value.

## 19. Smoothing-artifact kill test

The mechanics estimator introduces a trailing regression window.

Therefore every result must be compared against baseline variables smoothed over equivalent windows.

If the advantage disappears, the result came from temporal smoothing rather than inertia.

## 20. Reverse-causality test

The operational estimator uses lagged force to explain current acceleration.

Additional research must compare:

```text
F_{t-1} -> a_t
```

with:

```text
a_{t-1} -> F_t
```

If the reverse relationship is equally strong, causal language must be removed. The model may still be
useful predictively.

## 21. Intraday-seasonality test

Repeat evaluation with:

- raw variables,
- time-of-day normalized variables,
- opening 30 minutes excluded,
- closing 30 minutes excluded.

A result driven only by the open/close must be labeled as such.

## 22. Volatility-regime robustness

Stratify results by realized-volatility regime.

At minimum:

- bottom volatility tercile,
- middle tercile,
- top tercile.

Report whether mechanics is useful broadly or only in specific volatility states.

## 23. Session independence

Minutes are not independent observations.

Report results by session in addition to pooled minute statistics.

A large pooled sample dominated by a handful of abnormal sessions is not sufficient evidence.

## 24. No execution optimization

During the entire MVP research phase, the following are forbidden research targets:

- option strike selection,
- stop distance,
- profit target,
- leverage,
- trade sizing,
- P&L-maximizing threshold.

The mechanics layer is being tested as a market-state representation, not tuned into a strategy.

## 25. Required promotion sequence

```text
MVP estimator
    ->
synthetic causal/unit tests
    ->
historical SPY walk-forward kill tests
    ->
robustness tests
    ->
directional asymmetry validation
    ->
braking/launch inertia research
    ->
multi-horizon mechanics
    ->
Delta-readable state schema
```

Skipping directly to Delta is prohibited.

## 26. Failure policy

If the MVP fails, simplify.

Do not rescue it by:

- adding dozens of indicators,
- changing targets repeatedly,
- searching many window combinations,
- introducing options/futures data after seeing failure,
- optimizing thresholds on the test set.

Every new hypothesis requires a new frozen protocol and new untouched test data.

## 27. Current repository implementation

`beta_spy.mechanics.MechanicsEstimator` implements the frozen MVP.

`scripts/market_mechanics_research.py` runs the first walk-forward kill test against an existing
Beta-spy SQLite database.

Default research command:

```bash
python scripts/market_mechanics_research.py \
  --database /path/to/beta-spy.sqlite \
  --horizon 5 \
  --windows 60 120 240 \
  --output market-mechanics-report.json
```

Interpret `robust_candidate=true` only as permission to continue the research program.

## 28. Decision rule

The research program asks one question before any further expansion:

> Does inferred response resistance contain reproducible out-of-sample information that the underlying
> flow and motion variables do not already contain?

If no, stop.

If yes, proceed to Phase II and test whether the surviving effect is best understood as directional
inertia, braking inertia, liquidity response, or another microstructure construct.
