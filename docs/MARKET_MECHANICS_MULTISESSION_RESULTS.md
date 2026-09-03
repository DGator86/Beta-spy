# Market Mechanics V1 — Multi-Session Kill-Test Result

Evaluation date: 2026-09-03  
Frozen model: `MARKET_MECHANICS_MVP_V1`  
Data: Beta-spy Alpha-Beta daily session tapes, 2026-08-18 through 2026-09-01 trading sessions  
Status: **V1 REJECTED — DO NOT PROMOTE TO DELTA**

## Executive result

The strict V1 effective-inertia construction failed its pre-registered incremental-prediction test.

Across 11 archived trading sessions, the raw tape supplied 3,479 usable SPY price/flow minutes after
excluding materially partial capture-start minutes. The model produced enough exact 5-minute labels and
two-sided inertia states to run the five-fold walk-forward test for every frozen response window.

Adding V1 mechanics variables to the raw motion/flow baseline made out-of-sample probability quality
worse at all three windows.

| Response window | Exact-label mechanics rows | Median log-loss improvement | Folds improved | Median Brier improvement | Candidate |
|---:|---:|---:|---:|---:|---|
| 60 min | 644 | **-3.33%** | **0 / 5** | **-0.00908** | FAIL |
| 120 min | 540 | **-5.70%** | **1 / 5** | **-0.01965** | FAIL |
| 240 min | 460 | **-8.54%** | **1 / 5** | **-0.02943** | FAIL |

The frozen promotion requirement was a median log-loss improvement of at least +1%, improvement in at
least 4 of 5 folds, positive median Brier improvement, and robustness across more than one response
window. V1 met none of those requirements.

## 1. Session-tape coverage

The connected Alpha-Beta session-tape archive contains daily `tar.zst` packages with:

- `minute_bars.csv`,
- `spy_trades.csv`,
- `spy_quotes.csv`,
- decisions and other contemporaneous artifacts.

The test used the following raw-tape segments:

| Session | SPY bars | Usable flow minutes | Raw capture start | Start minute excluded? |
|---|---:|---:|---|---|
| 2026-08-18 | 390 | 283 | 15:14:34 UTC | Yes |
| 2026-08-19 | 390 | 390 | 13:30:00 UTC | No |
| 2026-08-20 | 390 | 120 | 17:59:51 UTC | Yes |
| 2026-08-21 | 390 | 360 | 13:59:55 UTC | Yes |
| 2026-08-24 | 390 | 161 | 17:18:33 UTC | Yes |
| 2026-08-25 | 390 | 390 | 13:30:00 UTC | No |
| 2026-08-26 | 390 | 215 | 16:24:20 UTC | Yes |
| 2026-08-27 | 390 | 390 | 13:30:00 UTC | No |
| 2026-08-28 | 390 | 390 | 13:30:00 UTC | No |
| 2026-08-31 | 390 | 390 | 13:30:00 UTC | No |
| 2026-09-01 | 390 | 390 | 13:30:00 UTC | No |

A capture-start minute more than five seconds into the minute was excluded rather than treated as a
complete flow minute. Missing early tape was not replaced with zeros. The mechanics estimator also
refused to create response observations across irregular minute gaps.

## 2. Frozen V1 force

No force weights were changed after seeing the data.

```text
F_t =
    0.60 * order-flow imbalance
  + 0.25 * top-of-book quote imbalance
  + 0.15 * liquidity-support asymmetry
```

Minute flow was reconstructed from the archived raw SPY prints and quotes using the same causal
aggressor and top-of-book concepts implemented by Beta-spy's `FlowAccumulator`.

## 3. Frozen V1 response model

For each session independently:

```text
a_t =
    alpha
  + beta_up   * max(F_{t-1}, 0)
  - beta_down * max(-F_{t-1}, 0)
  - gamma     * v_{t-1}
  + error_t
```

Only positive fitted directional response coefficients were admitted:

```text
M_up   = 1 / beta_up
M_down = 1 / beta_down
```

No absolute-value rescue, sign flip, coefficient clipping, or data-dependent force reweighting was
performed.

## 4. Estimability

After the 30-response warmup, 3,137 observations were eligible for coefficient diagnostics.

| Window | Valid upside | Valid downside | Both valid | Two-sided coverage |
|---:|---:|---:|---:|---:|
| 60 | 1,557 | 1,350 | 654 | 20.85% |
| 120 | 1,517 | 1,367 | 550 | 17.53% |
| 240 | 1,292 | 1,622 | 462 | 14.73% |

The multi-session sample confirms the initial concern: the strict directional inverse-response state is
available only intermittently. More importantly, there were nevertheless enough model-ready rows to run
the primary walk-forward kill test, so the V1 rejection is not merely a small-sample artifact.

## 5. Simple inertial-bias relationship

Spearman correlation between V1 inertial bias and exact future 5-minute SPY return on model-ready rows:

```text
60-minute window:  -0.0828
120-minute window: -0.1117
240-minute window: -0.1322
```

The fact that all three values are negative is not promoted as a new signal. The pre-registered test
allowed the downstream logistic model to learn either sign; the augmented model still failed to improve
out of sample.

## 6. Walk-forward design

Baseline features:

```text
velocity
acceleration
composite force
OFI force component
quote-imbalance force component
```

Augmented model adds:

```text
upside inertia
downside inertia
inertial bias
mechanical momentum
exponentially decayed impulse
```

Both models use the same model-ready rows. Feature standardization is fit only on each training fold.
The probability model is regularized logistic regression. Five sequential `TimeSeriesSplit` folds are
used with a five-observation purge gap for the five-minute target.

Primary metric: out-of-sample log loss.  
Secondary metric: Brier score.

## 7. 60-minute window folds

| Fold | Baseline log loss | Mechanics log loss | Relative improvement | Baseline Brier | Mechanics Brier |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7671 | 1.0881 | -41.85% | 0.2826 | 0.2691 |
| 2 | 0.6835 | 0.7044 | -3.06% | 0.2457 | 0.2548 |
| 3 | 0.7426 | 0.7994 | -7.66% | 0.2734 | 0.2855 |
| 4 | 0.7025 | 0.7175 | -2.14% | 0.2544 | 0.2603 |
| 5 | 0.6730 | 0.6954 | -3.33% | 0.2400 | 0.2496 |

Median relative log-loss improvement: **-3.33%**.  
Positive log-loss folds: **0 / 5**.  
Median Brier improvement: **-0.00908**.

Result: **FAIL**.

## 8. 120-minute window folds

| Fold | Baseline log loss | Mechanics log loss | Relative improvement | Baseline Brier | Mechanics Brier |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7126 | 1.4258 | -100.07% | 0.2592 | 0.4200 |
| 2 | 0.8392 | 1.1057 | -31.76% | 0.3172 | 0.3949 |
| 3 | 0.7593 | 0.7582 | +0.15% | 0.2825 | 0.2815 |
| 4 | 0.6796 | 0.6830 | -0.50% | 0.2433 | 0.2453 |
| 5 | 0.7018 | 0.7418 | -5.70% | 0.2540 | 0.2736 |

Median relative log-loss improvement: **-5.70%**.  
Positive log-loss folds: **1 / 5**.  
Median Brier improvement: **-0.01965**.

Result: **FAIL**.

## 9. 240-minute window folds

| Fold | Baseline log loss | Mechanics log loss | Relative improvement | Baseline Brier | Mechanics Brier |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9790 | 1.2600 | -28.70% | 0.3501 | 0.4286 |
| 2 | 0.7891 | 1.8784 | -138.05% | 0.2921 | 0.4573 |
| 3 | 0.8070 | 0.8304 | -2.90% | 0.3048 | 0.3112 |
| 4 | 0.7392 | 0.7286 | +1.44% | 0.2725 | 0.2667 |
| 5 | 0.6979 | 0.7574 | -8.54% | 0.2522 | 0.2816 |

Median relative log-loss improvement: **-8.54%**.  
Positive log-loss folds: **1 / 5**.  
Median Brier improvement: **-0.02943**.

Result: **FAIL**.

## 10. Interpretation

The tested proposition was not merely that flow relates to price. Beta already contains that information.
The stronger proposition was:

> transforming lagged flow response into inverse directional acceleration coefficients, and then deriving
> inertia, momentum, bias, and impulse, adds information beyond the underlying motion and flow variables.

On this sample, that proposition is rejected for V1.

The failure has two components:

1. **state availability:** the sign-constrained two-sided inverse-response state exists only about 15–21%
   of post-warmup observations;
2. **incremental prediction:** when it does exist, adding the V1 mechanics state degrades rather than
   improves five-minute probability forecasts.

The raw pressure-response idea is not disproven by this test. The specific V1 mapping

```text
lagged directional force -> acceleration -> inverse coefficient -> M -> p=Mv
```

is not supported as an incremental predictive representation.

## 11. Scientific decision

`MARKET_MECHANICS_MVP_V1` is now **REJECTED**.

That means:

- do not merge V1 into Delta as a signal source;
- do not optimize V1 force weights on these same sessions;
- do not take absolute values of failed response coefficients;
- do not flip inertial-bias sign because the pooled correlation is negative;
- do not cherry-pick the one positive 240-minute fold;
- do not market the one-day August 17 result as evidence of success.

The research branch remains useful because it contains the falsification machinery and preserves the
negative result.

## 12. What remains alive

Several hypotheses from the original Market Mechanics white paper were **not** tested by V1 and therefore
remain open as separate hypotheses, including:

- braking inertia versus launch inertia;
- four-quadrant response conditioned on the sign of existing velocity and applied force;
- absorption/replenishment as an explicit resistance state rather than inferred inverse acceleration;
- damping / momentum half-life;
- pressure-response models at sub-minute horizons;
- state-space or constrained response estimation;
- cross-market force from ES and the constituent basket.

They must not be treated as repairs to V1. Any next model requires its own frozen protocol and fresh or
explicitly designated holdout data.

## 13. Current promotion status

```text
V1 synthetic recovery tests:                PASS
V1 causal/gap/session engineering tests:    PASS
V1 real-tape estimability:                   WEAK
V1 5-fold incremental prediction:           FAIL
V1 robustness across windows:               FAIL
V1 robust_candidate:                        FALSE
V1 Delta integration:                       BLOCKED
V1 execution integration:                   PROHIBITED
```

The kill test did what it was supposed to do: it prevented an attractive analogy from becoming a
production signal without evidence.
