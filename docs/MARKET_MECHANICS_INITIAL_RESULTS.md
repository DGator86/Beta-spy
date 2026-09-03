# Market Mechanics MVP — Initial Real-Tape Result

Date evaluated: 2026-09-03  
Research data snapshot: latest accessible Beta-spy database backup, modified 2026-08-17  
Status: **DO NOT PROMOTE TO DELTA**

## 1. Purpose

This note records the first attempt to falsify the Market Mechanics MVP against Beta-spy's own stored
SPY data. It is intentionally preserved before any model redesign so later research cannot quietly
rewrite the initial result.

## 2. Data provenance

The Beta backup manifest identifies the runtime database as:

```text
/var/lib/beta-spy/beta-spy.sqlite
```

The accessible latest compressed snapshot was `beta-spy.sqlite.zst`, approximately 141.7 MB compressed
and approximately 523.7 MB after decompression.

Database inventory relevant to the mechanics test:

```text
SPY minute bars: 6,571 total
SPY minute-bar span: 2026-07-27 through 2026-08-17
stored SPY minute_flow rows: 0
raw SPY trades: 214,917
raw SPY quotes: 72,556
```

The raw SPY trade tape in this snapshot is concentrated on 2026-08-17, so the first genuine force test
has only one complete-enough RTH tape session.

For 2026-08-17 regular trading hours the usable joined substrate contained:

```text
SPY price/flow minutes: 376
raw RTH SPY trades: 206,304
raw RTH SPY quotes: 54,930
```

There are missing one-minute price observations inside the session. The research implementation was
therefore hardened to skip response fitting across irregular gaps and to create future labels using an
exact timestamp offset rather than a row offset.

## 3. Force reconstruction

Because `minute_flow` contains no SPY rows in this backup, force was reconstructed causally from the raw
SPY trade/quote tape using Beta-spy's `FlowAccumulator` logic.

The frozen MVP force remains:

```text
F = 0.60 * OFI + 0.25 * quote imbalance + 0.15 * liquidity-support asymmetry
```

Observed 2026-08-17 RTH distribution:

| Component | Mean | Std. dev. | Min | Max |
|---|---:|---:|---:|---:|
| Order-flow imbalance | 0.0486 | 0.2198 | -0.6908 | 0.6509 |
| Quote imbalance | -0.0231 | 0.1836 | -0.4634 | 0.4296 |
| Liquidity-support asymmetry | -0.0037 | 0.0434 | -0.1975 | 0.1840 |
| Composite force | 0.0228 | 0.1211 | -0.4504 | 0.3230 |

The force series is therefore non-trivial and directionally variable; failure below is not caused by a
constant-zero force input.

## 4. Estimability kill test

The strict MVP requires both fitted response coefficients to have the hypothesized positive sign before
reporting two-sided inertia:

```text
M_up   = 1 / beta_up
M_down = 1 / beta_down
```

After the 30-response warmup, 345 observations were eligible for coefficient-sign diagnostics.

### 60-minute response window

```text
valid upside inertia: 198 / 345 = 57.4%
valid downside inertia: 104 / 345 = 30.1%
valid two-sided inertia: 78 / 345 = 22.6%
model-ready rows with exact 5-minute labels: 76
IB vs future 5-minute return Spearman: +0.128
```

### 120-minute response window

```text
valid upside inertia: 236 / 345 = 68.4%
valid downside inertia: 65 / 345 = 18.8%
valid two-sided inertia: 47 / 345 = 13.6%
model-ready rows with exact 5-minute labels: 47
IB vs future 5-minute return Spearman: -0.154
```

### 240-minute response window

```text
valid upside inertia: 293 / 345 = 84.9%
valid downside inertia: 28 / 345 = 8.1%
valid two-sided inertia: 28 / 345 = 8.1%
model-ready rows with exact 5-minute labels: 28
IB vs future 5-minute return Spearman: +0.107
```

## 5. First verdict

The initial strict two-sided MVP **fails the estimability/data-sufficiency gate on the available tape**.

Reasons:

1. Two-sided inertia is simultaneously defined in only 8.1% to 22.6% of post-warmup observations,
   depending on the response window.
2. Only one usable full RTH raw-tape session is present in this snapshot.
3. The largest model-ready exact-label sample is 76 observations, far below the pre-registered minimum
   required to run the 5-fold walk-forward comparison.
4. The simple inertial-bias/future-return rank relationship changes sign across response windows.

Therefore no statistically defensible statement can currently be made that Market Mechanics improves
SPY forecasting.

This is **not** evidence that the entire pressure-response idea is false. It is evidence that the current
strict two-sided rolling inverse-response construction has not yet demonstrated itself as an operational
market state.

## 6. What is specifically prohibited now

Do not:

- wire this state into Delta,
- tune coefficients until the August 17 sample looks good,
- drop negative fitted coefficients by taking absolute values,
- select only the 60-minute window because its raw correlation is positive,
- optimize a trading rule on the same session,
- introduce options/futures inputs as an after-the-fact rescue.

Those actions would invalidate the purpose of the kill test.

## 7. Immediate research conclusion

The current hypothesis should remain in **RESEARCH / UNPROVEN** state.

The next legitimate evidence requirement is multiple additional full RTH SPY sessions with raw trades
and quotes (or faithfully persisted SPY minute-flow rows). The same frozen MVP should then be rerun
without changing its coefficients or promotion criteria.

Only after a multi-session sample exists should the project decide whether to:

1. reject the two-sided inertia construction;
2. test a pre-registered constrained/Bayesian positive-response estimator;
3. test a symmetric response model before directional asymmetry;
4. separate launch and braking response under a new Phase-II protocol.

## 8. Engineering corrections made because of this test

The research branch was updated so that:

- raw SPY trades/quotes are used to reconstruct minute force when SPY `minute_flow` is absent;
- the production `FlowAccumulator` supplies the reconstruction logic;
- multi-minute gaps cannot create fake one-minute acceleration observations;
- the estimator resets its rolling intraday state between sessions;
- future labels require the actual `timestamp + horizon`, not the Nth later database row;
- one-sided and two-sided estimability are reported separately;
- fewer than 300 fully model-ready exact-label rows cannot enter the walk-forward promotion test.

## 9. Promotion status

```text
Synthetic causal/unit tests:          IMPLEMENTED
Real-tape force reconstruction:       IMPLEMENTED
Real-tape estimability test:          FAILED / INSUFFICIENT
Purged walk-forward comparison:       NOT YET STATISTICALLY ADMISSIBLE
Delta integration:                    BLOCKED
Execution integration:                OUT OF SCOPE
```

The correct outcome of the first real test is therefore not a new signal.

It is a narrowed research question and a cleaner data requirement.
