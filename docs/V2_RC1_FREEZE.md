# Beta-SPY V2 RC1 Freeze

Freeze ID: `alpha-beta-v2.0.0-rc1`

This file defines the architecture and thresholds that may be used for the next untouched forward test. Do not tune these values from August 18-26 or from the August 27 outcome.

## Authority boundary

- Beta is a market-state estimator only.
- `strategy_authority = false`.
- Beta V2 does not open new option positions.
- Alpha V2 is the sole payoff-geometry, trade/no-trade, risk and execution authority.

## Forecast heads

Horizons: `5m, 15m, 30m`.

Each horizon independently estimates:

1. probability of an economically meaningful absolute move;
2. direction conditional on a meaningful move;
3. expected absolute move in basis points;
4. expected signed move in basis points.

Direction is trained only when the realized absolute move exceeds that horizon's economic threshold.

Economic move threshold:

`7.5 bps * sqrt(horizon_minutes / 15)`

Thus the 15-minute primary threshold is exactly 7.5 bps.

## Maturity-delayed validation

- A forecast cannot update its validator before its own target time matures.
- Validation lookback: 400 matured model-ready forecasts per horizon.
- Model warmup: 200 matured samples per horizon.
- Magnitude trust: rolling Brier skill against a causal Beta(10,10)-shrunk empirical base rate.
- Direction trust: Beta(10,10)-shrunk signed directional accuracy on meaningful moves only.
- Signed directional alignment may be negative; backward heads can be suppressed/inverted causally.

## MTF compression

Role weights:

- 5m: 0.20
- 15m: 0.55
- 30m: 0.25

Regime thresholds:

- `UNTRUSTED`: aggregate magnitude trust < 0.15
- `DIRECTIONAL_EXPANSION`: P(big move) >= 0.65, absolute validated direction edge >= 0.20, direction trust >= 0.20
- `EXPANSION_UNCERTAIN_DIRECTION`: P(big move) >= 0.65 without validated directional qualification
- `QUIET`: P(big move) <= 0.35
- otherwise `NORMAL`

## Features

V2 uses the existing `full_v1` point-in-time Beta feature set. No future-revised fields may enter the feature vector.

## Persistence / replay

- Persist timestamp-level V2 market state in the Beta database.
- Warm start uses historical replay in chronological order; each validator still waits for actual horizon maturity.
- The session-tape archiver now carries V2 market state plus Alpha's exact option-chain/candidate evidence.

## Forward-test rule

August 18-26 are burned development/diagnostic sessions for RC1. August 27 is the next untouched session. No parameter, threshold, feature, family or scoring change may use August 27 outcomes before the RC1 result is recorded.