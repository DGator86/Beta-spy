# V2 MTF validation compressor

Research only. Do not merge/deploy as trading policy.

## Architecture

V2 separates the forecast problem into three causal components at 5m, 15m, and 30m horizons:

1. **Move-size head** — estimates whether the future move exceeds a horizon-scaled tradeability threshold (4.5 / 7.5 / 10.5 bps).
2. **Conditional direction head** — learns UP vs DOWN only from outcomes whose move exceeded the tradeability threshold, so tiny noisy moves do not train direction.
3. **Expected absolute-move head** — estimates the magnitude of the move.

Each forecast is queued with its maturity timestamp. A running validation compressor may update a horizon only after that forecast has actually matured. It tracks magnitude Brier score, conditional-direction Brier score, expected-magnitude error, and signed directional alignment. Current 5m/15m/30m forecasts are weighted by those already-matured validation scores and compressed toward `NO_TRADE` when recent evidence does not validate the forecast.

The design explicitly allows signed direction validation: a persistently backward horizon can be shrunk toward zero or inverted rather than being forced to retain a positive calibration slope.

## Locked V2.1 configuration

SHA-256: `422ed0ddc382705420fba642ce54eafdc741099b49e0ba035ad278bad26d8a31`

The complete Aug. 26 tape was retrieved only after V2.1 had been frozen.

## Historical causal replay

Using session-level expanding walk-forward fitting and intra-session maturity-delayed validation:

- Development magnitude discrimination was useful (15m large-move ROC-AUC about 0.70).
- Aug. 18–25 discrimination degraded materially, confirming regime instability.
- V2.1 emitted **no directional trades** because the running validation layer did not validate a sufficiently strong directional edge.
- It did identify quiet windows accurately, but fixed 0DTE iron-fly/condor/butterfly monetization failed after fees and modeled friction. Predictable quiet is therefore not automatically a trade.

## Aug. 26 forward validation

The completed Aug. 26 archive contains all 390 SPY regular-session minute bars. V2.1 was run without changing its frozen configuration.

- 62 five-minute evaluation anchors from 10:00–15:05/15:40-compatible dataset logic.
- Directional trades: **0**.
- Quiet candidates: **0**.
- Actual average absolute 15m SPY move: about **5.16 bps**.
- Actual >=7.5-bp move frequency: about **25.8%**.
- V2.1's Aug. 26 15m validated magnitude ranking was poor, so V2 correctly did not promote the model to a trade signal.

The existing paper runtime nevertheless opened two bullish call-debit-spread positions on Aug. 26. Their realized P&L was **+$20 and -$126 = -$106 net**; the second position also carried $252 modeled max loss and nine contracts. V2.1 would have blocked both entries because neither the validated move probability nor the validated directional edge passed the compressor.

## Kill test: V2.2 online probability recalibration

A more aggressive regime-conditioned probability recalibration variant was tested after V2.1. It degraded the Aug. 26 forward behavior and produced excessive false quiet classifications, so it is rejected and must not replace V2.1.

## Interpretation

V2.1 is **not a validated alpha model**. Its contribution is a causal supervisory layer that can determine when the underlying Alpha/Beta forecast stack is not presently trustworthy and force `NO_TRADE`. The next untouched session should test whether the compressor continues to protect against weak forecasts while still permitting genuinely validated opportunities.
