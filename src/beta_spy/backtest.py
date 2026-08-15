from __future__ import annotations

import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Iterable

from .engine import Tape500Engine
from .models import EngineSnapshot, HoldingMeta, HorizonForecast
from .replay import HistoricalReplay
from .storage import Tape500Store


@dataclass
class ForecastObservation:
    timestamp: datetime
    target_time: datetime
    horizon_minutes: int
    probability_up: float
    expected_return_bps: float
    confidence: float
    model_ready: bool
    sample_count: int
    start_price: float
    actual_price: float
    actual_return_bps: float
    direction_correct: bool
    brier: float
    absolute_return_error_bps: float


@dataclass
class HorizonMetrics:
    horizon_minutes: int
    observations: int = 0
    model_ready_observations: int = 0
    direction_accuracy: float | None = None
    model_ready_direction_accuracy: float | None = None
    brier: float | None = None
    model_ready_brier: float | None = None
    return_mae_bps: float | None = None
    model_ready_return_mae_bps: float | None = None
    avg_confidence: float | None = None


@dataclass
class DecisionMetrics:
    trade_signals: int = 0
    matured_trade_signals: int = 0
    direction_accuracy: float | None = None
    avg_underlying_return_bps: float | None = None
    median_like_return_bps: float | None = None
    positive_underlying_return_rate: float | None = None
    no_trade_signals: int = 0
    gate_failure_counts: dict[str, int] = field(default_factory=dict)
    neutral_signals: int = 0
    matured_neutral_signals: int = 0
    # Mean |SPY move| over the hold while short premium was on. Small values
    # confirm the quiet-tape gate is selecting the regime condors want.
    avg_abs_move_bps_neutral: float | None = None


@dataclass
class BacktestReport:
    created_at: datetime
    start: datetime | None
    end: datetime | None
    snapshots: int
    first_snapshot: datetime | None
    last_snapshot: datetime | None
    average_coverage_ratio: float | None
    average_covered_weight: float | None
    flow_minutes: int
    flow_coverage_note: str
    horizons: tuple[HorizonMetrics, ...]
    decisions: DecisionMetrics
    limitations: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _PendingForecast:
    timestamp: datetime
    target_time: datetime
    forecast: HorizonForecast
    start_price: float


@dataclass
class _PendingDecision:
    timestamp: datetime
    target_time: datetime
    direction: int
    start_price: float


def _spy_price(snapshot: EngineSnapshot) -> float | None:
    spy = next((item for item in snapshot.symbols if item.symbol == "SPY"), None)
    return float(spy.close) if spy is not None and spy.close > 0 else None


def _safe_mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return mean(rows) if rows else None


def _direction_correct(probability_up: float, realized_bps: float) -> bool:
    predicted_up = probability_up >= 0.5
    actual_up = realized_bps > 0
    return predicted_up == actual_up


def _median_like(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def run_backtest(
    store: Tape500Store,
    holdings: Iterable[HoldingMeta],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[BacktestReport, list[ForecastObservation]]:
    """Run the production feature/forecast pipeline causally and score matured forecasts.

    Forecasts are evaluated only when their target minute is reached in the same session.
    The online models update only after labels mature, so no future row is used to predict
    an earlier row.
    """
    engine = Tape500Engine(tuple(holdings), store=None)
    replay = HistoricalReplay(store, engine)
    pending_forecasts: dict[int, deque[_PendingForecast]] = defaultdict(deque)
    pending_decisions: deque[_PendingDecision] = deque()
    observations: list[ForecastObservation] = []
    decision_returns: list[float] = []
    decision_correct: list[bool] = []
    failure_counts: Counter[str] = Counter()
    coverage: list[float] = []
    weights: list[float] = []
    snapshots = 0
    first_snapshot: datetime | None = None
    last_snapshot: datetime | None = None
    trade_signals = 0
    no_trade_signals = 0
    neutral_signals = 0
    pending_neutral: deque[_PendingDecision] = deque()
    neutral_abs_moves: list[float] = []

    for snapshot in replay.run(start=start, end=end):
        snapshots += 1
        first_snapshot = first_snapshot or snapshot.timestamp
        last_snapshot = snapshot.timestamp
        coverage.append(snapshot.factors.coverage_ratio)
        weights.append(snapshot.factors.covered_weight)
        price = _spy_price(snapshot)
        if price is None:
            continue

        for horizon, queue in pending_forecasts.items():
            while queue and queue[0].target_time <= snapshot.timestamp:
                pending = queue.popleft()
                if pending.target_time.date() != snapshot.timestamp.date():
                    continue
                realized_bps = (price / pending.start_price - 1.0) * 10_000.0
                actual_up = 1.0 if realized_bps > 0 else 0.0
                observations.append(
                    ForecastObservation(
                        timestamp=pending.timestamp,
                        target_time=pending.target_time,
                        horizon_minutes=horizon,
                        probability_up=pending.forecast.probability_up,
                        expected_return_bps=pending.forecast.expected_return_bps,
                        confidence=pending.forecast.confidence,
                        model_ready=pending.forecast.model_ready,
                        sample_count=pending.forecast.sample_count,
                        start_price=pending.start_price,
                        actual_price=price,
                        actual_return_bps=realized_bps,
                        direction_correct=_direction_correct(pending.forecast.probability_up, realized_bps),
                        brier=(pending.forecast.probability_up - actual_up) ** 2,
                        absolute_return_error_bps=abs(pending.forecast.expected_return_bps - realized_bps),
                    )
                )

        while pending_decisions and pending_decisions[0].target_time <= snapshot.timestamp:
            pending = pending_decisions.popleft()
            if pending.target_time.date() != snapshot.timestamp.date():
                continue
            realized_bps = (price / pending.start_price - 1.0) * 10_000.0 * pending.direction
            decision_returns.append(realized_bps)
            decision_correct.append(realized_bps > 0)

        while pending_neutral and pending_neutral[0].target_time <= snapshot.timestamp:
            pending = pending_neutral.popleft()
            if pending.target_time.date() != snapshot.timestamp.date():
                continue
            neutral_abs_moves.append(abs(price / pending.start_price - 1.0) * 10_000.0)

        for forecast in snapshot.forecasts:
            pending_forecasts[forecast.horizon_minutes].append(
                _PendingForecast(
                    timestamp=snapshot.timestamp,
                    target_time=snapshot.timestamp + timedelta(minutes=forecast.horizon_minutes),
                    forecast=forecast,
                    start_price=price,
                )
            )

        if snapshot.decision.action == "TRADE":
            trade_signals += 1
            direction = 1 if snapshot.decision.direction == "BULLISH" else -1
            pending_decisions.append(
                _PendingDecision(
                    timestamp=snapshot.timestamp,
                    target_time=snapshot.timestamp + timedelta(minutes=snapshot.decision.primary_horizon),
                    direction=direction,
                    start_price=price,
                )
            )
        elif snapshot.decision.action == "TRADE_NEUTRAL":
            neutral_signals += 1
            pending_neutral.append(
                _PendingDecision(
                    timestamp=snapshot.timestamp,
                    target_time=snapshot.timestamp + timedelta(minutes=snapshot.decision.primary_horizon),
                    direction=0,
                    start_price=price,
                )
            )
        else:
            no_trade_signals += 1
            for gate, passed in snapshot.decision.gates.items():
                if not passed:
                    failure_counts[gate] += 1

    horizon_metrics: list[HorizonMetrics] = []
    for horizon in sorted({item.horizon_minutes for item in observations} | {5, 15, 30}):
        rows = [item for item in observations if item.horizon_minutes == horizon]
        ready = [item for item in rows if item.model_ready]
        horizon_metrics.append(
            HorizonMetrics(
                horizon_minutes=horizon,
                observations=len(rows),
                model_ready_observations=len(ready),
                direction_accuracy=_safe_mean(float(item.direction_correct) for item in rows),
                model_ready_direction_accuracy=_safe_mean(float(item.direction_correct) for item in ready),
                brier=_safe_mean(item.brier for item in rows),
                model_ready_brier=_safe_mean(item.brier for item in ready),
                return_mae_bps=_safe_mean(item.absolute_return_error_bps for item in rows),
                model_ready_return_mae_bps=_safe_mean(item.absolute_return_error_bps for item in ready),
                avg_confidence=_safe_mean(item.confidence for item in rows),
            )
        )

    with store.lock:
        row = store.connection.execute("SELECT COUNT(*) FROM minute_flow").fetchone()
    flow_minutes = int(row[0] if row else 0)
    report = BacktestReport(
        created_at=datetime.now().astimezone(),
        start=start,
        end=end,
        snapshots=snapshots,
        first_snapshot=first_snapshot,
        last_snapshot=last_snapshot,
        average_coverage_ratio=_safe_mean(coverage),
        average_covered_weight=_safe_mean(weights),
        flow_minutes=flow_minutes,
        flow_coverage_note=(
            "Historical minute-flow rows were present and used where timestamps/symbols matched."
            if flow_minutes
            else "No historical trade/quote flow was present; this run scores the price/volume/indicator breadth layer only."
        ),
        horizons=tuple(horizon_metrics),
        decisions=DecisionMetrics(
            trade_signals=trade_signals,
            matured_trade_signals=len(decision_returns),
            direction_accuracy=_safe_mean(float(value) for value in decision_correct),
            avg_underlying_return_bps=_safe_mean(decision_returns),
            median_like_return_bps=_median_like(decision_returns),
            positive_underlying_return_rate=_safe_mean(float(value > 0) for value in decision_returns),
            no_trade_signals=no_trade_signals,
            gate_failure_counts=dict(failure_counts.most_common()),
            neutral_signals=neutral_signals,
            matured_neutral_signals=len(neutral_abs_moves),
            avg_abs_move_bps_neutral=_safe_mean(neutral_abs_moves),
        ),
        limitations=(
            "The historical universe uses the supplied point-in-time/current holdings file; older periods can contain survivorship and weight bias unless historical holdings are supplied.",
            "Underlying forecast accuracy is evaluated independently of options execution. Historical expired option chains are not reconstructed by this report.",
            "Bars-only runs do not test historical order-flow features; order-flow metrics require historical trades+quotes or a previously captured Beta-spy tape.",
        ),
    )
    return report, observations


def write_report(report: BacktestReport, output: Path | str) -> tuple[Path, Path]:
    path = Path(output)
    if path.suffix.lower() in {".md", ".json"}:
        base = path.with_suffix("")
    else:
        base = path
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(report.as_dict(), default=str, indent=2), encoding="utf-8")

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.2f}%"

    def num(value: float | None, suffix: str = "") -> str:
        return "n/a" if value is None or not math.isfinite(value) else f"{value:.3f}{suffix}"

    lines = [
        "# Beta-spy causal backtest",
        "",
        f"Created: {report.created_at.isoformat()}",
        f"Replay range: {report.first_snapshot.isoformat() if report.first_snapshot else 'n/a'} → {report.last_snapshot.isoformat() if report.last_snapshot else 'n/a'}",
        f"Snapshots: {report.snapshots:,}",
        f"Average universe coverage: {pct(report.average_coverage_ratio)}",
        f"Average covered SPY weight: {pct(report.average_covered_weight)}",
        f"Historical flow rows: {report.flow_minutes:,}",
        "",
        "## Forecast performance",
        "",
        "| Horizon | Obs | Model-ready | Accuracy | Ready accuracy | Brier | Ready Brier | Return MAE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.horizons:
        lines.append(
            f"| {item.horizon_minutes}m | {item.observations:,} | {item.model_ready_observations:,} | "
            f"{pct(item.direction_accuracy)} | {pct(item.model_ready_direction_accuracy)} | "
            f"{num(item.brier)} | {num(item.model_ready_brier)} | {num(item.model_ready_return_mae_bps, ' bps')} |"
        )
    decision = report.decisions
    lines.extend(
        [
            "",
            "## Decision layer",
            "",
            f"- Trade signals: {decision.trade_signals:,}",
            f"- Matured trade signals: {decision.matured_trade_signals:,}",
            f"- Direction accuracy: {pct(decision.direction_accuracy)}",
            f"- Mean 15m direction-adjusted SPY return: {num(decision.avg_underlying_return_bps, ' bps')}",
            f"- Median 15m direction-adjusted SPY return: {num(decision.median_like_return_bps, ' bps')}",
            f"- Positive direction-adjusted return rate: {pct(decision.positive_underlying_return_rate)}",
            f"- Neutral premium signals: {decision.neutral_signals:,}"
            f" (matured {decision.matured_neutral_signals:,},"
            f" mean |move| {num(decision.avg_abs_move_bps_neutral, ' bps')})",
            f"- NO_TRADE snapshots: {decision.no_trade_signals:,}",
            "",
            "### Most common failed gates",
            "",
        ]
    )
    if decision.gate_failure_counts:
        for gate, count in decision.gate_failure_counts.items():
            lines.append(f"- `{gate}`: {count:,}")
    else:
        lines.append("- None")
    lines.extend(["", "## Flow coverage", "", report.flow_coverage_note, "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report.limitations)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path
