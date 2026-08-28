#!/usr/bin/env python3
"""Backtest the short-premium engine over captured 0DTE tick tape.

Reads the ``ticks_YYYY-MM-DD.jsonl`` files produced by the capture service and
drives :mod:`beta_spy.premium` over them one session at a time: validate the
chain, assess the day, build a structure, size it, then mark it forward to a
target, a stop, or the close.

Nothing is peeked. Every decision at time *t* uses only the snapshot at *t*.

    python scripts/premium_backtest.py --tape ./tape
    python scripts/premium_backtest.py --tape ./tape --structure condor --delta 0.16
    python scripts/premium_backtest.py --tape ./tape --compare
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from beta_spy.premium import (  # noqa: E402
    ChainGate,
    PositionState,
    strike_pressure,
    PremiumConfig,
    assess_day,
    build_condor,
    build_credit_spread,
    monitor,
    required_credit_ratio,
    size_position,
)


def snapshots(path: Path) -> Iterator[dict[str, Any]]:
    """Yield full chain snapshots from one session file, in time order."""
    for line in path.open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("t") != "tick":
            continue
        if not row.get("option_rows") or not row.get("market", {}).get("spot"):
            continue
        yield row


def settlement(path: Path) -> float | None:
    for line in path.open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("t") == "settle":
            return float(row["price"])
    return None


def run_session(path: Path, cfg: PremiumConfig, structure_kind: str, equity: float) -> dict[str, Any]:
    """Drive one session: assess, enter, monitor, exit, and re-enter if allowed."""
    day = path.stem.replace("ticks_", "")
    gate = ChainGate(cfg)
    bars = list(snapshots(path))
    if not bars:
        return {"day": day, "status": "no_data"}

    session_open = bars[0]["market"]["spot"]
    rejected_total: dict[str, int] = {}
    trades: list[dict[str, Any]] = []
    aside_reasons: list[str] = []
    state: PositionState | None = None
    entry_bar: dict[str, Any] | None = None
    closed_at_min = -10_000
    session_equity = equity

    for bar in bars:
        hhmm = bar["ts"][11:16]
        spot = bar["market"]["spot"]
        chain = gate.clean(spot, bar["option_rows"])
        for key, count in chain.rejected.items():
            rejected_total[key] = rejected_total.get(key, 0) + count

        # ---- manage an open position first ----
        if state is not None:
            result = monitor(state, chain, bar["market"], hhmm, cfg)
            if result.exit_reason is not None and result.value is not None:
                pnl = (state.structure.credit - result.value) * 100 * state.contracts
                session_equity += pnl
                trades.append({
                    "entry_time": state.entry_hhmm, "exit_time": hhmm,
                    "kind": state.structure.kind, "contracts": state.contracts,
                    "credit": round(state.structure.credit, 4),
                    "exit_value": round(result.value, 4),
                    "risk_dollars": round(state.structure.max_loss * 100 * state.contracts, 2),
                    "exit_reason": result.exit_reason,
                    "pressure_at_exit": None if result.pressure is None else round(result.pressure, 3),
                    "unusable_marks": state.marks_unusable,
                    "pnl_dollars": round(pnl, 2),
                    "note": result.note,
                })
                state, entry_bar = None, None
                closed_at_min = _hhmm_minutes(hhmm)
            continue

        # ---- otherwise look for an entry, re-assessing every bar ----
        if hhmm < cfg.entry_after or hhmm > cfg.entry_before:
            continue
        if len(trades) >= cfg.max_entries_per_day:
            continue
        if _hhmm_minutes(hhmm) - closed_at_min < cfg.reentry_cooldown_minutes:
            continue
        if not chain.usable:
            continue

        verdict = assess_day(bar["market"], session_open=session_open, spot=spot, config=cfg)
        if not verdict.trade:
            if verdict.reasons and verdict.reasons[0] not in aside_reasons:
                aside_reasons.append(verdict.reasons[0])
            continue

        if structure_kind == "condor":
            structure = build_condor(chain, cfg)
        else:
            side = "call" if structure_kind == "call_credit" else "put"
            structure = build_credit_spread(chain, side, cfg)  # type: ignore[arg-type]
        if structure is None:
            continue
        if structure.credit_ratio < required_credit_ratio(hhmm, cfg):
            continue

        entry_pressure = strike_pressure(structure, spot, bar["market"], hhmm, cfg)
        if cfg.min_entry_pressure is not None and (
            entry_pressure is None or entry_pressure < cfg.min_entry_pressure
        ):
            continue

        contracts = size_position(session_equity, structure, cfg, now_hhmm=hhmm)
        if contracts <= 0:
            continue

        state = PositionState(structure=structure, contracts=contracts, entry_hhmm=hhmm,
                              entry_pressure=entry_pressure)
        entry_bar = bar

    # a position still open at the last bar is marked out there
    if state is not None and entry_bar is not None:
        last = bars[-1]
        chain = gate.clean(last["market"]["spot"], last["option_rows"])
        value = state.structure.value_at(chain)
        if value is not None:
            pnl = (state.structure.credit - value) * 100 * state.contracts
            session_equity += pnl
            trades.append({
                "entry_time": state.entry_hhmm, "exit_time": last["ts"][11:16],
                "kind": state.structure.kind, "contracts": state.contracts,
                "credit": round(state.structure.credit, 4), "exit_value": round(value, 4),
                "risk_dollars": round(state.structure.max_loss * 100 * state.contracts, 2),
                "exit_reason": "last_bar", "pressure_at_exit": None,
                "unusable_marks": state.marks_unusable,
                "pnl_dollars": round(pnl, 2), "note": "",
            })

    if not trades:
        return {"day": day, "status": "stand_aside" if aside_reasons else "no_entry",
                "reasons": aside_reasons, "rejected": rejected_total}

    return {
        "day": day, "status": "traded", "trades": trades,
        "entries": len(trades),
        "pnl_dollars": round(sum(t["pnl_dollars"] for t in trades), 2),
        "settle": settlement(path),
        "rejected": rejected_total,
    }


def _hhmm_minutes(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:5])


def summarise(results: list[dict[str, Any]], label: str, equity: float) -> dict[str, Any]:
    traded = [r for r in results if r["status"] == "traded"]
    aside = [r for r in results if r["status"] == "stand_aside"]
    if not traded:
        print(f"\n{label}: no sessions traded ({len(aside)} stood aside)")
        return {"label": label, "n": 0}

    flat = [t for r in traded for t in r["trades"]]
    day_pnl = [r["pnl_dollars"] for r in traded]
    total = sum(day_pnl)
    sd = st.stdev(day_pnl) if len(day_pnl) > 1 else 0.0
    se = sd / len(day_pnl) ** 0.5 if len(day_pnl) > 1 else 0.0

    print(f"\n{'=' * 96}")
    print(label)
    print("=" * 96)
    print(f"  {'day':11s} {'#':>2s} {'in':>6s} {'out':>6s} {'k':>3s} {'credit':>7s} "
          f"{'risk$':>8s} {'exit':>9s} {'press':>6s} {'P&L$':>9s}")
    for r in traded:
        for i, t in enumerate(r["trades"], 1):
            day = r["day"] if i == 1 else ""
            press = "--" if t["pressure_at_exit"] is None else f"{t['pressure_at_exit']:.2f}"
            print(f"  {day:11s} {i:2d} {t['entry_time']:>6s} {t['exit_time']:>6s} "
                  f"{t['contracts']:3d} {t['credit']:7.3f} {t['risk_dollars']:8.2f} "
                  f"{t['exit_reason']:>9s} {press:>6s} {t['pnl_dollars']:9.2f}")
        if len(r["trades"]) > 1:
            print(f"  {'':11s} {'':2s} {'':6s} {'':6s} {'':3s} {'':7s} {'':8s} "
                  f"{'day total':>9s} {'':6s} {r['pnl_dollars']:9.2f}")
    for r in aside:
        print(f"  {r['day']:11s} {'--':>2s} {'STOOD ASIDE':>13s}   {r['reasons'][0] if r['reasons'] else ''}")

    wins = [p for p in day_pnl if p > 0]
    tw = [t for t in flat if t["pnl_dollars"] > 0]
    print(f"\n  sessions traded    : {len(traded)}   stood aside: {len(aside)}   "
          f"other: {len(results) - len(traded) - len(aside)}")
    print(f"  positions opened   : {len(flat)}  ({len(flat)/len(traded):.2f} per traded session)")
    print(f"  win rate by session: {len(wins)}/{len(day_pnl)} = {len(wins)/len(day_pnl):.1%}")
    print(f"  win rate by trade  : {len(tw)}/{len(flat)} = {len(tw)/len(flat):.1%}")
    print(f"  total P&L          : ${total:+,.2f} on ${equity:,.0f} ({total/equity:+.1%})")
    print(f"  per traded session : ${total/len(day_pnl):+,.2f}")
    if se:
        print(f"  95% CI on the mean : [${total/len(day_pnl) - 1.96*se:+,.2f}, "
              f"${total/len(day_pnl) + 1.96*se:+,.2f}]   t = {(total/len(day_pnl))/se:+.2f}")
    print(f"  best / worst day   : ${max(day_pnl):+,.2f} / ${min(day_pnl):+,.2f}")

    by: dict[str, list[float]] = {}
    for t in flat:
        by.setdefault(t["exit_reason"], []).append(t["pnl_dollars"])
    print("\n  by exit reason (per position):")
    for reason, v in sorted(by.items(), key=lambda x: -sum(x[1])):
        w = sum(1 for x in v if x > 0)
        print(f"    {reason:10s} n={len(v):3d}  win {w/len(v):5.0%}  total ${sum(v):+9,.2f}  "
              f"avg ${sum(v)/len(v):+8,.2f}")

    unusable = sum(t["unusable_marks"] for t in flat)
    rej: dict[str, int] = {}
    for r in results:
        for k, c in (r.get("rejected") or {}).items():
            rej[k] = rej.get(k, 0) + c
    print(f"\n  marks the gate refused to act on: {unusable:,}")
    if any(rej.values()):
        print("  quotes rejected at ingest:")
        for k, c in sorted(rej.items(), key=lambda x: -x[1]):
            if c:
                print(f"    {k:18s} {c:,}")

    return {"label": label, "n": len(traded), "total": total,
            "win_rate": len(wins)/len(day_pnl), "positions": len(flat),
            "t": (total/len(day_pnl))/se if se else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tape", type=Path, required=True, help="directory of ticks_*.jsonl files")
    ap.add_argument("--structure", choices=["condor", "call_credit", "put_credit"], default="condor")
    ap.add_argument("--delta", type=float, default=0.16)
    ap.add_argument("--wing", type=float, default=2.0)
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--risk-fraction", type=float, default=0.02)
    ap.add_argument("--no-gate", action="store_true", help="disable chain validation (to show its effect)")
    ap.add_argument("--gamma-filter", action="store_true", help="enable the unproven dealer-gamma filter")
    ap.add_argument("--compare", action="store_true", help="run with and without the chain gate")
    ap.add_argument("--json", type=Path, help="write full per-session results here")
    args = ap.parse_args()

    files = sorted(args.tape.glob("ticks_*.jsonl"))
    if not files:
        print(f"no ticks_*.jsonl under {args.tape}", file=sys.stderr)
        return 1

    base = PremiumConfig(
        short_delta=args.delta,
        wing_points=args.wing,
        risk_fraction=args.risk_fraction,
        gamma_filter=args.gamma_filter,
        reject_below_intrinsic=not args.no_gate,
        reject_vertical_breach=not args.no_gate,
    )

    runs = [("chain gate ON", base)]
    if args.compare:
        runs.append(("chain gate OFF", replace(base, reject_below_intrinsic=False,
                                               reject_vertical_breach=False)))

    all_results = {}
    for label, cfg in runs:
        results = [run_session(f, cfg, args.structure, args.equity) for f in files]
        summarise(results, f"{args.structure.upper()} @ {args.delta:.2f}delta, "
                           f"{args.risk_fraction:.0%} risk  --  {label}", args.equity)
        all_results[label] = results

    if args.json:
        args.json.write_text(json.dumps(all_results, indent=1, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
