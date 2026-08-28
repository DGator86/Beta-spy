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
    PremiumConfig,
    assess_day,
    build_condor,
    build_credit_spread,
    manage_position,
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
    day = path.stem.replace("ticks_", "")
    gate = ChainGate(cfg)
    bars = list(snapshots(path))
    if not bars:
        return {"day": day, "status": "no_data"}

    session_open = bars[0]["market"]["spot"]
    rejected_total: dict[str, int] = {}
    entry = None

    for bar in bars:
        hhmm = bar["ts"][11:16]
        if hhmm < cfg.entry_after:
            continue
        if hhmm > cfg.entry_before:
            break

        spot = bar["market"]["spot"]
        chain = gate.clean(spot, bar["option_rows"])
        for key, count in chain.rejected.items():
            rejected_total[key] = rejected_total.get(key, 0) + count
        if not chain.usable:
            continue

        verdict = assess_day(bar["market"], session_open=session_open, spot=spot, config=cfg)
        if not verdict.trade:
            return {"day": day, "status": "stand_aside", "reasons": list(verdict.reasons),
                    "notes": verdict.notes, "rejected": rejected_total}

        if structure_kind == "condor":
            structure = build_condor(chain, cfg)
        else:
            side = "call" if structure_kind == "call_credit" else "put"
            structure = build_credit_spread(chain, side, cfg)  # type: ignore[arg-type]
        if structure is None:
            continue
        if structure.credit_ratio < cfg.min_credit_ratio:
            continue

        contracts = size_position(equity, structure, cfg)
        if contracts <= 0:
            return {"day": day, "status": "too_small", "structure": structure.as_dict(),
                    "rejected": rejected_total}

        entry = {"ts": bar["ts"], "hhmm": hhmm, "spot": spot,
                 "structure": structure, "contracts": contracts}
        break

    if entry is None:
        return {"day": day, "status": "no_entry", "rejected": rejected_total}

    structure = entry["structure"]
    contracts = entry["contracts"]
    exit_reason, exit_value, exit_ts, exit_spot = "flat_eod", None, None, None
    unusable = 0

    for bar in bars:
        hhmm = bar["ts"][11:16]
        if bar["ts"] <= entry["ts"]:
            continue
        chain = gate.clean(bar["market"]["spot"], bar["option_rows"])
        if not chain.usable:
            unusable += 1
            continue
        reason, value = manage_position(structure, chain, hhmm, cfg)
        if value is None:
            unusable += 1
            continue
        exit_value, exit_ts, exit_spot = value, bar["ts"], bar["market"]["spot"]
        if reason is not None:
            exit_reason = reason
            break

    if exit_value is None:
        return {"day": day, "status": "never_markable", "rejected": rejected_total}

    pnl_per_share = structure.credit - exit_value
    settle = settlement(path)
    shorts = structure.short_strikes
    in_range = None
    if settle is not None:
        in_range = (min(shorts) <= settle <= max(shorts)) if len(shorts) > 1 else None

    return {
        "day": day,
        "status": "traded",
        "entry_time": entry["hhmm"],
        "entry_spot": entry["spot"],
        "structure": structure.as_dict(),
        "contracts": contracts,
        "credit": round(structure.credit, 4),
        "max_loss_per_share": round(structure.max_loss, 4),
        "risk_dollars": round(structure.max_loss * 100 * contracts, 2),
        "exit_reason": exit_reason,
        "exit_time": (exit_ts or "")[11:16],
        "exit_value": round(exit_value, 4),
        "exit_spot": exit_spot,
        "settle": settle,
        "settled_in_range": in_range,
        "pnl_per_share": round(pnl_per_share, 4),
        "pnl_dollars": round(pnl_per_share * 100 * contracts, 2),
        "unusable_marks": unusable,
        "rejected": rejected_total,
    }


def summarise(results: list[dict[str, Any]], label: str, equity: float) -> dict[str, Any]:
    traded = [r for r in results if r["status"] == "traded"]
    aside = [r for r in results if r["status"] == "stand_aside"]
    if not traded:
        print(f"\n{label}: no sessions traded ({len(aside)} stood aside)")
        return {"label": label, "n": 0}

    pnl = [r["pnl_dollars"] for r in traded]
    wins = [p for p in pnl if p > 0]
    total = sum(pnl)
    sd = st.stdev(pnl) if len(pnl) > 1 else 0.0
    se = sd / len(pnl) ** 0.5 if len(pnl) > 1 else 0.0

    print(f"\n{'=' * 82}")
    print(f"{label}")
    print("=" * 82)
    print(f"  {'day':11s} {'entry':6s} {'structure':18s} {'k':>3s} {'credit':>7s} "
          f"{'risk$':>8s} {'exit':>9s} {'P&L$':>9s}")
    for r in traded:
        print(f"  {r['day']:11s} {r['entry_time']:6s} {r['structure']['kind']:18s} "
              f"{r['contracts']:3d} {r['credit']:7.3f} {r['risk_dollars']:8.2f} "
              f"{r['exit_reason']:>9s} {r['pnl_dollars']:9.2f}")
    for r in aside:
        print(f"  {r['day']:11s} {'--':6s} {'STOOD ASIDE':18s} {'':3s} {'':7s} {'':8s} "
              f"{'':>9s} {'':>9s}   {r['reasons'][0] if r['reasons'] else ''}")

    print(f"\n  sessions traded   : {len(traded)}   stood aside: {len(aside)}   "
          f"other: {len(results) - len(traded) - len(aside)}")
    print(f"  win rate          : {len(wins)}/{len(pnl)} = {len(wins) / len(pnl):.1%}")
    print(f"  total P&L         : ${total:+,.2f} on ${equity:,.0f} starting equity "
          f"({total / equity:+.1%})")
    print(f"  per traded session: ${total / len(pnl):+,.2f}")
    if se:
        print(f"  95% CI on the mean: [${total / len(pnl) - 1.96 * se:+,.2f}, "
              f"${total / len(pnl) + 1.96 * se:+,.2f}]   t = {(total / len(pnl)) / se:+.2f}")
    print(f"  best / worst      : ${max(pnl):+,.2f} / ${min(pnl):+,.2f}")

    by_reason: dict[str, list[float]] = {}
    for r in traded:
        by_reason.setdefault(r["exit_reason"], []).append(r["pnl_dollars"])
    print("\n  by exit reason:")
    for reason, v in sorted(by_reason.items(), key=lambda x: -sum(x[1])):
        print(f"    {reason:10s} n={len(v):2d}  total ${sum(v):+9,.2f}  avg ${sum(v) / len(v):+8,.2f}")

    rej: dict[str, int] = {}
    for r in results:
        for k, c in (r.get("rejected") or {}).items():
            rej[k] = rej.get(k, 0) + c
    if any(rej.values()):
        print("\n  quotes rejected by the chain gate:")
        for k, c in sorted(rej.items(), key=lambda x: -x[1]):
            if c:
                print(f"    {k:18s} {c:,}")

    return {"label": label, "n": len(traded), "total": total, "win_rate": len(wins) / len(pnl),
            "t": (total / len(pnl)) / se if se else None}


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
