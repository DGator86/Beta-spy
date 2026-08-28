#!/usr/bin/env python3
"""Score the price-only directional rules over a longer sample than the tape.

The captured 0DTE tape only reaches back to 2026-07-01, which is 25 tradeable
sessions - not enough to tell a directional edge from a month of up-drift. The
feature-rich rules (dealer gamma, breadth, CVD) cannot be extended, because
nothing outside the capture carries those fields. The *price-only* rules can:
they need nothing but intraday bars.

So this script fetches plain SPY intraday bars and re-runs the subset of the
day-rule battery from ``directionality_audit.py`` that survives on price alone.
It answers one question the tape cannot: is the 10:00 -> 13:30 up-drift that
shows up in 24 sessions still there over a quarter, or was it July?

Yahoo is blocked from the research container's network egress, so run this
somewhere with open outbound HTTPS and commit the CSV it writes:

    pip install yfinance
    python scripts/intraday_drift.py --fetch --out data/spy_intraday.csv

Then, anywhere:

    python scripts/intraday_drift.py --bars data/spy_intraday.csv

Yahoo caps 5-minute history at 60 calendar days per request, so --period 3mo
falls back to 15-minute bars, which is still finer than the 10:00 / 13:30 /
15:30 marks the rules need.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from datetime import datetime
from math import comb
from zoneinfo import ZoneInfo

ENTRY = "10:00"
EXITS = ("13:30", "15:30")


def fetch(period: str, interval: str, out: str) -> None:
    try:
        import yfinance
    except ImportError:  # pragma: no cover - depends on where this is run
        raise SystemExit("pip install yfinance first")

    frame = yfinance.Ticker("SPY").history(period=period, interval=interval,
                                           prepost=False, auto_adjust=False)
    if frame.empty:
        raise SystemExit(
            "yfinance returned nothing. Either the symbol request failed or "
            "outbound HTTPS to Yahoo is blocked here - run this from a network "
            "that can reach query1.finance.yahoo.com."
        )
    frame = frame.tz_convert("America/New_York") if frame.index.tz else frame
    with open(out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ts", "open", "high", "low", "close", "volume"])
        for ts, row in frame.iterrows():
            writer.writerow([ts.isoformat(), row["Open"], row["High"],
                             row["Low"], row["Close"], row["Volume"]])
    print(f"wrote {len(frame)} bars to {out}: "
          f"{frame.index.min()} -> {frame.index.max()}")


def load_sessions(path: str, assume_tz: str) -> dict[str, list[tuple[str, float]]]:
    """Group bars into ET sessions.

    Timestamps are normalised to America/New_York before the session window is
    applied. This is not pedantry: the tape's own bar arrays are stamped in UTC
    while its tick stream is stamped in ET, so slicing 09:30-16:00 off the raw
    string silently reads the wrong four hours of the day and turns the answer
    upside down.
    """
    eastern = ZoneInfo("America/New_York")
    fallback = ZoneInfo(assume_tz)
    sessions: dict[str, list[tuple[str, float]]] = defaultdict(list)
    naive = 0
    with open(path) as fh:
        for row in csv.DictReader(fh):
            raw = row["ts"]
            try:
                stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if stamp.tzinfo is None:
                naive += 1
                stamp = stamp.replace(tzinfo=fallback)
            stamp = stamp.astimezone(eastern)
            hhmm = stamp.strftime("%H:%M")
            if not "09:30" <= hhmm <= "16:00":
                continue
            try:
                close = float(row["close"])
            except (TypeError, ValueError):
                continue
            sessions[stamp.strftime("%Y-%m-%d")].append((hhmm, close))
    if naive:
        print(f"note: {naive} timestamps carried no UTC offset and were read as "
              f"{assume_tz} (override with --assume-tz)")
    for day in sessions:
        sessions[day].sort()
    return sessions


def price_at(bars: list[tuple[str, float]], hhmm: str) -> float | None:
    """Last close at or before hhmm."""
    found = None
    for stamp, close in bars:
        if stamp > hhmm:
            break
        found = close
    return found


def exact_binomial_p(k: int, n: int, p: float = 0.5) -> float:
    pmf = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(x for x in pmf if x <= pmf[k] * (1 + 1e-9)))


RULES = {
    "always long": lambda opening, entry: 1,
    "always short": lambda opening, entry: -1,
    "open momentum": lambda opening, entry: 1 if entry > opening else -1,
    "fade open momentum": lambda opening, entry: -1 if entry > opening else 1,
}


def score(sessions: dict[str, list[tuple[str, float]]]) -> None:
    days = sorted(sessions)
    print(f"sessions={len(days)}  {days[0]} -> {days[-1]}")
    for exit_time in EXITS:
        rows = []
        for day in days:
            bars = sessions[day]
            opening = price_at(bars, "09:35")
            entry = price_at(bars, ENTRY)
            out = price_at(bars, exit_time)
            if opening is None or entry is None or out is None:
                continue
            rows.append((opening, entry, out - entry))
        if not rows:
            continue
        ups = sum(1 for _, _, move in rows if move > 0)
        print()
        print("=" * 74)
        print(f"{ENTRY} -> {exit_time}   n={len(rows)} sessions   "
              f"up on {ups / len(rows):.1%}")
        print("=" * 74)
        print(f"{'rule':20} {'hit':>6} {'k/n':>9} {'exact p':>8} "
              f"{'right $':>8} {'wrong $':>8} {'$/day':>8}")
        for name, rule in RULES.items():
            correct = total = 0
            right, wrong, pnl = [], [], []
            for opening, entry, move in rows:
                if move == 0:
                    continue
                side = rule(opening, entry)
                total += 1
                ok = side * move > 0
                correct += ok
                (right if ok else wrong).append(abs(move))
                pnl.append(side * move)
            if not total:
                continue
            print(f"{name:20} {correct / total:6.1%} {correct:4d}/{total:<4d} "
                  f"{exact_binomial_p(correct, total):8.3f} "
                  f"{(st.mean(right) if right else 0):8.2f} "
                  f"{(st.mean(wrong) if wrong else 0):8.2f} "
                  f"{st.mean(pnl):+8.2f}")

    print()
    print("If 'always long' stops being significant once the sample leaves July,")
    print("the tape's 70.8% was the month, and every long-biased rule measured")
    print("on it inherits that and nothing else.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true", help="download from yfinance first")
    parser.add_argument("--period", default="3mo")
    parser.add_argument("--interval", default="15m",
                        help="Yahoo caps 5m at 60 days; 15m reaches a full quarter")
    parser.add_argument("--out", default="data/spy_intraday.csv")
    parser.add_argument("--bars", help="scored CSV (defaults to --out)")
    parser.add_argument("--assume-tz", default="America/New_York",
                        help="zone for timestamps that carry no UTC offset; the "
                             "tape's own bar union is stamped UTC")
    args = parser.parse_args()

    if args.fetch:
        fetch(args.period, args.interval, args.out)
    score(load_sessions(args.bars or args.out, args.assume_tz))


if __name__ == "__main__":
    main()
