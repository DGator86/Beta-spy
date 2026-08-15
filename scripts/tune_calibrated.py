"""Threshold and sizing checks on the calibrated signals dump."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/opt/beta-spy/scripts")
from sweep_decision import evaluate  # noqa: E402

df = pd.read_csv("/var/lib/beta-spy/research/signals2.csv", parse_dates=["timestamp"])
df["date"] = df["timestamp"].dt.date.astype(str)
train = df[df["date"] < "2026-08-08"].reset_index(drop=True)
test = df[df["date"] >= "2026-08-08"].reset_index(drop=True)

print("=== h=15 strict fast_pair flow, by threshold (train | test) ===")
for min_p in (0.55, 0.58, 0.60, 0.62):
    tr = evaluate(train, horizon=15, min_p=min_p, breadth_mode="strict",
                  agree_mode="fast_pair", flow_gate=True, conf_sizing=True)
    te = evaluate(test, horizon=15, min_p=min_p, breadth_mode="strict",
                  agree_mode="fast_pair", flow_gate=True, conf_sizing=True)
    print(f"p>={min_p}: train n={tr['trades']} acc={tr.get('accuracy', 0):.3f} "
          f"mean={tr.get('mean_bps', 0):.2f} sized={tr.get('sized_total_bps', 0):.0f} | "
          f"test n={te['trades']} acc={te.get('accuracy', 0):.3f} "
          f"mean={te.get('mean_bps', 0):.2f} sized={te.get('sized_total_bps', 0):.0f}")

print()
print("=== accuracy by calibrated edge bucket (threshold trades, h=15) ===")
for name, part in (("train", train), ("test", test)):
    p = part["p_up_15"]
    d = np.where(p >= 0.58, 1, np.where(p <= 0.42, -1, 0))
    fwd = part["fwd_bps_15"]
    m = (d != 0) & fwd.notna()
    realized = fwd[m] * d[m]
    edge = (p[m] - 0.5).abs()
    buckets = pd.cut(edge, [0.08, 0.12, 0.18, 0.25, 0.5])
    g = pd.DataFrame({"r": realized, "b": buckets}).groupby("b", observed=True)["r"]
    print(name, "n=", g.count().tolist(),
          "acc=", [round(v, 3) for v in g.apply(lambda s: (s > 0).mean())],
          "mean=", [round(v, 2) for v in g.mean()])
