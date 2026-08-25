#!/usr/bin/env python
"""Verify a resume was clean: no step gap, no loss discontinuity.

    python check_resume.py --run-dir run

Restart boundaries are found where wall_s goes backwards, which happens because
each process restarts its own clock. At each boundary it reports the step jump
and the loss delta. A clean resume shows a step gap equal to your log interval
and a loss delta inside normal step-to-step noise.
"""

from __future__ import annotations

import argparse

from runstate import RunDir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="/kaggle/working/run")
    p.add_argument("--tolerance", type=float, default=0.05, help="acceptable loss_ema jump")
    args = p.parse_args()

    rows = RunDir(args.run_dir).read_csv()
    if len(rows) < 2:
        raise SystemExit("not enough rows in loss.csv yet")

    def f(row, key):
        v = row.get(key, "")
        return float(v) if v not in ("", None) else None

    boundaries = [i for i in range(1, len(rows))
                  if (f(rows[i], "wall_s") or 0) < (f(rows[i - 1], "wall_s") or 0)]

    # Raw loss is what matters here, not the ema. Compare the jump at the
    # boundary against typical step-to-step noise elsewhere in the run.
    deltas = []
    for i in range(1, len(rows)):
        if i in boundaries:
            continue
        a, b = f(rows[i - 1], "loss"), f(rows[i], "loss")
        if a is not None and b is not None:
            deltas.append(abs(b - a))
    noise = sorted(deltas)[len(deltas) // 2] if deltas else 0.0
    limit = max(args.tolerance, 4 * noise)

    print(f"{len(rows)} log rows, steps {rows[0]['step']} to {rows[-1]['step']}")
    print(f"typical step-to-step |dloss| = {noise:.4f}, so the boundary limit is {limit:.4f}")
    if not boundaries:
        print("no restart detected in this csv. Kill the run and start it again, then rerun this.")
        return

    bad = 0
    for i in boundaries:
        prev, cur = rows[i - 1], rows[i]
        step_gap = int(float(cur["step"])) - int(float(prev["step"]))
        lp, lc = f(prev, "loss"), f(cur, "loss")
        delta = None if lp is None or lc is None else lc - lp
        ok_step = step_gap > 0
        ok_loss = delta is not None and delta <= limit  # improving is always fine
        flag = "OK " if (ok_step and ok_loss) else "BAD"
        if flag == "BAD":
            bad += 1
        print(f"\n[{flag}] restart at row {i}")
        print(f"  step  {prev['step']} -> {cur['step']}   (gap {step_gap:+d}, "
              f"should be a small positive multiple of --log-every)")
        print(f"  loss  {lp} -> {lc}" + (f"   (delta {delta:+.4f})" if delta is not None else ""))
        print(f"  scale {prev.get('scale')} -> {cur.get('scale')}   "
              f"(a jump back to 65536 means GradScaler state was not restored)")
        if step_gap <= 0:
            print("  step went backwards or stalled: the checkpoint step count is not being restored")
        if not ok_loss:
            print("  loss jumped: most likely optimizer moments or the loss scale were dropped")

    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
