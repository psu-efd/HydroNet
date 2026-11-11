#!/usr/bin/env python3
"""
Live progress bars for FVM-PINN array runs, read from their logs.

The trainers log a line every 100 steps (no tqdm), so this script parses
runs/<RUN_ID>/stdout.log and draws one Adam bar and one L-BFGS bar per run,
with the Slurm state for runs that have not started.

L-BFGS logs count loss evaluations; the bar uses the budget
outer_steps x 25 evaluations (max_eval = 1.25 x max_iter), and runs often stop before it, so that
bar is an upper-bound estimate. A run is marked done when the log reports
its total wall time.

Usage (from anywhere; Ctrl-C to quit)
-----
    python watch_runs.py                       # R2-4 extrapolation runs
    python watch_runs.py --jobs 57081273,57081274 --runs 83-88
    python watch_runs.py --once                # print once and exit
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LBFGS_EVALS_PER_STEP = 25   # torch L-BFGS max_eval = 1.25 x max_iter (20)
BAR = 28


def run_ids() -> list[str]:
    """RUN_IDS array from run_experiments_array.sbatch (index = array task)."""
    txt = (HERE / "run_experiments_array.sbatch").read_text()
    body = re.search(r"RUN_IDS=\((.*?)\)", txt, re.S).group(1)
    return body.split()


def case_dirs() -> dict[str, str]:
    out = {}
    for line in (HERE / "run_all_experiments.sh").read_text().splitlines():
        m = re.match(r'\s+"(\S+)\s+(\S+)\s', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def slurm_states(jobs: str) -> dict[int, tuple[str, str, str]]:
    """array task -> (state, elapsed, expected start)."""
    try:
        txt = subprocess.run(["squeue", "-h", "-r", "-j", jobs, "-o", "%i|%T|%M|%S|%R"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001 - squeue unavailable or slow
        return {}
    out = {}
    for line in txt.splitlines():
        jid, state, elapsed, start, reason = line.split("|")
        if "_" in jid and "[" not in jid:
            out[int(jid.split("_")[1])] = (state, elapsed, start if start != "N/A" else reason)
        elif "[" in jid:   # still-collapsed pending array: 123_[85-88]
            lo, hi = re.search(r"\[(\d+)-?(\d*)", jid).groups()
            for k in range(int(lo), int(hi or lo) + 1):
                out[k] = (state, elapsed, reason if state == "PENDING" else start)
    return out


def parse_log(path: Path) -> dict:
    s = {"adam": (0, None), "lbfgs": (0, None), "rate": None, "done": None, "error": False, "last": ""}
    if not path.exists():
        return s
    adam_n = lbfgs_n = None
    adam_k = lbfgs_k = 0
    for line in path.read_text(errors="replace").splitlines():
        if m := re.search(r"Adam: (\d+)", line):
            adam_n = int(m.group(1))
        elif m := re.search(r"L-BFGS(?: polish)?: (\d+)", line):
            lbfgs_n = int(m.group(1)) * LBFGS_EVALS_PER_STEP
        if m := re.search(r"Adam \[(\d+)", line):
            adam_k = int(m.group(1))
        elif m := re.search(r"phase=adam\s+step=(\d+)", line):
            adam_k = max(adam_k, int(m.group(1)))
        if m := re.search(r"L-BFGS \[(\d+)", line):
            lbfgs_k = int(m.group(1))
        elif m := re.search(r"phase=lbfgs\s+step=(\d+)", line):
            lbfgs_k = max(lbfgs_k, int(m.group(1)))
        if m := re.search(r"\(([\d.]+) s/step\)", line):
            s["rate"] = float(m.group(1))
        if m := re.search(r"Total wall time: ([\d.]+)", line):
            s["done"] = float(m.group(1))
        if "Traceback" in line:
            s["error"] = True
        if "L2(|V|)" in line or "L2(h)" in line:
            s["last"] = line.split("INFO")[-1].strip()
    s["adam"], s["lbfgs"] = (adam_k, adam_n), (lbfgs_k, lbfgs_n)
    return s


def bar(k: int, n: int | None) -> str:
    if not n:
        return "[" + "·" * BAR + "]   --  "
    f = min(k / n, 1.0)
    fill = int(round(f * BAR))
    return "[" + "█" * fill + "·" * (BAR - fill) + f"] {100 * f:5.1f}%"


def hms(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m"


def render(jobs: str, tasks: list[int]) -> str:
    ids, dirs, states = run_ids(), case_dirs(), slurm_states(jobs)
    rows = [time.strftime("FVM-PINN runs   %Y-%m-%d %H:%M:%S   (Ctrl-C to quit)"), ""]
    for t in tasks:
        rid = ids[t]
        log = HERE / dirs.get(rid, "") / "runs" / rid / "stdout.log"
        st = states.get(t)
        p = parse_log(log) if (st is None or st[0] == "RUNNING") else parse_log(Path("/nonexistent"))
        head = f"{t:3d} {rid:<12}"
        if p["done"] is not None:
            rows.append(f"{head} DONE in {hms(p['done'])}{'  (ERROR in log)' if p['error'] else ''}")
            if p["last"]:
                rows.append(f"{'':17}{p['last']}")
        elif p["error"] and (st is None):
            rows.append(f"{head} FAILED - see {log}")
        elif st and st[0] != "RUNNING":
            rows.append(f"{head} {st[0]:<9} {st[2]}")
        elif st is None and not log.exists():
            rows.append(f"{head} not in queue, no log")
        else:
            (ak, an), (lk, ln) = p["adam"], p["lbfgs"]
            eta = ""
            if p["rate"]:
                left = max((an or 0) - ak, 0) + max((ln or 0) - lk, 0)
                eta = f"  ≤{hms(left * p['rate'])} left" if (an or ln) else ""
            rows.append(f"{head} RUNNING {st[1] if st else ''}{eta}")
            rows.append(f"{'':17}Adam   {bar(ak, an)}  {ak}/{an or '?'}")
            rows.append(f"{'':17}L-BFGS {bar(lk, ln)}  {lk}/{ln or '?'} (budget)")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="57081273,57081274")
    ap.add_argument("--runs", default="83-88", help="array task range, e.g. 83-88 or 83,85")
    ap.add_argument("--every", type=float, default=15.0, help="refresh interval (s)")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    tasks = []
    for part in a.runs.split(","):
        lo, _, hi = part.partition("-")
        tasks += list(range(int(lo), int(hi or lo) + 1))
    if a.once:
        print(render(a.jobs, tasks))
        return 0
    try:
        while True:
            out = render(a.jobs, tasks)
            sys.stdout.write("\033[2J\033[H" + out + "\n")
            sys.stdout.flush()
            time.sleep(a.every)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
