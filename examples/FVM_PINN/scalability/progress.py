#!/usr/bin/env python3
"""Live progress of the R2-6 benchmark array (and the R2-4 runs).

Run in a separate terminal:   watch -n 30 -t python3 progress.py [ARRAY_ID]
"""
import glob
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ARRAY = sys.argv[1] if len(sys.argv) > 1 else "57083665"
LOGS = Path(__file__).resolve().parent.parent / "logs"
TAGS = [("n1306", 1306), ("n5000", 4848), ("n10000", 9606), ("n20000", 19061),
        ("n50000", 47476), ("n100000", 94586), ("n200000", 188876)]
N_SET = 6


def sh(cmd):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=20).stdout
    except Exception:
        return ""


def bar(frac, width):
    n = round(frac * width)
    return "█" * n + "░" * (width - n)


states = {}
# Finished tasks from sacct; queued/running ones from squeue -r (one line per task)
for line in sh(["sacct", "-j", ARRAY, "-X", "-n", "-P", "-o", "JobID,State,Elapsed"]).splitlines():
    jid, st, el = (line.split("|") + ["", "", ""])[:3]
    if "_" in jid and "[" not in jid:
        states[int(jid.split("_")[1])] = (st.split()[0] if st else "?", el)
for line in sh(["squeue", "-j", ARRAY, "-h", "-r", "-o", "%i|%T|%M"]).splitlines():
    jid, st, el = line.split("|")
    if "_" in jid and "[" not in jid:
        states[int(jid.split("_")[1])] = (st, el)

done_total, rows = 0, []
for i, (tag, nc) in enumerate(TAGS):
    st, el = states.get(i, ("UNKNOWN", ""))
    done, oom, last = 0, 0, ""
    for f in glob.glob(str(LOGS / f"fvm_scal.{ARRAY}_{i}.out")):
        for line in open(f, errors="ignore"):
            if line.startswith('{"setting"'):
                done += 1
                oom += '"status": "oom"' in line
                last = line.split('"setting": "')[1].split('"')[0]
    done_total += done
    extra = f"  last={last}" if last else ""
    extra += f"  OOM×{oom}" if oom else ""
    rows.append(f" {i}  {tag:<8}{nc:>8,} cells  [{bar(done / N_SET, N_SET)}] {done}/{N_SET}  "
                f"{st:<10}{el:>9}{extra}")

total = len(TAGS) * N_SET
print(f"R2-6 benchmark — array {ARRAY}        {datetime.now():%Y-%m-%d %H:%M:%S}")
print(f"Overall  [{bar(done_total / total, 30)}] {done_total}/{total} settings\n")
print("\n".join(rows))
start = [x for x in sh(["squeue", "-j", ARRAY, "--start", "-h", "-o", "%S"]).split() if x != "N/A"]
if start:
    print(f"\nEstimated start (pending tasks): {min(start).replace('T', ' ')}")

r24 = sh(["squeue", "-u", os.environ.get("USER", ""), "-h", "-n", "fvm_pinn",
          "-o", "%i|%T|%M|%l|%R"]).splitlines()
if r24:
    print("\nR2-4 runs (fvm_pinn): elapsed vs time limit")
    for row in r24:
        jid, st, el, lim, why = row.split("|")
        print(f"  {jid:<18} {st:<8} {el:>9} / {lim:<9} {why if st != 'RUNNING' else ''}")
