"""
Earn or die, computed after the fact from a run's CSV so cost / earn can be
calibrated without re-running the connectome.

  python3 -m matrixfly.life results/A.csv [--cost 0.002 --earn 0.02 --threshold 0.15]
  → results/A.life.csv (n, energy, alive) and one line: died at window N / survived
"""
import argparse, csv
from pathlib import Path
from .plastic import Metabolism


def run(path, cost=0.002, earn=0.02, threshold=0.15, energy=1.0):
    rows = list(csv.DictReader(open(path)))
    m = Metabolism(energy=energy, cost=cost, earn=earn, threshold=threshold)
    out = []
    for r in rows:
        m.tick(float(r["reaction"]), int(r["n"]))
        out.append({"n": r["n"], "energy": round(m.energy, 4), "alive": int(m.alive)})
    dst = Path(path).with_suffix(".life.csv")
    w = csv.DictWriter(open(dst, "w", newline=""), fieldnames=["n", "energy", "alive"]); w.writeheader(); w.writerows(out)
    return m.died_at, dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("csv", nargs="+")
    ap.add_argument("--cost", type=float, default=0.002); ap.add_argument("--earn", type=float, default=0.02); ap.add_argument("--threshold", type=float, default=0.15)
    a = ap.parse_args()
    for p in a.csv:
        died, dst = run(p, a.cost, a.earn, a.threshold)
        print(f"{p}: {'died at window ' + str(died) if died is not None else 'survived'} → {dst}")
