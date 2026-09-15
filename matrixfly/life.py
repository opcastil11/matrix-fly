"""
Earn or die, computed after the fact from a run's CSV so cost / rate can be
calibrated without re-running the connectome.

  python3 -m matrixfly.life --calibrate results/C.csv      # prints feeding with sugar in / in silence and the cost between them
  python3 -m matrixfly.life results/A.csv [--cost 0.35 --rate 0.01]
  → results/A.life.csv (n, energy, alive) and one line: died at window N / survived
"""
import argparse, csv, statistics as st
from pathlib import Path
from .plastic import Metabolism, CHANNELS

FED, QUIET = 15, 30   # windows after sugar got in that count as "fed"; windows after any input that count as "silence"


def calibrate(path):
    """feeding with sugar getting in vs feeding in silence, on one run (use the open world, gates never close there)"""
    rows = list(csv.DictReader(open(path)))
    last = {c: -10**9 for c in CHANNELS}; fed = []; quiet = []
    for r in rows:
        n = int(r["n"])
        for c in CHANNELS:
            if float(r["let_in_" + c]) > 0: last[c] = n
        if 1 <= n - last["sugar"] <= FED: fed.append(float(r["feeding"]))
        elif all(n - last[c] > QUIET for c in CHANNELS): quiet.append(float(r["feeding"]))
    f, q = (st.mean(fed) if fed else float("nan")), (st.mean(quiet) if quiet else float("nan"))
    return {"fed": f, "n_fed": len(fed), "quiet": q, "n_quiet": len(quiet), "cost": (f + q) / 2}


def run(path, cost=0.35, rate=0.01, energy=1.0):
    rows = list(csv.DictReader(open(path)))
    m = Metabolism(energy=energy, cost=cost, rate=rate)
    out = []
    for r in rows:
        m.tick(float(r["feeding"]), int(r["n"]))
        out.append({"n": r["n"], "energy": round(m.energy, 4), "alive": int(m.alive)})
    dst = Path(path).with_suffix(".life.csv")
    w = csv.DictWriter(open(dst, "w", newline=""), fieldnames=["n", "energy", "alive"]); w.writeheader(); w.writerows(out)
    return m.died_at, dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("csv", nargs="+")
    ap.add_argument("--calibrate", action="store_true", help="only measure feeding fed / in silence on the given runs")
    ap.add_argument("--cost", type=float, default=0.35); ap.add_argument("--rate", type=float, default=0.01)
    a = ap.parse_args()
    for p in a.csv:
        if a.calibrate:
            c = calibrate(p); print(f"{p}: feeding {c['fed']:.3f} with sugar in ({c['n_fed']} windows), {c['quiet']:.3f} in silence ({c['n_quiet']}) → cost {c['cost']:.3f}")
            continue
        died, dst = run(p, a.cost, a.rate)
        print(f"{p}: {'died at window ' + str(died) if died is not None else 'survived'} → {dst}")
