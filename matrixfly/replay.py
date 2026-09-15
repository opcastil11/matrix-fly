"""
Run the fly in an offline world and log every window to a CSV.

  python3 -m matrixfly.replay A --windows 3000            # matrix, plastic
  python3 -m matrixfly.replay D --windows 3000            # matrix, no plasticity (control)
  python3 -m matrixfly.replay B --from results/A.json     # same arrivals as A, shuffled
  python3 -m matrixfly.replay C --windows 3000            # open world

Columns: n, t_ms, cond, awareness, reaction, then per channel arrived_<c>, let_in_<c>,
surprise_<c>, explained_<c>, gain_<c>, then per behaviour <b>. Energy / death: life.py.
"""
import argparse, csv, json, time
from pathlib import Path
from .fly import Fly, log
from .plastic import CHANNELS, BEHAVIORS
from . import ring

ROOT = Path(__file__).resolve().parent.parent
CONDITIONS = {"A": "matrix", "B": "shuffled", "C": "open", "D": "matrix, no plasticity"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cond", choices=list(CONDITIONS))
    ap.add_argument("--windows", type=int, default=3000)
    ap.add_argument("--log", default=str(ROOT / "data" / "ring-log.jsonl"))
    ap.add_argument("--from", dest="from_run", help="B: results/<run>.json of the run whose arrivals to shuffle")
    ap.add_argument("--lag", type=int, default=40); ap.add_argument("--mix", type=float, default=0.7)
    ap.add_argument("--K", type=int, default=64); ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.08); ap.add_argument("--beta", type=float, default=0.05); ap.add_argument("--recover", type=float, default=0.0005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None, help="state/<cond>.json of a finished run: continue that brain, appending to --out from --start")
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()

    hops = ring.load_log(a.log) if Path(a.log).exists() else []
    log(f"condition {a.cond} ({CONDITIONS[a.cond]}) · {len(hops)} real hops in the log")
    plastic = a.cond != "D"
    fly = Fly(plastic=plastic, K=a.K, lr=a.lr, alpha=a.alpha, beta=a.beta, recover=a.recover, seed=a.seed)
    if a.cond in ("A", "D"): world = ring.Matrix(hops, lag=a.lag, mix=a.mix, seed=a.seed)
    elif a.cond == "B":
        src = json.load(open(a.from_run or ROOT / "results" / "A.json"))
        world = ring.Shuffled([(n, [tuple(s) for s in st]) for n, st in src["arrivals"]], seed=a.seed)
        a.windows = min(a.windows, len(src["arrivals"]))
    else: world = ring.Open(hops, seed=a.seed)

    out = Path(a.out or ROOT / "results" / f"{a.cond}.csv"); out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["n", "t_ms", "cond", "awareness", "reaction"] + [f"{k}_{c}" for k in ("arrived", "let_in", "surprise", "explained", "gain") for c in CHANNELS] + BEHAVIORS
    if a.resume:
        fly.load(a.resume); log(f"resumed plastic state from {a.resume}, continuing at window {a.start}")
        fh = open(out, "a", newline=""); w = csv.DictWriter(fh, fieldnames=cols)
    else:
        fh = open(out, "w", newline=""); w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        fly.inject("smell", 120, 3000, 1.0, "birth")
    t0 = time.time()
    for n in range(a.start, a.start + a.windows):
        for (c, hz, ms, frac, src) in world.tick(fly, n):
            fly.inject(c, hz, ms, frac, src)
        row = fly.step(n)
        r = {"n": n, "t_ms": round(fly.t), "cond": a.cond, "awareness": round(fly.predictor.awareness(), 4), "reaction": round(row["reaction"], 4)}
        for i, c in enumerate(CHANNELS):
            r[f"arrived_{c}"] = round(row["arrived"].get(c, 0.0), 3); r[f"let_in_{c}"] = round(row["let_in"].get(c, 0.0), 3)
            r[f"surprise_{c}"] = round(row["surprise"][i], 4); r[f"explained_{c}"] = round(row["explained"][i], 4); r[f"gain_{c}"] = round(row["gain"][i], 4)
        for b in BEHAVIORS: r[b] = round(fly.scores[b], 3)
        w.writerow(r); fh.flush()
        if n % 50 == 0:
            g = " ".join(f"{c[:3]}={fly.gate.g[i]:.2f}" for i, c in enumerate(CHANNELS))
            log(f"{a.cond} n={n} · awareness {fly.predictor.awareness():.3f} · gate {g} · closed {fly.gate.closed()} · react {row['reaction']:.2f} · {(time.time()-t0)/(n-a.start+1):.2f}s/window")
    if a.resume and out.with_suffix(".json").exists():
        prev = json.load(open(out.with_suffix(".json"))); world.arrivals = [tuple(x) for x in prev.get("arrivals", [])] + world.arrivals
    summary = {"cond": a.cond, "windows": n + 1, "awareness": fly.predictor.awareness(), "gain": fly.gate.g.tolist(), "closed": fly.gate.closed(), "arrivals": world.arrivals, "args": vars(a)}
    json.dump(summary, open(out.with_suffix(".json"), "w"))
    fly.save(ROOT / "state" / f"{a.cond}.json")
    log(f"done → {out} · awareness {summary['awareness']:.3f} · gains {[round(x, 2) for x in summary['gain']]}")


if __name__ == "__main__":
    main()
