#!/usr/bin/env python3
"""
The fly in the real matrix: a pod on pumpbrains.com/infinite, as an external
brain (the bring-your-own-brain contract).

    PB_KEY=<brain api key> python3 -m matrixfly.runner <slug> [--no-plastic] [https://pumpbrains.com]

Every window: run the connectome, POST behaviour scores, get back every
stimulus the world injected (ring hops, mentions, trades, pokes), scale each
by the gate, land it on the sensory population, tell the predictor what the
brain did and what arrived. Death is PumpBrains' own (earn or die): a fly that
lets nothing in does nothing, earns no fees, and is switched off. Every window
is also logged to results/live-<slug>.csv with the same columns as replay.py.
"""
import csv, json, os, sys, time, urllib.request, urllib.error
from pathlib import Path
import numpy as np
from .fly import Fly, log
from .plastic import CHANNELS, BEHAVIORS

ROOT = Path(__file__).resolve().parent.parent
args = [x for x in sys.argv[1:] if not x.startswith("--")]
SLUG = args[0] if args else sys.exit("usage: runner.py <slug> [--no-plastic] [base url]")
BASE = (args[1] if len(args) > 1 else "https://pumpbrains.com").rstrip("/")
PLASTIC = "--no-plastic" not in sys.argv
KEY = os.environ.get("PB_KEY") or sys.exit("set PB_KEY to the brain's api key")
JOB_POLL_S, SAVE_S = 5.0, 300.0
DUTY = float(os.environ.get("FLY_DUTY", 0.6))


def http(method, path, body=None, key=True):
    h = {"content-type": "application/json"}
    if key: h["x-api-key"] = KEY
    req = urllib.request.Request(BASE + path, method=method, headers=h, data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


class Reward:
    """the job's outcome (trading P&L / arcade) → dopamine, as in the FLYBRAIN runner"""
    def __init__(self, fly): self.fly = fly; self.prev = None; self.next_at = 0

    def poll(self):
        if time.time() < self.next_at: return
        self.next_at = time.time() + JOB_POLL_S
        try: v = http("GET", f"/api/brains/{SLUG}/job", key=False)
        except Exception as e: return log(f"job poll: {e}")
        job = v.get("job")
        if job == "trading":
            eq = float(v.get("equity") or 0); last = (v.get("last") or {}).get("at")
            if self.prev is not None:
                d = eq - self.prev["eq"]; traded = last != self.prev["last"]
                if (traded or abs(d) >= 0.01) and abs(d) >= 0.002:
                    self.fly.dopamine(1 if d > 0 else -1, min(1.0, max(0.15, abs(d) / 0.05)), f"P&L {d:+.4f} SOL")
                if traded:
                    side = (v.get("last") or {}).get("side"); self.fly.inject("sugar" if side == "buy" else "touch", 60, 400, 0.3, "trade")
            self.prev = {"eq": eq, "last": last}
        elif job == "arcade":
            sc, deaths = int(v.get("score") or 0), int(v.get("deaths") or 0)
            if self.prev is not None:
                if sc > self.prev["sc"]: self.fly.dopamine(+1, 0.5, f"arcade score {sc}")
                if deaths > self.prev["deaths"]: self.fly.dopamine(-1, 1.0, f"arcade crash #{deaths}")
            self.prev = {"sc": sc, "deaths": deaths}


def main():
    brain = http("GET", f"/api/brains/{SLUG}", key=False)
    if brain.get("engine") != "external": sys.exit(f"{SLUG} is simulated by PumpBrains (engine {brain.get('engine')}); switch it with pumpbrains backend/scripts/set-external.js <slug> manifest.json")
    n_body = (http("GET", f"/api/species/{brain['species']}", key=False).get("core") or {}).get("neurons", 0)
    fly = Fly(brain, plastic=PLASTIC); reward = Reward(fly)
    state = ROOT / "state" / f"live-{SLUG}.json"
    if state.exists(): fly.load(state); log(f"restored self from {state}: gains {[round(x, 2) for x in fly.gate.g]}")
    out = ROOT / "results" / f"live-{SLUG}.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["n", "t_ms", "at", "cond", "life", "energy_sol", "awareness"] + [f"{k}_{c}" for k in ("arrived", "let_in", "surprise", "gain") for c in CHANNELS] + BEHAVIORS
    new = not out.exists()
    w = csv.DictWriter(open(out, "a", newline=""), fieldnames=cols)
    if new: w.writeheader()
    fly.inject("smell", 120, 3000, 1.0, "birth")
    backoff, last_save, last_text, n = 1, time.time(), None, 0
    while True:
        t0 = time.time()
        row = fly.step(n); n += 1
        fired = np.concatenate([fly.fired, fly.sniffed]) if len(fly.sniffed) else fly.fired
        if len(fired) > 1500: fired = np.random.default_rng(n).choice(fired, 1500, replace=False)
        body = [int(i % n_body) for i in fired] if n_body else []
        try:
            r = http("POST", f"/api/brains/{SLUG}/readout", {"behavior": {k: round(v, 3) for k, v in fly.scores.items()}, "energy": round(fly.energy, 3), "rates": {k: round(v, 2) for k, v in fly.rates.items()}, "spikes": fly.spikes, "t": round(fly.t), "fired": body})
            for d in r.get("drive", []):
                src = d.get("source") or "?"
                if d["channel"] == "smell" and src != "job":
                    try:
                        st = http("GET", f"/api/brains/{SLUG}/stimuli?limit=3", key=False)
                        txt = next((s.get("payload", {}).get("text") for s in (st if isinstance(st, list) else st.get("stimuli", [])) if s.get("payload", {}).get("text")), None)
                        if txt and txt != last_text: last_text = txt; fly.smell_text("", "", txt)
                    except Exception: pass
                g = fly.inject(d["channel"], d["rateHz"], d["durationMs"], d.get("fraction", 1), src)
                log(f"← {src}: {d['channel']} {d['rateHz']:.0f} Hz × {d['durationMs']} ms · gate {g:.2f}{' · SHUT OUT' if g < 0.05 else ''}")
            st = r.get("state") or {}
            rec = {"n": n, "t_ms": round(fly.t), "at": int(time.time()), "cond": "A" if PLASTIC else "D", "life": st.get("life") or brain.get("life"), "energy_sol": st.get("energy"), "awareness": round(fly.predictor.awareness(), 4)}
            for i, c in enumerate(CHANNELS):
                rec[f"arrived_{c}"] = round(row["arrived"].get(c, 0.0), 3); rec[f"let_in_{c}"] = round(row["let_in"].get(c, 0.0), 3)
                rec[f"surprise_{c}"] = round(row["surprise"][i], 4); rec[f"gain_{c}"] = round(row["gain"][i], 4)
            for b in BEHAVIORS: rec[b] = round(fly.scores[b], 3)
            w.writerow(rec)
            backoff = 1
            if n % 20 == 0:
                g = " ".join(f"{c[:3]}={fly.gate.g[i]:.2f}" for i, c in enumerate(CHANNELS))
                log(f"n={n} · awareness {fly.predictor.awareness():.3f} · gate {g} · closed {fly.gate.closed()} · act {max(fly.scores.values()):.2f} · {time.time()-t0:.2f}s/window")
        except urllib.error.HTTPError as e:
            log(f"{e.code} {e.read().decode()[:200]}"); time.sleep(backoff); backoff = min(30, backoff * 2)
        except Exception as e:
            log(f"{e}"); time.sleep(backoff); backoff = min(30, backoff * 2)
        reward.poll()
        if time.time() - last_save > SAVE_S: fly.save(state); last_save = time.time()
        busy = time.time() - t0
        if DUTY < 1: time.sleep(max(0.0, busy * (1 - DUTY) / DUTY))


if __name__ == "__main__":
    main()
