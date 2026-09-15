"""
The fly: flycoinrh's connectome (165,122 traced neurons of the FlyEM male CNS
as a leaky integrate-and-fire network, DoOR nose, dopamine-gated mushroom
body) with the two plastic pieces from plastic.py wrapped around its senses.

This is the FLYBRAIN runner from pumpbrains/hackathon/flybrain with the
network I/O taken out: `inject` is where the world gets in, `step` is one
window of biological time, `scores` is what the descending neurons decided.
With plasticity on, every stimulus is scaled by the gate before it reaches
the receptor neurons, and every window the predictor is told what the brain
did and what arrived.
"""
import os, sys, math, time, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
VENDOR = Path(os.environ.get("FLY_REPO") or HERE.parent / "vendor" / "flycoinrh")
sys.path.insert(0, str(VENDOR))
os.environ.setdefault("FLY_STATE_DIR", os.environ.get("FLY_STATE_DIR") or str(HERE.parent / "state"))

from flysim import FlyBrain                      # noqa: E402
from olfaction import Nose, MAX_HZ               # noqa: E402
from mushroom import MushroomBody                # noqa: E402
import calibration                               # noqa: E402
from .plastic import EchoPredictor, Gate, CHANNELS, BEHAVIORS  # noqa: E402

WINDOW_MS = float(os.environ.get("FLY_WINDOW_MS", 40))     # biological ms per window (state carried)
SNIFF_MS = 20.0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Fly:
    def __init__(self, brain=None, plastic=True, K=64, lr=0.02, alpha=0.03, beta=0.10, recover=0.0005, seed=0):
        brain = brain or {"name": "MATRIXFLY", "symbol": "MATRIX", "description": "a fruit fly brain that can rewrite what it lets in"}
        t = time.time()
        self.fb = FlyBrain(VENDOR / "build" / "graph.npz")
        self.nose = Nose(self.fb, root=VENDOR, equal_sniff=2.0)
        self.mb = MushroomBody(self.fb, calibration=calibration.CHOSEN, sides=VENDOR / "build" / "mb_sides.json")
        self.gains = calibration.gains_for(self.fb, calibration.CHOSEN)
        self.mb.apply()
        fb = self.fb
        sc, sub = fb.superclass, fb.subclass
        pick = lambda m: np.flatnonzero(m)
        self.senses = {
            "smell": fb.where(type_re=r"^ORN_"),
            "sugar": pick((sc == "cb_sensory") & np.isin(sub, ["labellar bristle", "taste peg", "pharyngeal sensillum"]) | (sc == "vnc_sensory") & (sub == "taste bristle")),
            "touch": pick((sc == "vnc_sensory") & np.isin(sub, ["mechanosensory bristle", "leg bristle", "wing bristle"])),
            "sight": fb.where(type_re=r"^LC4$|^LPLC2$"),
            "wind": pick((sc == "cb_sensory") & np.isin(sub, ["wind_gravity", "auditory"])),
        }
        self.rec = {
            "walking": fb.where(type_re=r"^DNa0[12]$|^DNp09$"),
            "escape": fb.where(type_re=r"^DNp0[1246]$|^DNp11$|^MDN$"),
            "courtship": fb.where(type_re=r"^pC1|^aSP|^P1_"),
            "grooming": fb.where(type_re=r"^DNg1[12]|^aDN|^DNg08"),
            "kc": self.mb.kc, "orn": self.senses["smell"], "pam": fb.where(type_re=r"^PAM"), "ppl1": fb.where(type_re=r"^PPL1"),
        }
        self.rec.update(calibration.readout(self.mb))
        self.own_smell = self.nose.smell(brain.get("name", ""), brain.get("symbol", ""), brain.get("description", ""))
        self.smell_now = self.own_smell
        self.state = None
        self.t = 0.0
        self.active = []
        self.scores = {b: 0.0 for b in BEHAVIORS}
        self.rates = {}
        self.spikes = 0
        self.energy = 0.05
        self.lean = 0.0; self.lean_ref = None
        self.fired = np.array([], dtype=np.int64); self.sniffed = self.fired
        self.kc_hz = 0.0; self.mbon = (0.0, 0.0)
        self.rng = np.random.default_rng(seed)
        # the plastic part
        self.plastic = plastic
        self.predictor = EchoPredictor(K=K, lr=lr if plastic else 0.0)
        self.gate = Gate(alpha=alpha if plastic else 0.0, beta=beta if plastic else 0.0, recover=recover if plastic else 0.0)
        self.arrived = {}          # channel → max raw intensity this window (before the gate)
        self.let_in = {}           # channel → intensity after the gate
        log(f"connectome {fb.n:,} neurons · {len(fb.wdata):,} edges · load {time.time()-t:.1f}s · plastic={plastic}")

    # ── the world gets in here ──
    def inject(self, channel, rate_hz, duration_ms, fraction=1.0, source=None):
        idx = self.senses.get(channel)
        if idx is None or not len(idx): return 0.0
        raw = min(float(rate_hz), MAX_HZ) / MAX_HZ
        self.arrived[channel] = max(self.arrived.get(channel, 0.0), raw)
        g = self.gate.gain(channel)
        rate = raw * MAX_HZ * g * max(0.0, min(1.0, float(fraction) if fraction is not None else 1.0))
        self.let_in[channel] = max(self.let_in.get(channel, 0.0), rate / MAX_HZ)
        if rate < 0.5: return g                       # gated out: the receptor neurons never hear it
        if channel == "smell":
            self.active.append({"channel": channel, "rate": rate, "drive": {k: v * rate / MAX_HZ for k, v in self.nose.drive(self.smell_now).items()}, "until": self.t + duration_ms, "source": source})
        else:
            k = max(1, int(len(idx) * max(0.05, float(fraction or 1.0))))
            sub = np.random.default_rng(int(self.t)).choice(idx, size=min(k, len(idx)), replace=False)
            self.active.append({"channel": channel, "rate": rate, "drive": {tuple(sub.tolist()): rate}, "until": self.t + duration_ms, "source": source})
        return g

    def smell_text(self, name, symbol, text):
        s = self.nose.smell(name or "", symbol or "", text or "")
        if s["profile"]: self.smell_now = s
        return s

    def sniff(self, n):
        smell = next((a for a in self.active if a["channel"] == "smell"), None)
        drive = smell["drive"] if smell else {k: v * 0.4 for k, v in self.nose.drive(self.own_smell).items()}
        out = self.fb.run(drive, int(SNIFF_MS / self.fb.p.dt), gains=self.gains, record={"kc": self.mb.kc, "approach": self.rec["approach"], "avoid": self.rec["avoid"]}, state=None, seed=n)
        fired = out["_fired"]
        self.mb.observe(fired)
        r = calibration.syn_drive(self.mb, fired)
        if calibration.has_reading(r):
            lean = calibration.leaning(r)
            self.lean_ref = lean if self.lean_ref is None else self.lean_ref * 0.99 + lean * 0.01
            self.lean = self.lean * 0.5 + lean * 0.5
        self.kc_hz = float(np.mean(out["kc"])); self.mbon = (float(np.mean(out["approach"])), float(np.mean(out["avoid"])))
        return fired

    # ── one window ──
    def step(self, n):
        self.active = [a for a in self.active if a["until"] > self.t]
        drive = {}
        for a in self.active:
            for k, v in a["drive"].items(): drive[k] = max(drive.get(k, 0.0), v)
        amb = tuple(self.senses["wind"][::7].tolist()); drive[amb] = max(drive.get(amb, 0.0), 4.0)
        out = self.fb.run(drive, int(WINDOW_MS / self.fb.p.dt), gains=self.gains, record=self.rec, state=self.state)
        self.state = out["_state"]; self.t += WINDOW_MS
        self.fired = out["_fired"]
        self.spikes += int(out["_spikes_per_sec"] * WINDOW_MS / 1000)
        if n % 2 == 0: self.sniffed = self.sniff(n)
        self.mb.forget()
        hz = {k: float(np.mean(out[k])) if len(out[k]) else 0.0 for k in self.rec}
        want = {"walking": hz["walking"] / 150, "escape": hz["escape"] / 80, "courtship": hz["courtship"] / 80, "grooming": hz["grooming"] / 200}
        appetite = 1 / (1 + math.exp(-12 * (self.lean - (self.lean_ref or self.lean)))) if self.lean_ref is not None else 0.5
        sugar = max([a["rate"] for a in self.active if a["channel"] == "sugar"], default=0.0) / MAX_HZ
        want["feeding"] = max(0.0, (appetite - 0.5) * 2) + 0.7 * sugar
        a = 0.5
        for b in self.scores: self.scores[b] = min(1.0, max(0.0, self.scores[b] * (1 - a) + want[b] * a))
        mean_hz = out["_total_hz"]
        self.energy = min(1.0, max(0.02, self.energy * 0.6 + 0.4 * min(1.0, mean_hz / 50)))
        self.rates = {"ORN": hz["orn"], "KC": self.kc_hz, "MBON+": self.mbon[0], "MBON-": self.mbon[1], "PAM": hz["pam"], "PPL1": hz["ppl1"], "DNa02": hz["walking"], "GF": hz["escape"], "pC1": hz["courtship"], "DNg": hz["grooming"], "all": mean_hz, "lean": self.lean}
        # plasticity: what arrived this window vs what its own past predicted, then the gate
        surprise = self.predictor.observe(self.arrived)
        self.gate.update(self.arrived, surprise)
        self.predictor.push(self.scores)
        self.predictor.predict()
        row = {"arrived": dict(self.arrived), "let_in": dict(self.let_in), "surprise": surprise.tolist(), "gain": self.gate.g.tolist()}
        self.arrived, self.let_in = {}, {}
        return row

    def dopamine(self, valence, amount, why):
        hit = self.mb.dopamine(valence, amount); self.mb.apply()
        st = self.mb.stats()
        log(f"{'🍬' if valence > 0 else '⚡'} dopamine {'+' if valence > 0 else '-'}{amount:.2f} · {why} · {hit} KC→MBON synapses depressed · mean gain {st.get('mean_gain')}")
        return hit

    # ── the self it rewrites, on disk ──
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"predictor": self.predictor.to_dict(), "gate": self.gate.to_dict(), "t": self.t}, open(path, "w"))
        self.mb.save()

    def load(self, path):
        d = json.load(open(path))
        self.predictor = EchoPredictor.from_dict(d["predictor"]); self.gate = Gate.from_dict(d["gate"]); self.t = d.get("t", 0.0)
