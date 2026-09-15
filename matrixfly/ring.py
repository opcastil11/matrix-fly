"""
Offline worlds the fly can be put in. Each one is called once per window and
returns the stimuli that arrive: [(channel, rate_hz, duration_ms, fraction, source)].

  Matrix    the closed loop. What the fly did `lag` windows ago comes back as
            a sensation (feeding → sugar, escape → wind, walking/grooming →
            touch, courtship → wind (song, Johnston's organ), arousal → smell),
            the way pumpbrains backend/src/brain/hive.js hands one brain's
            behaviour to the next. Between the fly and its own echo sit 92
            other flies, so the echo is mixed with the ring's own traffic:
            `mix` of what arrives is the echo, the rest is drawn from a
            recorded log of real hops (data/ring-log.jsonl).
  Shuffled  the same arrivals as a recorded Matrix run, in shuffled order:
            identical statistics, no relation to what the fly does.
  Open      an open world: independent Poisson arrivals with the real log's
            channel and intensity marginals, at the same rate as the ring.

Hop → stimulus is the real mapping (stimulus.js fromPoke):
  rateHz = 40 + 180·k, durationMs = 300 + 900·k, fraction = 0.3 + 0.7·k, k = intensity.
"""
import json, random
from collections import deque

# hive channel → the fly's sensory population (the FLYBRAIN manifest's senses)
HIVE_TO_SENSE = {"sugar": "sugar", "wind": "wind", "bristle": "touch", "taste_peg": "touch", "auditory": "wind", "pheromone": "smell", "smell": "smell", "touch": "touch", "sight": "sight"}
BEHAVIOR_TO_HIVE = {"feeding": "sugar", "escape": "wind", "walking": "bristle", "grooming": "taste_peg", "courtship": "auditory", "arousal": "pheromone"}


def poke(channel, k, source):
    k = max(0.05, min(1.0, float(k)))
    return (channel, 40 + 180 * k, 300 + 900 * k, 0.3 + 0.7 * k, source)


def load_log(path):
    hops = []
    for line in open(path):
        line = line.strip()
        if not line: continue
        h = json.loads(line)
        if h.get("type", "hop") != "hop": continue
        hops.append({"channel": HIVE_TO_SENSE.get(h["channel"], "touch"), "intensity": float(h.get("intensity", 0.5)), "behavior": h.get("behavior")})
    return hops


class World:
    def __init__(self, seed=0): self.rng = random.Random(seed); self.arrivals = []
    def tick(self, fly, n): raise NotImplementedError
    def _emit(self, n, stim):
        self.arrivals.append((n, stim)); return stim


class Matrix(World):
    def __init__(self, log, lag=40, hop_every=20, threshold=0.25, mix=0.7, traffic_every=12, seed=0):
        super().__init__(seed)
        self.log, self.lag, self.hop_every, self.threshold, self.mix, self.traffic_every = log, lag, hop_every, threshold, mix, traffic_every
        self.delay = deque([None] * lag, maxlen=lag)     # behaviour handed on `lag` windows ago
        self.last_hop = -10**9

    def tick(self, fly, n):
        # what the fly hands on this window (one hop per hop_every windows, like HIVE_HOP_MS)
        handed = None
        b, s = max(fly.scores.items(), key=lambda kv: kv[1])
        if s >= self.threshold and n - self.last_hop >= self.hop_every:
            handed = (b, s); self.last_hop = n
        self.delay.append(handed)
        back = self.delay[0]
        out = []
        if back is not None:
            b, s = back
            if self.rng.random() < self.mix:
                out.append(poke(HIVE_TO_SENSE[BEHAVIOR_TO_HIVE[b]], s, "echo"))
            elif self.log:
                h = self.rng.choice(self.log); out.append(poke(h["channel"], h["intensity"], "ring"))
        elif self.log and self.traffic_every and n % self.traffic_every == 0 and self.rng.random() < 0.5:
            h = self.rng.choice(self.log); out.append(poke(h["channel"], h["intensity"], "ring"))
        return self._emit(n, out)


class Shuffled(World):
    """replays another world's arrivals in shuffled order"""
    def __init__(self, arrivals, seed=0):
        super().__init__(seed)
        stims = [s for _, s in arrivals if s]
        self.rng.shuffle(stims)
        slots = sorted(self.rng.sample(range(len(arrivals)), len(stims))) if arrivals else []
        self.plan = dict(zip(slots, stims))
        self.n_total = len(arrivals)

    def tick(self, fly, n):
        return self._emit(n, [(c, r, d, f, "shuffled") for (c, r, d, f, _) in self.plan.get(n % max(1, self.n_total), [])])


class Open(World):
    def __init__(self, log, rate=1 / 20, seed=0):
        super().__init__(seed); self.log, self.rate = log, rate

    def tick(self, fly, n):
        out = []
        if self.log and self.rng.random() < self.rate:
            h = self.rng.choice(self.log); out.append(poke(h["channel"], h["intensity"], "world"))
        return self._emit(n, out)
