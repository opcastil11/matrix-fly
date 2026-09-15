"""
The two things this brain can change about itself. Both live around the
sensory populations, not inside flysim: the connectome stays what Janelia
traced, what changes is what the brain expects and how much it lets in.

  EchoPredictor  efference copy → sensory prediction. A delay line of the
                 brain's own behaviour events (the last K windows of what its
                 descending neurons emitted) is regressed onto the stimulus
                 that arrives on each sensory channel. Learned online with a delta rule every
                 window. Its output is `surprise`: how far each channel's
                 stimulus was from what the brain's own past actions predicted.
                 In an open world the future is not a function of your own
                 past → surprise stays high. In a closed loop it is → surprise
                 falls. This is the corollary-discharge circuit real flies use
                 to cancel self-generated sensation (Kim et al. 2015, Fujiwara
                 2017), pointed at the world instead of the body.

  Gate           one gain per sensory channel, multiplied into the rate of
                 every stimulus before it lands on the receptor neurons.
                 Input the brain's own past explains closes it a little
                 (corollary discharge cancelling self-caused sensation, then
                 habituation: ORN/PN adaptation, LN inhibition); input it
                 cannot explain opens it (dishabituation); it drifts back open
                 very slowly when nothing arrives (spontaneous recovery). A
                 world that only ever sends back what you did ends up not
                 getting in.

Both are state, saved and restored with the brain (`to_dict` / `from_dict`).
"""
from collections import deque
import numpy as np

CHANNELS = ["smell", "sugar", "touch", "sight", "wind"]
BEHAVIORS = ["walking", "feeding", "escape", "courtship", "grooming"]


class EchoPredictor:
    """
    Two heads trained side by side on the same arrivals:
      full   efference copy (the last K windows of own behaviour) + bias
      base   the channel's running mean over every window, with no idea of what the brain did
    `explained` is how much better the full head does than the base head on what
    arrived: the share of the arrival attributable to the brain's own past.
    Predicting *when* is most of it: base never knows when, full can only know
    when if the arrival follows from something the brain did.
    """
    def __init__(self, K=64, lr=0.5, channels=CHANNELS, behaviors=BEHAVIORS, seed=0):
        self.K, self.lr = K, lr
        self.channels, self.behaviors = list(channels), list(behaviors)
        self.hist = deque([np.zeros(len(behaviors), dtype=np.float32) for _ in range(K)], maxlen=K)
        self.W = np.zeros((len(channels), len(behaviors) * K + 1), dtype=np.float32)
        self.b = np.zeros(len(channels), dtype=np.float32)          # base head: running mean on arrivals
        self.pred = np.zeros(len(channels), dtype=np.float32)
        self.surprise = np.ones(len(channels), dtype=np.float32)
        self.explained = np.zeros(len(channels), dtype=np.float32)
        self.ema = np.zeros(len(channels), dtype=np.float32)        # slow `explained`, what awareness reads
        self.traffic = np.zeros(len(channels), dtype=np.float32)    # how much each channel gets, to weight awareness
        self.tau = 0.15
        self.n = 0

    def _x(self):
        return np.concatenate([np.concatenate(list(self.hist)), [1.0]]).astype(np.float32)

    def push(self, event):
        """
        the efference copy: the behaviour event the brain emitted this window,
        (behaviour, intensity) or None. Events, not standing levels — a fly
        walking in place all day emits nothing, and nothing is what the ring
        hands on for it.
        """
        v = np.zeros(len(self.behaviors), dtype=np.float32)
        if event: v[self.behaviors.index(event[0])] = float(event[1])
        self.hist.append(v)

    def predict(self):
        self.pred = np.clip(self.W @ self._x(), 0.0, 1.0)
        return self.pred

    def observe(self, stim):
        """
        stim: dict channel → 0..1 intensity that arrived this window (0 = nothing).
        Learns both heads (normalised LMS), returns per-channel surprise (|error| of the full head).
        """
        x = self._x()
        s = np.array([float(stim.get(c, 0.0)) for c in self.channels], dtype=np.float32)
        err = s - self.pred
        self.W += self.lr * np.outer(err, x) / (1.0 + float(np.dot(x, x)))
        self.surprise = np.abs(err)
        arrived = s > 0
        err_base = np.abs(s - self.b)
        self.b += 0.01 * (s - self.b)                               # base: the channel's mean over every window, timing included
        ex = np.clip(1.0 - self.surprise / (err_base + 0.05), 0.0, 1.0)
        self.explained = np.where(arrived, ex, 0.0).astype(np.float32)
        judged = arrived | (self.pred > 0.2)                        # an arrival, or a prediction that did not come true
        self.ema[judged] = (1 - self.tau) * self.ema[judged] + self.tau * self.explained[judged]
        self.traffic += 0.01 * (arrived.astype(np.float32) - self.traffic)
        self.n += 1
        return self.surprise

    def awareness(self):
        """how much of what arrives its own past explains (0 open world … 1 pure echo), channels weighted by their traffic"""
        w = self.traffic + 1e-6
        return float(np.dot(self.ema, w) / w.sum())

    def to_dict(self):
        return {"K": self.K, "lr": self.lr, "W": self.W.tolist(), "b": self.b.tolist(), "ema": self.ema.tolist(), "traffic": self.traffic.tolist(), "n": self.n}

    @classmethod
    def from_dict(cls, d, **kw):
        p = cls(K=d["K"], lr=d["lr"], **kw)
        p.W = np.array(d["W"], dtype=np.float32); p.b = np.array(d.get("b", p.b), dtype=np.float32); p.ema = np.array(d["ema"], dtype=np.float32); p.traffic = np.array(d.get("traffic", p.traffic), dtype=np.float32); p.n = d.get("n", 0)
        return p


class Gate:
    def __init__(self, channels=CHANNELS, alpha=0.08, beta=0.05, recover=0.0005, floor=0.0, closed_below=0.05):
        self.channels = list(channels)
        self.alpha, self.beta, self.recover, self.floor = alpha, beta, recover, floor
        self.closed_below = closed_below
        self.g = np.ones(len(channels), dtype=np.float32)

    def gain(self, channel):
        try: return float(self.g[self.channels.index(channel)])
        except ValueError: return 1.0

    def update(self, stim, surprise, explained):
        """
        stim: dict channel → intensity that arrived (before gating)
        surprise / explained: per channel from the predictor, same order as channels
        """
        for i, c in enumerate(self.channels):
            if stim.get(c, 0.0) > 0:
                s, e = float(surprise[i]), float(explained[i])
                self.g[i] += self.beta * s * (1.0 - e) - self.alpha * e   # caused it yourself → close; genuinely new → open
            else:
                self.g[i] += self.recover * (1.0 - self.g[i])          # nothing arrived → drift back open, slowly
        self.g = np.clip(self.g, self.floor, 1.0)
        return self.g

    def closed(self):
        return [c for i, c in enumerate(self.channels) if self.g[i] < self.closed_below]

    def to_dict(self):
        return {"g": self.g.tolist(), "alpha": self.alpha, "beta": self.beta, "recover": self.recover}

    @classmethod
    def from_dict(cls, d, **kw):
        g = cls(alpha=d["alpha"], beta=d["beta"], recover=d["recover"], **kw)
        g.g = np.array(d["g"], dtype=np.float32)
        return g


class Metabolism:
    """
    Earn or die, offline. On PumpBrains a brain lives on the creator fees its
    coin earns, and what earns is what it *does* — behaviour events (a fly
    that walks in place all day says nothing, trades nothing). Here the same
    chain in minutes: every window costs `cost`; a reaction (behaviour leaving
    its own baseline by more than `threshold`) earns `earn` × its size.
    Computed after the fact from a run's CSV (life.py), so cost/earn can be
    calibrated without re-running the connectome.
    """
    def __init__(self, energy=1.0, cost=0.002, earn=0.02, threshold=0.15):
        self.energy, self.cost, self.earn, self.threshold = energy, cost, earn, threshold
        self.alive = True
        self.died_at = None

    def tick(self, reaction, t):
        if not self.alive: return self.energy
        self.energy -= self.cost
        if reaction >= self.threshold: self.energy += self.earn * reaction
        self.energy = min(1.0, self.energy)
        if self.energy <= 0:
            self.energy, self.alive, self.died_at = 0.0, False, t
        return self.energy


class Reaction:
    """
    Behaviour leaving its own baseline (fast EMA vs slow EMA of the scores),
    and the behaviour *event* that makes: when the largest deviation crosses
    `threshold`, and not within `refractory` windows of the last one, the brain
    emits (behaviour, intensity); failing that, every `period` windows it emits
    its dominant behaviour. That event is what the ring hands on
    (hive.js fires on behaviour transitions, one hop per brain per HIVE_HOP_MS)
    and what the predictor gets as efference copy.
    """
    def __init__(self, behaviors=BEHAVIORS, fast=0.3, slow=0.01, threshold=0.15, refractory=20, period=20):
        self.behaviors, self.fast_t, self.slow_t, self.threshold, self.refractory, self.period = list(behaviors), fast, slow, threshold, refractory, period
        self.fast = None; self.slow = None; self.last = -10**9; self.n = 0
        self.react = 0.0; self.event = None

    def tick(self, scores):
        self.n += 1
        v = np.array([float(scores.get(b, 0.0)) for b in self.behaviors], dtype=np.float32)
        if self.fast is None: self.fast = v.copy(); self.slow = v.copy(); return 0.0
        self.fast += self.fast_t * (v - self.fast); self.slow += self.slow_t * (v - self.slow)
        d = np.abs(self.fast - self.slow); i = int(np.argmax(d)); self.react = float(d[i])
        self.event = None
        if self.react >= self.threshold and self.n - self.last >= self.refractory:
            self.event = (self.behaviors[i], float(v[i])); self.last = self.n
        elif self.period and self.n - self.last >= self.period:
            # nothing crossed the threshold: the ring still samples what the fly is doing (its dominant behaviour),
            # the way the runner keeps posting its readout every window and the hive picks the brain up every HIVE_HOP_MS
            j = int(np.argmax(v)); self.event = (self.behaviors[j], float(v[j])); self.last = self.n
        return self.react
