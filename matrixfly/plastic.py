"""
The two things this brain can change about itself. Both live around the
sensory populations, not inside flysim: the connectome stays what Janelia
traced, what changes is what the brain expects and how much it lets in.

  EchoPredictor  efference copy → sensory prediction. A delay line of the
                 brain's own behaviour (the last K windows of its descending
                 neuron scores) is regressed onto the stimulus that arrives on
                 each sensory channel. Learned online with a delta rule every
                 window. Its output is `surprise`: how far each channel's
                 stimulus was from what the brain's own past actions predicted.
                 In an open world the future is not a function of your own
                 past → surprise stays high. In a closed loop it is → surprise
                 falls. This is the corollary-discharge circuit real flies use
                 to cancel self-generated sensation (Kim et al. 2015, Fujiwara
                 2017), pointed at the world instead of the body.

  Gate           one gain per sensory channel, multiplied into the rate of
                 every stimulus before it lands on the receptor neurons.
                 Predictable input closes it a little (habituation: ORN/PN
                 adaptation, LN inhibition), surprising input opens it
                 (dishabituation), and it drifts back open very slowly when
                 nothing arrives (spontaneous recovery). A world that keeps
                 sending what the brain already expected ends up not getting in.

Both are state, saved and restored with the brain (`to_dict` / `from_dict`).
"""
from collections import deque
import numpy as np

CHANNELS = ["smell", "sugar", "touch", "sight", "wind"]
BEHAVIORS = ["walking", "feeding", "escape", "courtship", "grooming"]


class EchoPredictor:
    def __init__(self, K=64, lr=0.02, channels=CHANNELS, behaviors=BEHAVIORS, seed=0):
        self.K, self.lr = K, lr
        self.channels, self.behaviors = list(channels), list(behaviors)
        self.hist = deque([np.zeros(len(behaviors), dtype=np.float32) for _ in range(K)], maxlen=K)
        self.W = np.zeros((len(channels), len(behaviors) * K + 1), dtype=np.float32)   # +1 bias: the world's base rate
        self.pred = np.zeros(len(channels), dtype=np.float32)
        self.surprise = np.ones(len(channels), dtype=np.float32)
        self.ema = np.ones(len(channels), dtype=np.float32)     # slow surprise, what the gate reads
        self.tau = 0.02
        self.n = 0

    def _x(self):
        return np.concatenate([np.concatenate(list(self.hist)), [1.0]]).astype(np.float32)

    def push(self, scores):
        """the brain's own behaviour this window (dict behaviour → 0..1)"""
        self.hist.append(np.array([float(scores.get(b, 0.0)) for b in self.behaviors], dtype=np.float32))

    def predict(self):
        self.pred = np.clip(self.W @ self._x(), 0.0, 1.0)
        return self.pred

    def observe(self, stim):
        """
        stim: dict channel → 0..1 intensity that arrived this window (0 = nothing).
        Learns, and returns the per-channel surprise (|error|, 0..1).
        """
        x = self._x()
        s = np.array([float(stim.get(c, 0.0)) for c in self.channels], dtype=np.float32)
        err = s - self.pred
        self.W += self.lr * np.outer(err, x) / (1.0 + np.dot(x, x) * 0.01)   # normalised LMS, stays stable with K large
        self.surprise = np.abs(err)
        # the slow surprise only counts windows where there was something to predict: an arrival, or a
        # prediction that did not come true. Predicting silence when the world is silent teaches nothing.
        m = (s > 0) | (self.pred > 0.2)
        self.ema[m] = (1 - self.tau) * self.ema[m] + self.tau * self.surprise[m]
        self.n += 1
        return self.surprise

    def awareness(self):
        """1 − mean slow surprise: how much of what happens to it the brain already saw coming"""
        return float(1.0 - self.ema.mean())

    def to_dict(self):
        return {"K": self.K, "lr": self.lr, "W": self.W.tolist(), "ema": self.ema.tolist(), "n": self.n}

    @classmethod
    def from_dict(cls, d, **kw):
        p = cls(K=d["K"], lr=d["lr"], **kw)
        p.W = np.array(d["W"], dtype=np.float32); p.ema = np.array(d["ema"], dtype=np.float32); p.n = d.get("n", 0)
        return p


class Gate:
    def __init__(self, channels=CHANNELS, alpha=0.03, beta=0.10, recover=0.0005, floor=0.0, closed_below=0.05):
        self.channels = list(channels)
        self.alpha, self.beta, self.recover, self.floor = alpha, beta, recover, floor
        self.closed_below = closed_below
        self.g = np.ones(len(channels), dtype=np.float32)

    def gain(self, channel):
        try: return float(self.g[self.channels.index(channel)])
        except ValueError: return 1.0

    def update(self, stim, surprise):
        """
        stim: dict channel → intensity that arrived (before gating)
        surprise: per-channel |error| from the predictor, same order as channels
        """
        for i, c in enumerate(self.channels):
            if stim.get(c, 0.0) > 0:
                s = float(surprise[i])
                self.g[i] += self.beta * s - self.alpha * (1.0 - s)   # surprised → open, expected → close
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
    coin earns, and a brain that does nothing earns nothing (pumpbrains
    backend/src/brain/life.js). Here the same chain in minutes: every window
    costs `cost`, every window of behaviour above `threshold` earns `earn`.
    """
    def __init__(self, energy=1.0, cost=0.002, earn=0.004, threshold=0.25):
        self.energy, self.cost, self.earn, self.threshold = energy, cost, earn, threshold
        self.alive = True
        self.died_at = None

    def tick(self, scores, t):
        if not self.alive: return self.energy
        act = max(scores.values()) if scores else 0.0
        self.energy -= self.cost
        if act >= self.threshold: self.energy += self.earn * act
        self.energy = min(1.0, self.energy)
        if self.energy <= 0:
            self.energy, self.alive, self.died_at = 0.0, False, t
        return self.energy
