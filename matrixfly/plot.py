"""
One figure: awareness, sensory gain, behaviour and energy over time, the four
conditions overlaid.   python3 -m matrixfly.plot [results/A.csv results/B.csv …] → figs/matrix-fly.svg
Pure numpy + hand-written SVG so the repo needs no plotting library.
"""
import csv, sys
from pathlib import Path
import numpy as np
from .plastic import CHANNELS, BEHAVIORS

ROOT = Path(__file__).resolve().parent.parent
COLOR = {"A": "#ff3b3b", "B": "#2ea36f", "C": "#3b82f6", "D": "#9ca3af"}
LABEL = {"A": "A matrix (plastic)", "B": "B shuffled", "C": "C open world", "D": "D matrix, no plasticity"}


def load(path):
    rows = list(csv.DictReader(open(path)))
    f = lambda k: np.array([float(r[k]) for r in rows])
    gain = np.mean([f(f"gain_{c}") for c in CHANNELS], axis=0)
    act = np.max([f(b) for b in BEHAVIORS], axis=0)
    return {"n": f("n"), "awareness": f("awareness"), "gain": gain, "act": act, "energy": f("energy"), "alive": f("alive"), "cond": rows[0]["cond"]}


def smooth(y, k=25):
    if len(y) < k: return y
    return np.convolve(y, np.ones(k) / k, mode="same")


def panel(runs, key, title, y0, x, y, w, h, ylim=(0, 1)):
    s = [f'<text x="{x}" y="{y-8}" class="t">{title}</text>', f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="p"/>']
    nmax = max(r["n"][-1] for r in runs) or 1
    for r in runs:
        ys = smooth(r[key]) if key != "energy" else r[key]
        pts = " ".join(f"{x + w * n / nmax:.1f},{y + h - h * (min(max(v, ylim[0]), ylim[1]) - ylim[0]) / (ylim[1] - ylim[0]):.1f}" for n, v in zip(r["n"], ys))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{COLOR[r["cond"]]}" stroke-width="2"/>')
        dead = np.flatnonzero(r["alive"] == 0)
        if len(dead): s.append(f'<text x="{x + w * r["n"][dead[0]] / nmax:.1f}" y="{y + 16}" fill="{COLOR[r["cond"]]}" class="s">💀 {int(r["n"][dead[0]])}</text>')
    s.append(f'<text x="{x-6}" y="{y+12}" class="ax" text-anchor="end">{ylim[1]}</text><text x="{x-6}" y="{y+h}" class="ax" text-anchor="end">{ylim[0]}</text>')
    return "\n".join(s)


def main(paths):
    runs = [load(p) for p in paths]
    W, H, PW, PH = 900, 760, 800, 130
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           '<style>text{font:13px system-ui,sans-serif;fill:#e5e7eb}.t{font-weight:600}.s{font-size:12px}.ax{font-size:11px;fill:#9ca3af}.p{fill:#111827;stroke:#374151}</style>',
           f'<rect width="{W}" height="{H}" fill="#0b0f19"/>',
           '<text x="60" y="30" class="t" style="font-size:18px">matrix-fly — a self-modifying fly brain in a closed loop</text>']
    y = 70
    for key, title, ylim in [("awareness", "awareness = 1 − surprise on what arrives", (0, 1)), ("gain", "sensory gain (mean of the five gates)", (0, 1)), ("act", "behaviour (max descending-neuron score)", (0, 1)), ("energy", "energy (earn or die)", (0, 1))]:
        out.append(panel(runs, key, title, 0, 60, y, PW, PH, ylim)); y += PH + 40
    out.append(f'<text x="60" y="{y-10}" class="ax">windows of 40 ms biological time →</text>')
    lx = 60
    for r in runs:
        out.append(f'<rect x="{lx}" y="{y+2}" width="14" height="14" fill="{COLOR[r["cond"]]}"/><text x="{lx+20}" y="{y+14}" class="s">{LABEL[r["cond"]]}</text>'); lx += 220
    out.append("</svg>")
    dst = ROOT / "figs" / "matrix-fly.svg"; dst.parent.mkdir(exist_ok=True)
    dst.write_text("\n".join(out)); print(dst)


if __name__ == "__main__":
    main(sys.argv[1:] or [str(ROOT / "results" / f"{c}.csv") for c in "ABCD" if (ROOT / "results" / f"{c}.csv").exists()])
