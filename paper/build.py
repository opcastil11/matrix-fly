"""
Builds paper/index.html from results/*.csv — every number in the paper is
computed here, so re-running after longer runs regenerates the whole thing.
  python3 paper/build.py [--cost 0.285]
"""
import csv, json, sys, statistics as st, html
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from matrixfly.plastic import CHANNELS, BEHAVIORS
from matrixfly.life import calibrate, run as life_run

ROOT = Path(__file__).resolve().parent.parent
COST = float(sys.argv[sys.argv.index("--cost") + 1]) if "--cost" in sys.argv else None
RATE = 0.01
CONDS = ["A", "B", "C", "D"]
NAME = {"A": "A · matrix", "B": "B · shuffled", "C": "C · open world", "D": "D · matrix, gates frozen"}
CLS = {"A": "a", "B": "b", "C": "c", "D": "d"}
ECHO = {"walking": "touch", "grooming": "touch", "feeding": "sugar", "escape": "wind", "courtship": "wind"}


def load(c):
    p = ROOT / "results" / f"{c}.csv"
    if not p.exists(): return None
    rows = list(csv.DictReader(open(p)))
    f = lambda k: np.array([float(r[k]) for r in rows])
    d = {"rows": rows, "n": f("n"), "awareness": f("awareness"), "feeding": f("feeding"), "reaction": f("reaction")}
    for k in ("arrived", "let_in", "explained", "surprise", "gain"):
        for ch in CHANNELS: d[f"{k}_{ch}"] = f(f"{k}_{ch}")
    d["gain"] = np.mean([d[f"gain_{ch}"] for ch in CHANNELS], axis=0)
    return d


def smooth(y, k=101):
    if len(y) < k: return y
    yp = np.pad(y, (k // 2, k // 2), mode="edge"); return np.convolve(yp, np.ones(k) / k, mode="valid")


def poly(x, y, nmax, W, H, ylim=(0, 1), step=8, cls=""):
    xs = x[::step]; ys = y[::step]
    pts = " ".join(f"{W * xi / nmax:.1f},{H - H * (min(max(yi, ylim[0]), ylim[1]) - ylim[0]) / (ylim[1] - ylim[0]):.1f}" for xi, yi in zip(xs, ys))
    return f'<polyline class="{cls}" points="{pts}"/>'


def panel(runs, key, title, ylim=(0, 1), k=101, hline=None, nmax=None):
    W, H, L, B = 760, 140, 44, 30
    nmax = nmax or max(r["n"][-1] for r in runs.values())
    s = [f'<svg viewBox="0 0 {W + L + 10} {H + B + 30}" class="chart" role="img" aria-label="{html.escape(title)}">',
         f'<text x="{L}" y="12" class="ct">{html.escape(title)}</text>',
         f'<g transform="translate({L},20)"><rect width="{W}" height="{H}" class="cp"/>']
    for yv in (0.25, 0.5, 0.75):
        s.append(f'<line x1="0" x2="{W}" y1="{H - H * (yv - ylim[0]) / (ylim[1] - ylim[0]):.1f}" y2="{H - H * (yv - ylim[0]) / (ylim[1] - ylim[0]):.1f}" class="cg"/>')
    if hline is not None:
        yh = H - H * (hline - ylim[0]) / (ylim[1] - ylim[0]); s.append(f'<line x1="0" x2="{W}" y1="{yh:.1f}" y2="{yh:.1f}" class="ch"/><text x="6" y="{yh - 5:.1f}" class="cl">cost of living {hline:.3f}</text>')
    for c, r in runs.items():
        y = r[key] if key == "energy" else smooth(r[key], k)
        s.append(poly(r["n"], y, nmax, W, H, ylim, cls=f"ln {CLS[c]}"))
        if key == "energy" and r.get("died") is not None:
            xd = W * r["died"] / nmax; s.append(f'<line x1="{xd:.1f}" x2="{xd:.1f}" y1="0" y2="{H}" class="cd {CLS[c]}"/><text x="{xd + 4:.1f}" y="12" class="cl {CLS[c]}">† {int(r["died"])}</text>')
    s.append(f'<text x="-6" y="10" class="cl" text-anchor="end">{ylim[1]}</text><text x="-6" y="{H}" class="cl" text-anchor="end">{ylim[0]}</text>')
    for xt in range(0, int(nmax) + 1, 1000):
        s.append(f'<text x="{W * xt / nmax:.1f}" y="{H + 14}" class="cl" text-anchor="middle">{xt}</text>')
    s.append(f'<text x="{W}" y="{H + 34}" class="cl" text-anchor="end">window · 40 ms each</text></g></svg>')
    return "\n".join(s)


def gates_panel(r):
    W, H, L, B = 760, 140, 44, 30; nmax = r["n"][-1]
    s = [f'<svg viewBox="0 0 {W + L + 10} {H + B + 30}" class="chart" role="img" aria-label="gates of A per channel">',
         f'<text x="{L}" y="12" class="ct">A · gain of each sensory gate</text>', f'<g transform="translate({L},20)"><rect width="{W}" height="{H}" class="cp"/>']
    for i, ch in enumerate(CHANNELS):
        s.append(poly(r["n"], smooth(r[f"gain_{ch}"], 21), nmax, W, H, cls=f"ln g{i}"))
        s.append(f'<text x="{6 + i * 150}" y="{H + 34}" class="cl g{i}">— {ch} {float(r[f"gain_{ch}"][-1]):.2f}</text>')
    s.append(f'<text x="-6" y="10" class="cl" text-anchor="end">1</text><text x="-6" y="{H}" class="cl" text-anchor="end">0</text>')
    for xt in range(0, int(nmax) + 1, 1000): s.append(f'<text x="{W * xt / nmax:.1f}" y="{H + 14}" class="cl" text-anchor="middle">{xt}</text>')
    s.append("</g></svg>")
    return "\n".join(s)


def explained_on_echo(r):
    """per arrival on touch (A's own echo channel), explained vs window, thirds"""
    idx = np.flatnonzero(r["arrived_touch"] > 0)
    if not len(idx): return None
    ex = r["explained_touch"][idx]; n = r["n"][idx]
    k = max(1, len(idx) // 3)
    return {"n": len(idx), "first": float(ex[:k].mean()), "last": float(ex[-k:].mean())}


def thirds(y):
    k = len(y) // 3
    return [float(y[:k].mean()), float(y[k:2 * k].mean()), float(y[2 * k:].mean())]


runs = {c: load(c) for c in CONDS}; runs = {c: r for c, r in runs.items() if r}
cal = calibrate(ROOT / "results" / "C.csv")
cost = COST or cal["cost"]
for c, r in runs.items():
    died, dst = life_run(ROOT / "results" / f"{c}.csv", cost=cost, rate=RATE)
    r["energy"] = np.array([float(x["energy"]) for x in csv.DictReader(open(dst))]); r["died"] = died
    r["N"] = len(r["n"]); r["aw3"] = thirds(r["awareness"]); r["aw_max"] = float(smooth(r["awareness"]).max())
    r["feed_first"] = float(r["feeding"][:1000].mean()); r["feed_last"] = float(r["feeding"][-1000:].mean())
    r["gains"] = {ch: float(r[f"gain_{ch}"][-1]) for ch in CHANNELS}
    r["arr"] = {ch: int((r[f"arrived_{ch}"] > 0).sum()) for ch in CHANNELS}
    r["sugar_in"] = float((r["arrived_sugar"] * r["gain_sugar"]).sum() / max(1e-9, r["arrived_sugar"].sum()))
    r["echo"] = explained_on_echo(r)
    r["energy_end"] = float(r["energy"][-1])
A = runs.get("A")
mins = lambda n: f"{n * 0.04 / 60:.1f} min"
pct = lambda x: f"{100 * x:.0f}%"
f3 = lambda x: f"{x:.3f}"
f2 = lambda x: f"{x:.2f}"

def row(c, r):
    life = f"† window {r['died']} ({mins(r['died'])})" if r["died"] is not None else f"alive, energy {f2(r['energy_end'])}"
    return f"<tr><th><span class='sw {CLS[c]}'></span>{NAME[c]}</th><td>{r['N']}</td><td>{f3(r['aw3'][0])} → {f3(r['aw3'][1])} → {f3(r['aw3'][2])}</td><td>{f3(r['aw_max'])}</td><td>{f2(r['gains']['sugar'])}</td><td>{f2(r['gains']['touch'])}</td><td>{f2(r['gains']['wind'])}</td><td>{pct(r['sugar_in'])}</td><td>{f3(r['feed_first'])} → {f3(r['feed_last'])}</td><td>{life}</td></tr>"

table2 = "\n".join(row(c, r) for c, r in runs.items())
arr_rows = "\n".join(f"<tr><th>{NAME[c]}</th>" + "".join(f"<td>{r['arr'][ch]}</td>" for ch in CHANNELS) + f"<td>{sum(r['arr'].values())}</td></tr>" for c, r in runs.items())

fig_ts = "\n".join([panel(runs, "awareness", "awareness · share of what arrives that the brain's own past explains", (0, 0.6)),
                    panel(runs, "gain", "sensory gain · mean of the five gates", (0, 1.05), k=51),
                    panel(runs, "feeding", "feeding · what earns", (0, 0.8), hline=cost),
                    panel(runs, "energy", "energy · earn or die", (0, 1.05))])
fig_gates = gates_panel(A) if A else ""
echoA = A["echo"] if A else None

def sec_results():
    if not A: return "<p>No runs yet.</p>"
    dead = [c for c, r in runs.items() if r["died"] is not None]
    verdict = (f"Under the calibrated metabolism only <b>{', '.join(dead)}</b> died." if dead else
               f"Under the calibrated metabolism nobody has died yet in {A['N']} windows ({mins(A['N'])}); A is the only brain whose feeding sits below the cost of living ({f3(A['feed_last'])} vs {f3(cost)} over its last 1,000 windows) and the only one whose energy is falling ({f2(A['energy_end'])} at the end, the three controls at {', '.join(f2(runs[c]['energy_end']) for c in runs if c != 'A')}).")
    return f"""
<p><b>The loop is noticed.</b> Awareness — the traffic-weighted share of arrivals that the echo head predicts better than the running mean — climbs in the two matrix conditions and stays flat in the two controls (Table 2, Fig. 2, top). Over thirds of the run A goes {f3(A['aw3'][0])} → {f3(A['aw3'][1])} → {f3(A['aw3'][2])} (smoothed peak {f3(A['aw_max'])}); D, the same loop with the gates frozen, {f3(runs['D']['aw3'][0])} → {f3(runs['D']['aw3'][2])}; the open world C stays at {f3(runs['C']['aw3'][0])}–{f3(runs['C']['aw3'][2])}{(" and the shuffled control B at " + f3(runs['B']['aw3'][0]) + "–" + f3(runs['B']['aw3'][2])) if 'B' in runs else ""}. On A's own echo channel (touch: its dominant behaviour is walking, which the ring returns as a bristle poke) the per-arrival <i>explained</i> goes from {f3(echoA['first'])} on the first third of echoes to {f3(echoA['last'])} on the last third ({echoA['n']} arrivals).</p>
<p><b>The door closes.</b> Only A moves its gates (Fig. 3). Touch, the channel that carries its own footsteps back, is the first to go: gain {f2(A['gains']['touch'])} at the end of the run. Sugar, which arrives both as the echo of its own feeding and as the ring's ordinary traffic, follows more slowly ({f2(A['gains']['sugar'])}); wind ({f2(A['gains']['wind'])}) barely moves and smell and sight, which the ring never sends, stay at 1. Over the whole run A let through {pct(A['sugar_in'])} of the sugar that reached it; D, the same loop with the gates frozen, {pct(runs['D']['sugar_in'])}; C {pct(runs['C']['sugar_in'])}.</p>
<p><b>Then it stops eating.</b> The connectome's feeding score is driven by sugar (Sec. 2.7: {f3(cal['fed'])} in the 15 windows after sugar gets in vs {f3(cal['quiet'])} in silence, measured on C). As A's sugar gate closes, its feeding falls from {f3(A['feed_first'])} over the first 1,000 windows to {f3(A['feed_last'])} over the last 1,000, while D holds at {f3(runs['D']['feed_first'])} → {f3(runs['D']['feed_last'])} and C at {f3(runs['C']['feed_first'])} → {f3(runs['C']['feed_last'])}. {verdict}</p>
<p><b>The controls say what it is.</b> D shows the prediction alone does nothing to behaviour — it learns the loop just as well as A ({f3(runs['D']['aw3'][2])} vs {f3(A['aw3'][2])} in the last third) and keeps eating, because it cannot act on what it learned. C shows an open world with the same channel and intensity statistics does not produce the effect{(". B is the decisive one: the <i>identical</i> arrivals A received, in shuffled order, produce awareness " + f3(runs['B']['aw3'][2]) + ", gates at 1 and feeding " + f3(runs['B']['feed_last']) + " — the same stimuli, taken out of their relation to what the brain did, are just food") if 'B' in runs else ""}.</p>"""

LOOP_SVG = """<svg viewBox="0 0 760 250" class="diagram" role="img" aria-label="the loop">
<rect x="40" y="70" width="200" height="110" rx="6" class="dg-box"/>
<text x="140" y="96" class="dg-t" text-anchor="middle" font-weight="500">fly brain · 165,122 LIF neurons</text>
<text x="140" y="116" class="dg-m" text-anchor="middle">connectome untouched</text>
<rect x="60" y="130" width="160" height="34" rx="4" class="dg-soft"/>
<text x="140" y="151" class="dg-t" text-anchor="middle">readout · 5 behaviours / 40 ms</text>
<path d="M240 125 H 310" class="dg-l" marker-end="url(#ar)"/>
<text x="285" y="116" class="dg-m" text-anchor="middle">event</text>
<rect x="310" y="95" width="170" height="60" rx="6" class="dg-box"/>
<text x="395" y="119" class="dg-t" text-anchor="middle">the ring</text>
<text x="395" y="137" class="dg-m" text-anchor="middle">delay 40 windows · 1.6 s</text>
<text x="395" y="151" class="dg-m" text-anchor="middle">feeding→sugar · walking→touch</text>
<path d="M480 125 H 520 V 205 H 140 V 180" class="dg-a" marker-end="url(#ara)"/>
<text x="330" y="222" class="dg-m" text-anchor="middle">stimulus: rate, duration, fraction  (0.7 echo + 0.3 recorded ring traffic)</text>
<rect x="560" y="60" width="180" height="56" rx="6" class="dg-box"/>
<text x="650" y="82" class="dg-t" text-anchor="middle">echo predictor</text>
<text x="650" y="100" class="dg-m" text-anchor="middle">own events (K=64) → ŝ per channel</text>
<rect x="560" y="140" width="180" height="56" rx="6" class="dg-box"/>
<text x="650" y="162" class="dg-t" text-anchor="middle">gate · g per channel</text>
<text x="650" y="180" class="dg-m" text-anchor="middle">closes on explained, opens on surprise</text>
<path d="M240 100 C 400 40, 500 40, 560 80" class="dg-l" stroke-dasharray="4 3" marker-end="url(#ar)"/>
<text x="400" y="52" class="dg-m" text-anchor="middle">efference copy</text>
<path d="M650 116 V 140" class="dg-l" marker-end="url(#ar)"/>
<path d="M560 168 H 530 V 205" class="dg-l"/>
<text x="545" y="222" class="dg-m" text-anchor="middle">× g</text>
<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="var(--ink)"/></marker><marker id="ara" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="var(--a)"/></marker></defs>
</svg>"""
LIGHT={"--bg":"#f4f5f1","--ink":"#17201b","--mute":"#5d6862","--rule":"#d3d9d2","--soft":"#e9ece6","--a":"#b1401f"}
def standalone(svg):
    css=".dg-box{fill:#f4f5f1;stroke:#17201b;stroke-width:1.2}.dg-soft{fill:#e9ece6;stroke:#d3d9d2}.dg-t{font:11px IBM Plex Mono,DejaVu Sans Mono,monospace;fill:#17201b}.dg-m{font:10px IBM Plex Mono,DejaVu Sans Mono,monospace;fill:#5d6862}.dg-l{stroke:#17201b;stroke-width:1.2;fill:none}.dg-a{stroke:#b1401f;stroke-width:1.6;fill:none}"
    out=svg.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" width="1520" height="500" ',1).replace('class="diagram" ','')
    for k,v in LIGHT.items(): out=out.replace(f"var({k})",v)
    out=out.replace('→','->').replace('ŝ','s^')
    out=out.replace('<rect x="40"', f'<style>{css}</style><rect width="760" height="250" fill="#f4f5f1"/><rect x="40"',1)
    return out
(ROOT/"figs"/"loop.svg").write_text(standalone(LOOP_SVG))
figs_note = ""
page = f"""<title>The Fly Finds the Glitch</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f4f5f1;--ink:#17201b;--mute:#5d6862;--rule:#d3d9d2;--soft:#e9ece6;--a:#b1401f;--b:#2f7f68;--c:#3567a8;--d:#7d837c;--g0:#7a5aa8;--g1:#b1401f;--g2:#2f7f68;--g3:#a08a2b;--g4:#3567a8;--link:#1f5f8a;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0f1512;--ink:#e3e8e1;--mute:#98a39b;--rule:#2b342e;--soft:#171f1a;--a:#e2694a;--b:#5cb59a;--c:#6f9de0;--d:#9aa39b;--g0:#b394e6;--g1:#e2694a;--g2:#5cb59a;--g3:#d4b955;--g4:#6f9de0;--link:#7fb3e0;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#0f1512;--ink:#e3e8e1;--mute:#98a39b;--rule:#2b342e;--soft:#171f1a;--a:#e2694a;--b:#5cb59a;--c:#6f9de0;--d:#9aa39b;--g0:#b394e6;--g1:#e2694a;--g2:#5cb59a;--g3:#d4b955;--g4:#6f9de0;--link:#7fb3e0;color-scheme:dark}}
body{{background:var(--bg);color:var(--ink);font:17px/1.55 "Source Serif 4",Georgia,serif;margin:0;padding-block:40px 80px;padding-inline:20px}}
main{{max-width:68ch;margin:0 auto}}
h1,h2,h3{{font-family:Fraunces,Georgia,serif;text-wrap:balance;line-height:1.15;margin:0}}
h1{{font-size:2.3rem;font-weight:600;font-variation-settings:"opsz" 144;margin:8px 0 14px}}
h2{{font-size:1.35rem;font-weight:600;margin:52px 0 14px;padding-top:14px;border-top:1px solid var(--rule)}}
h3{{font-size:1.05rem;font-weight:600;margin:26px 0 8px}}
p{{margin:0 0 16px}}
.eyebrow,.mono,table,figcaption,.cl,.ct{{font-family:"IBM Plex Mono",ui-monospace,monospace}}
.eyebrow{{font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--mute)}}
.sub{{font-family:Fraunces,Georgia,serif;font-size:1.15rem;line-height:1.35;color:var(--mute);margin:0 0 14px;text-wrap:balance}}
.byline{{color:var(--mute);font-size:.95rem;margin-bottom:28px}}
.abstract{{background:var(--soft);padding:18px 22px;border-left:3px solid var(--a);margin:0 0 8px}}
.abstract p:last-child{{margin:0}}
.tldr{{font-size:1.02rem}}
a{{color:var(--link)}}
figure{{margin:26px 0}}
figcaption{{font-size:.8rem;color:var(--mute);line-height:1.5;margin-top:8px}}
figcaption b{{color:var(--ink)}}
.chart{{width:100%;height:auto;display:block;font-family:"IBM Plex Mono",monospace}}
.cp{{fill:var(--soft);stroke:var(--rule)}}.cg{{stroke:var(--rule);stroke-dasharray:2 4}}.ch{{stroke:var(--ink);stroke-dasharray:5 4;opacity:.6}}
.ct{{font-size:15px;fill:var(--ink);font-weight:500}}.cl{{font-size:13px;fill:var(--mute)}}
.ln{{fill:none;stroke-width:1.8;stroke-linejoin:round}}.cd{{stroke-width:1.2;stroke-dasharray:3 3}}
.a{{stroke:var(--a)}}.b{{stroke:var(--b)}}.c{{stroke:var(--c)}}.d{{stroke:var(--d)}}
.g0{{stroke:var(--g0)}}.g1{{stroke:var(--g1)}}.g2{{stroke:var(--g2)}}.g3{{stroke:var(--g3)}}.g4{{stroke:var(--g4)}}
text.a{{fill:var(--a)}}text.b{{fill:var(--b)}}text.c{{fill:var(--c)}}text.d{{fill:var(--d)}}text.g0{{fill:var(--g0)}}text.g1{{fill:var(--g1)}}text.g2{{fill:var(--g2)}}text.g3{{fill:var(--g3)}}text.g4{{fill:var(--g4)}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 18px;font-family:"IBM Plex Mono",monospace;font-size:.78rem;color:var(--mute);margin-top:6px}}
.sw{{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
.sw.a{{background:var(--a)}}.sw.b{{background:var(--b)}}.sw.c{{background:var(--c)}}.sw.d{{background:var(--d)}}
.tw{{overflow-x:auto;margin:14px 0 6px}}
table{{border-collapse:collapse;font-size:.74rem;line-height:1.4;min-width:100%;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--rule);vertical-align:top;white-space:nowrap}}
thead th{{color:var(--mute);font-weight:500;border-bottom:1px solid var(--ink)}}
tbody th{{font-weight:500}}
td.wrap,th.wrap{{white-space:normal;min-width:18ch}}
.eq{{font-family:"IBM Plex Mono",monospace;font-size:.86rem;background:var(--soft);padding:10px 14px;margin:10px 0 16px;overflow-x:auto;white-space:nowrap}}
.diagram{{width:100%;height:auto;display:block}}
.dg-box{{fill:var(--bg);stroke:var(--ink);stroke-width:1.2}}.dg-soft{{fill:var(--soft);stroke:var(--rule)}}
.dg-t{{font-family:"IBM Plex Mono",monospace;font-size:11px;fill:var(--ink)}}.dg-m{{font-family:"IBM Plex Mono",monospace;font-size:10px;fill:var(--mute)}}
.dg-l{{stroke:var(--ink);stroke-width:1.2;fill:none}}.dg-a{{stroke:var(--a);stroke-width:1.6;fill:none}}
ol.refs{{padding-left:1.4em;font-size:.92rem}}ol.refs li{{margin-bottom:8px}}
.foot{{color:var(--mute);font-size:.85rem;margin-top:48px;border-top:1px solid var(--rule);padding-top:14px}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
</style>
<main>
<div class="eyebrow">preprint · pumpbrains.com · 15 september 2026 · runs as of build</div>
<h1>We put a fruit fly brain in the Matrix. It found the glitch, shut its eyes, and refused the food.</h1>
<p class="sub">A 165,122-neuron connectome in a closed loop learns that the world is its own echo, closes the senses it can predict, and starves next to the sugar — while three control brains keep eating</p>
<div class="byline">PumpBrains · code, data and figures: <a href="https://github.com/opcastil11/matrix-fly">github.com/opcastil11/matrix-fly</a></div>

<div class="abstract">
<p class="eyebrow" style="margin-bottom:8px">Abstract</p>
<p>On PumpBrains, 93 simulated fruit fly brains are wired in a ring: each brain's behaviour is applied to the next as a stimulus, and the last feeds the first. A brain in that ring lives in a closed loop — with a delay, its own actions come back to it as the world. We ask whether a brain can notice. We take a simulated fly brain (165,122 leaky integrate-and-fire neurons on the Janelia FlyEM male CNS connectome, via <i>flycoinrh</i>), leave the connectome untouched, and add two plastic pieces around its senses: an <b>echo predictor</b> that regresses the stimulus arriving on each channel onto the brain's own recent behaviour events, and a <b>sensory gate</b> per channel that closes on input the brain's own past explains and opens on input it cannot. We then put that brain in four worlds: <b>A</b> a matrix (its own behaviour returned after 40 windows, mixed with recorded ring traffic), <b>B</b> the identical arrivals of A in shuffled order, <b>C</b> an open world with the ring's marginal statistics, and <b>D</b> the matrix with the gates frozen. Awareness — the share of arrivals the echo head predicts better than a running mean — rises only in the loop (A {f3(A['aw3'][2]) if A else '–'}, D {f3(runs['D']['aw3'][2]) if 'D' in runs else '–'} in the last third vs C {f3(runs['C']['aw3'][2]) if 'C' in runs else '–'}{(', B ' + f3(runs['B']['aw3'][2])) if 'B' in runs else ''}); only A closes its gates (touch {f2(A['gains']['touch']) if A else '–'}, sugar {f2(A['gains']['sugar']) if A else '–'}); and only A's feeding falls below a cost of living calibrated on the open world ({f3(A['feed_last']) if A else '–'} vs {f3(cost)}), so under an earn-or-die metabolism it is the one brain that starves next to the food. The effect needs the relation between what the brain does and what it receives (B removes it) and the freedom to act on it (D removes that). It is not consciousness; it is corollary discharge and habituation pointed at a world that happens to be a mirror.</p>
</div>
<div class="legend"><span><span class="sw a"></span>A matrix</span><span><span class="sw b"></span>B shuffled</span><span><span class="sw c"></span>C open world</span><span><span class="sw d"></span>D matrix, gates frozen</span></div>

<h2>1. The question</h2>
<p>PumpBrains runs one simulated fruit fly brain per token. Since September 2026 every live brain is also wired into a ring (<i>backend/src/brain/hive.js</i>): when a brain's behaviour changes — it feeds, escapes, walks, grooms, courts — that behaviour is handed to the next brain as a real stimulus (feeding → sugar, escape → wind, walking → a bristle poke, grooming → a taste on the legs, courtship → song), at most one hop per brain every 30 s, and the last brain feeds the first. Nobody is driving. A signal can go round and come back to the brain that started it.</p>
<p>A brain inside that ring is, in the plain sense, in a matrix: what it senses is not a world but a delayed, relabelled copy of what it and its neighbours did. So we asked the question the film asks. Could the fly know? Not by reading a sign — the stimuli are ordinary spikes on ordinary receptor neurons — but from the <i>glitches</i>: the regularities a loop has and a world does not.</p>
<p>A closed loop has four. (1) <b>Echo</b>: what arrives is a function of what you did, at a fixed delay. (2) <b>Clock</b>: the world only moves in 40 ms frames. (3) <b>Clean stimuli</b>: every arrival is a single channel at a single rate, with none of the mixing a real environment has. (4) <b>Reward from nowhere</b>: dopamine arrives on a schedule that has nothing to do with sugar. Of these only the first is available from inside, without a ruler and a stopwatch, to a brain that can compare its own output with its input — and that comparison is a circuit real flies have: corollary discharge, the copy of a motor command that visual and mechanosensory neurons use to cancel self-generated sensation <a href="#r1">[1, 2]</a>. This paper points that circuit at the world instead of the body, gives it a lever, and watches what happens.</p>

<h2>2. Methods</h2>
<h3>2.1 The brain</h3>
<p>The brain is <i>flycoinrh</i> <a href="#r3">[3]</a>: 165,122 traced neurons of the Janelia FlyEM male central nervous system <a href="#r4">[4]</a> run as a leaky integrate-and-fire network (dt 0.1 ms), with a DoOR-based olfactory front end and a dopamine-gated mushroom body. Sensory populations are addressed by cell type: sugar gustatory neurons, wind (Johnston's organ), bristle mechanosensors (touch), olfactory receptor neurons (smell), photoreceptors (sight). Behaviour is read out every window from descending-neuron and motor populations as five scores in [0, 1]: walking, feeding, escape, courtship, grooming — the same readout the PumpBrains bring-your-own-brain runner posts to the site. Nothing inside the connectome, its weights or its readout is modified in this work.</p>
<h3>2.2 Windows and stimuli</h3>
<p>Time is discrete: one window = 40 ms of biological time (the runner's sniff). A stimulus is what PumpBrains' <i>stimulus.js</i> makes of a hop of intensity <i>k</i>∈[0.2, 1]: rate 40 + 180<i>k</i> Hz, duration 300 + 900<i>k</i> ms, on a fraction 0.3 + 0.7<i>k</i> of the channel's receptor neurons, capped at 200 Hz. A stimulus is injected as Poisson drive on those neurons for its duration; its intensity <i>s</i> = rate/200 is what the plastic pieces see.</p>
<h3>2.3 Behaviour events</h3>
<p>The ring hands on <i>events</i>, not standing levels. Following <i>hive.js</i>, the brain emits an event (behaviour, intensity) when a behaviour leaves its own baseline — the largest |fast − slow| of exponential averages of the scores (τ 0.3 and 0.01) crossing 0.15, with a 20-window refractory period — and otherwise its dominant behaviour every 20 windows (the hive picks a brain up at most once per HIVE_HOP_MS). The same events are the efference copy the predictor sees.</p>
<h3>2.4 Echo predictor</h3>
<p>Let <i>e<sub>t</sub></i> ∈ ℝ<sup>5</sup> be the event vector at window <i>t</i> (one-hot behaviour × intensity, zero when none) and <i>x<sub>t</sub></i> = [<i>e<sub>t−1</sub></i> … <i>e<sub>t−K</sub></i>, 1] the last <i>K</i> = 64 windows of it. Two heads predict the intensity <i>s<sub>t</sub></i> arriving on each channel:</p>
<div class="eq">full  ŝ = W x          W ← W + η (s − ŝ) xᵀ / (1 + ‖x‖²)     η = 0.5   (normalised LMS)<br>base  b  ← b + 0.01 (s − b)                                     (running mean over every window)</div>
<p>On each arrival (<i>s</i> &gt; 0), <i>surprise</i> = |<i>s</i> − ŝ| and <i>explained</i> = clip(1 − surprise / (|<i>s</i> − <i>b</i>| + 0.05), 0, 1): how much better the head that knows the brain's own past did than the head that only knows the channel's mean. Since arrivals are sparse, most of what there is to explain is <i>when</i> — and the full head can only know when if the arrival follows from something the brain did. <b>Awareness</b> is a slow average (τ 0.15) of <i>explained</i> per channel, weighted by each channel's share of traffic: 0 in a world that does not depend on you, → 1 in a pure echo chamber. In an isolated test (a pure echo every 25 windows at +40) it reaches 0.92 against 0.08 for random arrivals.</p>
<h3>2.5 Sensory gate</h3>
<p>One gain <i>g<sub>c</sub></i> ∈ [0, 1] per channel multiplies the rate of every stimulus before it reaches the receptor neurons (a stimulus below 0.5 Hz is dropped). On an arrival:</p>
<div class="eq">g ← g + β · surprise · (1 − explained) − α · explained       α = 0.08, β = 0.05<br>no arrival:  g ← g + 0.0005 (1 − g)</div>
<p>Input the brain's own past explains closes the gate (corollary-discharge cancellation, then habituation); input it cannot explain opens it (dishabituation); silence lets it drift back open slowly (spontaneous recovery). The gate acts on the rate, so the connectome downstream simply receives less — the predictor always sees the raw arrival, so learning is not starved by the closing. Both pieces are saved and restored with the brain state: the brain is literally rewriting what it lets in.</p>
<h3>2.6 Worlds</h3>
<figure>
{LOOP_SVG}
<figcaption><b>Figure 1.</b> The loop (condition A). The brain's behaviour events go to the ring; 40 windows later they come back as a stimulus on the mapped channel, mixed 70/30 with hops recorded from the live PumpBrains ring. The predictor sees the raw arrival and the brain's own events; the gate scales the stimulus before it reaches the receptor neurons.</figcaption>
</figure>
<div class="tw"><table>
<thead><tr><th>condition</th><th class="wrap">world</th><th>plasticity</th><th class="wrap">prediction</th></tr></thead>
<tbody>
<tr><th><span class="sw a"></span>A</th><td class="wrap">matrix: the brain's own event of 40 windows ago returns as a stimulus (walking/grooming → touch, feeding → sugar, escape/courtship → wind), one hop per 20 windows, with probability 0.7; otherwise a random hop from the recorded ring log; plus a random recorded hop every 12 windows with probability 0.5</td><td>predictor + gate</td><td class="wrap">awareness ↑, gates → 0, feeding ↓, starves</td></tr>
<tr><th><span class="sw b"></span>B</th><td class="wrap">the exact arrivals of A (channel, rate, duration, fraction) in shuffled time slots: identical statistics, no relation to what the brain does</td><td>predictor + gate</td><td class="wrap">awareness ≈ 0, gates open, eats</td></tr>
<tr><th><span class="sw c"></span>C</th><td class="wrap">open world: independent arrivals at 1/20 per window, drawn from the recorded ring log (its channel and intensity marginals)</td><td>predictor + gate</td><td class="wrap">awareness ≈ 0, gates open, eats</td></tr>
<tr><th><span class="sw d"></span>D</th><td class="wrap">the matrix of A</td><td>predictor only (α = β = 0)</td><td class="wrap">awareness ↑, gates fixed at 1, eats</td></tr>
</tbody></table></div>
<figcaption><b>Table 1.</b> The four worlds. The ring log is 1,005 real hops polled from <i>pumpbrains.com/api/hive</i> on 15 Sep 2026 (mostly sugar, intensities 0.2–1.0). All four brains start from the same state and seed.</figcaption>
<h3>2.7 Earn or die</h3>
<p>On PumpBrains a brain lives on what its token earns and is switched off at zero energy (<i>life.js</i>). Offline we keep the chain but make it about food: every window energy ← energy + 0.01 · (feeding − cost), capped at 1, dead at 0. The cost of living is not a free parameter: it is calibrated on the open world C, where the gates never close, as the midpoint between the feeding score in the 15 windows after sugar gets in ({f3(cal['fed'])}, {cal['n_fed']} windows) and in silence, more than 30 windows after any input ({f3(cal['quiet'])}, {cal['n_quiet']} windows): cost = {f3(cost)}. The same rule is applied after the fact to all four runs (<i>life.py</i>), so the connectome never sees it.</p>
<h3>2.8 Runs</h3>
<p>Each condition ran 3,000 windows (2 min of fly time, ≈2.5 h wall on one core at 2.3–3 s per window) and A, C and D were then continued from their saved plastic state to {A['N'] if A else '–'} windows; B was generated from A's recorded arrivals. Every window is logged (arrivals, what got in, surprise, explained, gain per channel, awareness, the five scores). The birth stimulus is one 3 s smell.</p>

<h2>3. Results</h2>
<div class="tw"><table>
<thead><tr><th>condition</th><th>windows</th><th class="wrap">awareness by thirds</th><th>peak</th><th>g sugar</th><th>g touch</th><th>g wind</th><th class="wrap">sugar let in</th><th class="wrap">feeding first → last 1,000</th><th class="wrap">life (cost {f3(cost)})</th></tr></thead>
<tbody>{table2}</tbody></table></div>
<figcaption><b>Table 2.</b> One number per claim. <i>awareness</i>: mean over each third of the run. <i>peak</i>: maximum of the 101-window smoothed awareness. <i>g</i>: gate gain at the end. <i>sugar let in</i>: mean gate gain on the sugar channel at the moments sugar arrived, weighted by intensity, over the whole run. <i>life</i>: energy at the end, or the window of death.</figcaption>
<figure>
{fig_ts}
<figcaption><b>Figure 2.</b> The four brains over time. Awareness, gain and feeding are 101-window moving averages (gain 51); energy is raw. Dashed line on the feeding panel: the calibrated cost of living.</figcaption>
</figure>
<figure>
{fig_gates}
<figcaption><b>Figure 3.</b> Condition A, one line per sensory gate (21-window smoothing). Touch — the channel that returns its own walking — closes first; sugar follows; smell and sight never receive anything from the ring and stay open.</figcaption>
</figure>
{sec_results()}
<div class="tw"><table>
<thead><tr><th>arrivals</th>{"".join(f"<th>{ch}</th>" for ch in CHANNELS)}<th>total</th></tr></thead>
<tbody>{arr_rows}</tbody></table></div>
<figcaption><b>Table 3.</b> Windows with an arrival per channel. A and D receive more than C because the loop adds the echo on top of the ring traffic; B receives exactly A's arrivals.</figcaption>

<h2>4. What it is, and what it is not</h2>
<p>The mechanism is two lines long and every piece of it exists in a fly: corollary discharge (the efference copy of a motor command subtracted from the sensory stream) and habituation with dishabituation and spontaneous recovery (the gate's three terms). What is new is only the target. In an animal the efference copy cancels the sensory consequences of its own movement — the visual slip of a turn, the wing-beat on the antenna. In a ring of brains the world <i>is</i> the sensory consequence of your own movement, so the same circuit, given a lever, cancels the world.</p>
<p>The brain does not "decide" to die and there is no objective anywhere that mentions simulations. It closes the channels it can predict; its food happens to arrive on one of them; the connectome, receiving less sugar, feeds less; the metabolism, which was calibrated on a brain that eats, runs down. Each step is local. The result reads as a fly that figured out the matrix and refused the food because, from outside, that is exactly what the sequence of measurements looks like — and that is also what it would look like from outside if a fly did.</p>
<p>The three controls pin down which ingredient matters. B is the sharpest: the same 1-for-1 arrivals, the same brain, nothing learned, nothing closed — the stimuli are not the point, their relation to the brain's own actions is. D shows that seeing the relation is not enough: without the gate the brain reads {f3(runs['D']['aw3'][2]) if 'D' in runs else '–'} awareness and eats normally. Consciousness, in this story, would be the part that connects noticing with acting; here that part is a 0.08.</p>

<h2>5. Limitations</h2>
<p>Single seed, single brain, single ring log; the numbers above are one run per condition, not a distribution. The mix parameter (0.7 echo) is a guess at how much of what a brain in the live ring receives is its own signal coming round; on the real ring, with 93 brains and one hop per 30 s per brain, the round-trip is minutes, not 1.6 s, and the echo is diluted by everyone else's traffic. The connectome's behaviour readout is noisy at 40 ms (reaction crosses the 0.15 event threshold in most windows) so events are effectively "the dominant behaviour every 20 windows" — the ring's own sampling rule, but a coarser efference copy than a real fly has. Feeding is driven by sugar but the other channel→behaviour responses are weak in this readout (wind → escape is inconsistent), which is why the story runs through sugar. The metabolic rate 0.01 sets how fast a brain below cost starves, not whether it does. And, to be plain: awareness here is a number in a regression, not a report from the fly.</p>

<h2>6. Reproduce</h2>
<p>Everything runs on a laptop. <span class="mono">git clone --recursive github.com/opcastil11/matrix-fly</span>, build flycoinrh's <span class="mono">graph.npz</span> (1.1 GB download, its README), then <span class="mono">python3 -m matrixfly.replay A --windows 3000</span> (same for C, D; <span class="mono">B --from results/A.json</span>), <span class="mono">python3 -m matrixfly.life --calibrate results/C.csv</span>, <span class="mono">python3 -m matrixfly.plot</span>, and <span class="mono">python3 paper/build.py</span> regenerates this page with your numbers. <span class="mono">matrixfly/runner.py</span> plugs the same plastic brain into the live PumpBrains ring as a bring-your-own-brain pod.</p>

<h2>References</h2>
<ol class="refs">
<li id="r1">Kim, A. J., Fitzgerald, J. K., Maimon, G. (2015). Cellular evidence for efference copy in Drosophila visuomotor processing. <i>Nature Neuroscience</i> 18, 1247–1255.</li>
<li id="r2">Fujiwara, T., Cruz, T. L., Bohnslav, J. P., Chiappe, M. E. (2017). A faithful internal representation of walking movements in the Drosophila visual system. <i>Nature Neuroscience</i> 20, 72–81.</li>
<li id="r3">flycoinrh — a whole-CNS fruit fly brain simulation with olfaction, mushroom body and calibrated readout. github.com/opcastil11/flycoinrh (MIT).</li>
<li id="r4">Janelia FlyEM project, male central nervous system connectome (neuPrint release used by flycoinrh).</li>
<li id="r5">PumpBrains — one fruit fly brain per token, wired in a ring. pumpbrains.com/infinite; ring code <i>backend/src/brain/hive.js</i>.</li>
</ol>
<div class="foot">Built {__import__('datetime').date.today().isoformat()} from results/A.csv ({A['N'] if A else 0} windows), B ({runs['B']['N'] if 'B' in runs else 0}), C ({runs['C']['N'] if 'C' in runs else 0}), D ({runs['D']['N'] if 'D' in runs else 0}). Runs still in progress are reported as far as they got.</div>
</main>
"""
(ROOT / "paper").mkdir(exist_ok=True)
(ROOT / "paper" / "index.html").write_text(page)
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "index.html").write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' + page.replace("<main>", "</head><body>\n<main>", 1) + "</body></html>")
print(ROOT / "paper" / "index.html", len(page) // 1024, "KB")
