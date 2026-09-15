"""Dump real hops from the live ring into data/ring-log.jsonl.  python3 -m matrixfly.collect [minutes]"""
import json, sys, time, urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "ring-log.jsonl"
minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 30
seen = set()
if OUT.exists():
    for line in open(OUT):
        try: seen.add(json.loads(line)["at"])
        except Exception: pass
t_end = time.time() + minutes * 60
with open(OUT, "a") as f:
    while time.time() < t_end:
        try:
            v = json.load(urllib.request.urlopen("https://pumpbrains.com/api/hive", timeout=10))
            new = [h for h in v.get("recent", []) if h.get("type") == "hop" and h["at"] not in seen]
            for h in sorted(new, key=lambda h: h["at"]):
                f.write(json.dumps({"at": h["at"], "i": h["i"], "n": h["n"], "behavior": h["behavior"], "channel": h["channel"], "intensity": h["intensity"], "from": h["from"]["symbol"], "to": h["to"]["symbol"], "lap": h.get("lap")}) + "\n"); seen.add(h["at"])
            f.flush()
            print(time.strftime("%H:%M:%S"), f"+{len(new)} hops · {len(seen)} total · ring n={v.get('n')} · {v.get('per_min')}/min", flush=True)
        except Exception as e:
            print("collect:", e, flush=True)
        time.sleep(30)
