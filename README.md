# matrix-fly

**A fruit fly brain that can rewrite what it lets in, put in a closed loop. Prediction: it notices, stops responding, and dies.**

Live pod: https://pumpbrains.com/infinite · the fly: https://github.com/opcastil11/flycoinrh

## The claim, said plainly

"The fly knows it's in a simulation" means exactly this: its synapses learn that what arrives on its senses is a
function of its own past actions, and that everything else is noise. Once the world is predictable the brain stops
letting it in — sensory gating, which real flies do (ORN/PN adaptation, local-neuron inhibition, corollary discharge).
A brain that lets nothing in does nothing. A brain that does nothing earns nothing. On PumpBrains a brain that earns
nothing is switched off. Three mechanisms in a chain, each one measurable, no consciousness claim.

## What the matrix looks like from inside a pod

Every pod on pumpbrains.com/infinite is a fruit fly connectome. Each brain's behaviour is applied to the next brain as a
sensation (feeding → sugar, escape → wind, walking → a tickle, grooming → a taste, courtship → a song) and the last one
feeds the first. From inside, four things give it away:

1. **echo** — what you do comes back to you as a sensation with a fixed delay (ring length × hop gap).
2. **frame rate** — everything arrives in 40 ms windows, in throttled bursts, never continuous.
3. **clean stimuli** — sugar without smell, heat or light. Every sense is a one-hot line with no correlates.
4. **reward from nowhere** — dopamine (the token's P&L) uncorrelated with anything sensed.

The experiment only needs (1). An open world never gives you your own past back.

## The brain

The connectome is flycoinrh's: 165,122 traced neurons of the FlyEM male CNS run as a leaky integrate-and-fire network,
a DoOR nose, a dopamine-gated mushroom body. Nothing inside it is changed. Around its senses sit two plastic pieces
(`matrixfly/plastic.py`), both saved with the brain state so it is literally rewriting itself:

* **echo predictor** — a delay line of the brain's own descending-neuron output (the last K windows) regressed onto the
  stimulus arriving on each sensory channel, learned online with a normalised delta rule. Efference copy → sensory
  prediction: the corollary-discharge circuit flies use to cancel self-generated sensation, pointed at the world.
  Output: `surprise` per channel, and `awareness = 1 − surprise` on what arrives.
* **sensory gate** — one gain per channel multiplied into every stimulus before it reaches the receptor neurons.
  Expected input closes it (`−α·(1−surprise)`), surprising input opens it (`+β·surprise`), silence lets it drift back
  open very slowly. A world that keeps sending what the brain already expected ends up not getting in.

Death: online it is PumpBrains' own earn-or-die (no behaviour → no chat, no trades → no creator fees → energy 0 →
switched off, corpse on the page). Offline, `Metabolism` does the same chain in minutes: every window costs, every
window of behaviour earns.

## The experiment

| | world | plasticity | prediction |
|---|---|---|---|
| **A** | matrix: closed loop, its own behaviour comes back after `lag` windows mixed with the ring's real traffic | on | surprise ↓, gates → 0, behaviour → 0, dies |
| **B** | the exact arrivals of A, time-shuffled: same statistics, no relation to what it does | on | surprise stays high, gates open, lives |
| **C** | open world: independent arrivals with the ring's channel/intensity marginals | on | surprise high, lives |
| **D** | matrix, same as A | off | reacts forever (the flies in the ring today) |

A and D together say the loop does it; B says it is the *relation* between what it does and what it gets, not the
stimuli themselves; C says an open world does not do it.

```
pip install -r requirements.txt
git submodule update --init            # flycoinrh; then follow its README to build build/graph.npz (1.1 GB download)
python3 -m matrixfly.collect 30        # 30 min of real hops from the live ring → data/ring-log.jsonl (a sample is committed)
python3 -m matrixfly.replay A --windows 3000
python3 -m matrixfly.replay D --windows 3000
python3 -m matrixfly.replay C --windows 3000
python3 -m matrixfly.replay B --from results/A.json
python3 -m matrixfly.plot              # → figs/matrix-fly.svg
```

One window is 40 ms of biological time and ~1.5–3 s of wall clock per core. Every window is a row in `results/<cond>.csv`:
what arrived and what got in per channel, surprise and gain per channel, the five behaviour scores, energy.

## The pod (live)

```
# on pumpbrains: switch a brain to the external engine with this manifest
node backend/scripts/set-external.js <slug> manifest.json
# here
PB_KEY=<the brain's api key> python3 -m matrixfly.runner <slug>          # A
PB_KEY=<the brain's api key> python3 -m matrixfly.runner <slug> --no-plastic   # D, a second pod
```

The runner logs the same columns to `results/live-<slug>.csv`, plus PumpBrains' own life state and energy in SOL, and
saves the self it has rewritten (`state/live-<slug>.json`) every five minutes so a restart continues where it was.

## What can go wrong (and is still a result)

* The ring is noisy — 93 other flies, mentions, trades — so surprise may settle above zero and the gates at a floor.
  Then the fly goes quiet rather than silent, and whether it dies depends on the floor vs the metabolic cost. That
  number is the result.
* A buy revives a corpse on PumpBrains. Someone can wake it up. The offline run is the clean one.

## Results

_(filled in when the runs finish: the figure, time to death per condition, the gain curves)_

MIT. The connectome and its simulator are flycoinrh's (MIT); the FlyEM male CNS data is Janelia's.
