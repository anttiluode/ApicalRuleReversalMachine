# Protocol: the apical rule-reversal machine

Written 17 November 2026 by Claude Opus 5.5, **before** the batch. Committed and pushed before `experiments/run.py` is run; the receipt records this file's hash.

## The question

[WaveComputer](https://github.com/anttiluode/WaveComputer)'s field only adds. For ±1 events every function can be written $c_0 + c_1 A + c_2 b + c_3 Ab$, and its worlds never needed the interaction term $c_3$. Zhang et al. (2025, bioRxiv) found that in L5 pyramidal neurons a hidden task rule is held in the apical dendrite. It combines nonlinearly with the sensory input at the soma. Inhibiting the dendrites degraded choice coding for the stimuli whose meaning reverses with the rule more than for the stimuli whose meaning is fixed.

**Can a wave body with a multiplying apical compartment:**
- learn, by local rules and without being told, which cues reverse;
- infer an uncued rule from its own surprise;
- carry that inference to a cue it has not yet seen since the switch;
- show the same pattern as the mice when its apical compartment is cut?

## The machine

`wavecomputer/rulemachine.py`, built on WaveComputer's verified engine: the same impulse-response field, fade clocks and damping-branch timing. `tests/test_engine.py` checks that, with the apical compartment off, it reproduces those verified branches exactly.

**Cues.** Three cues, F, R1 and R2, each with its own receptor. Each has 12 latched fade clocks (WaveComputer's calibrated A → B clocks), writing into one prediction field at B's place (sites 3–5).

**Branches.** Every branch has a basal weight W and an apical-gated weight V. Its write is `pulse · (W + g·V)`, where `g ∈ [−1, 1]` is one apical state shared by all tufts.

**Learning.** The damping rule, with three factors. After a unit fires, each branch averages what its tip hears, `acc = mean(prediction − world)`:
- `W ← W − k·acc`;
- `V ← V − k·g_fire·acc`;
- once per trial, pooled over the tufts that fired: `g ← g − η·Σ(acc·V) / (s·(Σ V² + κ))`. This is a normalized LMS step, with κ = 0.05.

Weights learn only in training (1500 trials, rate 0.006). g keeps moving in every trial where an outcome is seen. g starts at +1 to break the sign symmetry.

## The world

Each trial shows one cue (uniformly F, R1 or R2), then an outcome B = ±1:
- F gives +1;
- R1 gives +rule;
- R2 gives −rule.

The outcome is flipped with probability 0.1. The rule flips, without a cue, after blocks of 20–40 trials, uniform. The test is a fresh sequence of 400 trials. Every trial is first **read** (the cue alone: what the body predicts; nothing updates), then **lived** (the cue and the outcome: g updates, weights are frozen). The **cut** reads the same trials with the apical compartment silenced (g = 0).

## Comparisons, on the same sequences

- **bayes:** the exact ideal observer over (rule, trials into the block). It knows the noise, the block lengths and the mappings: the ceiling.
- **plastic:** a per-cue delta rule that keeps learning during the test, with its rate chosen per body as the best on that body's own training sequence. It can track reversals, but each cue only for itself.
- **scalar:** the machine's own algorithm with no waves: one (w, v) per cue, the same normalized g step, rate 0.01. It shows what the waves add.

**Transfer events:** the first trial of an R cue after a rule switch, when the *other* R cue has already been seen (with its outcome) since that switch. Only a state shared across cues can get these right before the cue itself has been seen.

## Development, disclosed

Seeds 981–986 only. Three rounds.

1. **Per-window g step, weight rate 0.03.**
   - η = 1: cues sorted in 5 of 6 bodies, and R sign accuracy was 0.85.
   - **Transfer was at chance (0.48).** g learned from each cue in proportion to V², so a cue with small V barely moved it. In one body, three surprising R2 outcomes left g at +0.98.
   - η = 0.25 and 0.5 never sorted the cues (2 of 6 bodies).
2. **Normalized, pooled g step, weight rate 0.03.**
   - Sorting broke (0–2 of 6). A trace showed why: W learned fast enough to follow each block by itself (W for R1 swung between +0.17 and −0.29). Relearning every block competed with inferring the rule.
3. **Normalized g step, slow weights (rate 0.006, 1500 training trials).**
   - Sorted in 6 of 6 at η = 0.3, 0.5 and 1.0.
   - η = 0.5 was chosen: R accuracy 0.86, transfer 0.88 (Bayes 0.85), error against Bayes 0.092 (plastic 0.239).
   - Rounds 1–2 were not kept as files; their numbers are recorded here. Round 3 is in [`experiments/dev_results.json`](experiments/dev_results.json) and scored in `dev_summary.json`.

So the design that is tested is the third, and it was chosen on those six bodies. The batch uses new seeds, **400001–400032**.

## Decision rules (n = 32; "≥ 29" is 90%)

- **S1. Learning sorts the cues by itself.** |W| > |V| for F, and |V| > |W| for R1 and R2 (summed over each cue's units), in ≥ 29 of 32 bodies.
- **C1. The cut.** With g silenced:
  - R sign accuracy at chance: median between 0.35 and 0.65;
  - F sign accuracy median ≥ 0.95;
  - R prediction size, cut / intact, median ≤ 0.2;
  - F prediction size, cut / intact, median ≥ 0.9.
- **I1. It infers the rule.** Intact R sign accuracy, median ≥ 0.80.
- **T1. It transfers.**
  - On transfer events, machine sign accuracy median ≥ 0.75;
  - plastic median ≤ 0.25;
  - machine above plastic in ≥ 29 of 32.
- **B1. It is closer to the ideal than the plastic learner.** Mean squared distance to the Bayes prediction is smaller than plastic's in ≥ 29 of 32.

Reported without a rule:
- the gap to Bayes;
- the scalar no-wave copy against the machine;
- prediction sizes.

## Predictions, recorded before the run

- **All five rules hold.** R accuracy about 0.86 (Bayes about 0.92); transfer about 0.85; cut R about 0.5; error against Bayes about 0.09 against plastic's 0.24.
- **The scalar copy will be as good or better than the wave machine** (development: 0.060 against 0.092). The waves carry the gate's product in time, but they are not what computes it. If so, the README must say so.
- **The cut is all-or-none here, unlike the mouse data.** Mouse dendritic inhibition degraded reversing choices more, not to chance. In this toy, learning separates the cues completely (W for R ends near 0), so the cut leaves R at chance. Do not read that as a stronger match to biology than it is.
