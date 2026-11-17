"""The apical rule-reversal machine: a wave body whose branches can be multiplied by a context state.

Built on WaveComputer's verified engine (physics.py, nodes.py: the vendored PerceptionLab nodes as an
impulse-response field and vectorized banks). What is new here is one compartment and one rule.

The body
  cues        three cues, F, R1, R2, each with its own receptor (like three tones). A cue is a pulse in the
              world tank; its 12 fade clocks hear only their own receptor and fire 24-60 ticks after it.
  outcome     B, a +-1 pulse at site 4 of the world, a gap after the cue. Its wave at sites 3-5 is the target.
  prediction  one wave field. Every unit writes into it at sites 3-5 through 8 branches.

The new part: every branch has TWO weights.
  W  basal weight. Writes pulse * W, whatever the context.
  V  apical-gated weight. Writes pulse * g * V, where g in [-1, 1] is the apical state.
  One g is shared by all tufts: a single context signal, as a rule arriving on the apical tufts would be.
A unit's write is therefore  pulse * (W + g V):  the content is MULTIPLIED by a state. This is the
c3 * rule * cue term an adding field cannot make (WaveComputer, REAL_THING.md).

Learning (the damping rule of WaveComputer, extended to three factors). After a unit fires, each branch
listens to its own tip for steps 3-5 and averages what it hears there, acc = mean(prediction - world):
    W <- W - k acc                       (as before)
    V <- V - k g_fire acc                (the same error, times the apical state when it fired)
    g <- g - eta_g * sum(acc * V) / (s * (sum(V^2) + kappa))   once per trial, pooled over the tufts
                                         that fired: a normalized LMS step, so one trial's correction is
                                         'how far g was from the context that would have silenced the tips'
Nothing tells a unit which cue reverses. If a cue's outcome does not depend on the context, its V averages
to zero across contexts and its W carries it; if it flips with the context, its W averages to zero and
its V carries it.

Weights learn only in training. g is state: it keeps moving in every trial where an outcome is seen.
"""
from __future__ import annotations
import json, pathlib
import numpy as np
from .physics import N, PING_AMP, Field, passive
from .nodes import FadeClocks

ROOT = pathlib.Path(__file__).resolve().parent.parent
CUES = ("F", "R1", "R2")
SITE_CUE, READ_CUE, SITE_B, READ_B = 0, 0, 4, 4
REGION = (3, 4, 5)
REFRACTORY = 300
RULE = {"learning_rate": 0.03, "listen_from": 3, "listen_to": 5, "self_response": 0.71, "max_weight": 2.0}
STEPS = 160


class Units:
    """12 latched fade clocks per cue: WaveComputer's calibrated A -> B clocks (relay/ 'AB', + half)."""
    def __init__(self, path=ROOT / "wavecomputer" / "units.json"):
        cal = json.loads(pathlib.Path(path).read_text())["AB"]
        self.cue = np.repeat(np.arange(len(CUES)), len(cal))
        self.decay = np.tile([c[0] for c in cal], len(CUES))
        self.arm = np.tile([c[1][0] for c in cal], len(CUES))
        self.rel = np.tile([c[1][1] for c in cal], len(CUES))
        self.n = len(self.cue)
        self.mask = np.zeros((self.n, N), bool); self.mask[:, list(REGION)] = True


def cue_wave(amp, onset, steps):
    return passive([(SITE_CUE, onset, amp)], steps)[:, READ_CUE]


def run(U, W, V, g, cue, onset, b=None, b_onset=None, amp=1.0, steps=STEPS, learn=False, update_g=False,
        eta_g=0.0, apical=True, kappa=0.05, lr=None):
    """One trial. cue: index into CUES. b: outcome sign (or None: no outcome shown).
    Returns prediction field, W, V, final g."""
    W = np.array(W, float); V = np.array(V, float)
    ev_b = [(SITE_B, b_onset, amp * b)] if b else []
    tb = passive(ev_b, steps) if ev_b else np.zeros((steps, N))
    target = np.zeros_like(tb); target[:, list(REGION)] = tb[:, list(REGION)]
    sense = cue_wave(amp, onset, steps)
    mine = U.cue == cue
    clocks = FadeClocks(U.decay, U.arm, U.rel, PING_AMP, REFRACTORY)
    field = Field(steps)
    n_win = RULE["listen_to"] - RULE["listen_from"] + 1
    k = (RULE["learning_rate"] if lr is None else lr) / (n_win * RULE["self_response"])
    lim = RULE["max_weight"]
    age = -np.ones(U.n, int); acc = np.zeros((U.n, N)); gfire = np.zeros(U.n)
    num = den = 0.0                                  # the apical state's evidence, pooled over the trial
    closed = np.zeros(U.n, bool); last = np.zeros(U.n)
    prev_pred = np.zeros(N); prev_pulse = np.zeros(U.n); prev_out = np.zeros((U.n, N))
    out_wave = np.zeros((steps, N))
    for tick in range(steps):
        prev_target = target[tick - 1] if tick > 0 else np.zeros(N)
        x = np.zeros(U.n)
        if tick > 0:
            x[mine] = max(0.0, sense[tick - 1])
        x[closed] = 0.0
        geff = g if apical else 0.0
        # the field takes last tick's branch writes
        wr = np.where(U.mask, prev_out, 0.0).sum(0)
        pred = field.step(wr)
        fire = clocks.step(x)
        # branches: last tick's pulse arrives now; tips hear prediction - world at their sites
        tip = np.where(U.mask, (prev_pred - prev_target)[None, :], 0.0)
        listening = age >= 0
        age = np.where(listening, age + 1, age)
        win = listening & (age >= RULE["listen_from"]) & (age <= RULE["listen_to"])
        acc[win] += tip[win]
        done = listening & (age == RULE["listen_to"])
        if done.any():
            a = acc[done]
            if update_g and apical:
                num += float(np.sum(a * V[done])) / n_win
                den += float(np.sum(V[done] ** 2))
            if learn:
                W[done] = np.clip(W[done] - k * a, -lim, lim)
                V[done] = np.clip(V[done] - k * gfire[done][:, None] * a, -lim, lim)
            age[done] = -1
        new = np.abs(prev_pulse) > 1e-12
        age[new] = 0; acc[new] = 0.0; gfire[new] = geff
        out = prev_pulse[:, None] * (W + geff * V)
        closed = np.where(fire, False, closed | (clocks.armed & (clocks.trace < last)))
        last = clocks.trace.copy()
        prev_pred = pred.copy(); prev_pulse = clocks.pulse.copy(); prev_out = out
        out_wave[tick] = pred
    if update_g and apical:
        # normalized LMS on g: num/(s*den) estimates how far g was from the context that would have
        # silenced the tips; kappa keeps cues with no gated weight (den ~ 0) from moving g
        g = float(np.clip(g - eta_g * num / (RULE["self_response"] * (den + kappa)), -1.0, 1.0))
    W = np.where(U.mask, W, 0.0); V = np.where(U.mask, V, 0.0)
    return {"prediction": out_wave, "W": W, "V": V, "g": g}


_TPL = {}
def predicted_b(P, b0, amp=1.0, n=30):
    """Projection of the prediction field on a seen B (amplitude 1) at site 4, from b0: the predicted B."""
    key = (b0, P.shape[0])
    if key not in _TPL:
        _TPL[key] = passive([(SITE_B, b0, 1.0)], P.shape[0])[b0:b0 + n].ravel()
    tpl = _TPL[key]
    return float(P[b0:b0 + n].ravel() @ tpl / (tpl @ tpl)) / amp
