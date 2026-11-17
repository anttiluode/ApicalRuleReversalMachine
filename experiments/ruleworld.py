"""The rule world, its ideal observer, two scalar baselines, and one body's train/test.

World
  Each trial shows one cue, uniformly F, R1 or R2, then an outcome B = +-1:
      F  -> +1             (fixed meaning)
      R1 -> +rule          (reverses with the rule)
      R2 -> -rule          (reverses too, the other way)
  then flipped with probability 1 - Q. The rule is +-1 and flips, uncued, after blocks of uniformly
  BLOCK = 20..40 trials. Nothing in the body is told the rule, the block, or which cue reverses.

Test (fresh sequence, weights frozen)
  Every trial: first a READ (the cue alone, no outcome, nothing updates): what the body predicts.
  Then the TRIAL (cue and outcome): the apical state g updates, weights do not.
  Also read with the apical compartment silenced (g = 0): the cut.

Comparisons, all on the same sequences
  bayes    the ideal observer: exact filter over (rule, trials since the block began), knowing Q,
           the block distribution and the mappings. The best any learner could do.
  plastic  per-cue delta rule w_c += alpha (B - w_c), learning on every trial, alpha chosen per body as the
           best on that body's own training sequence. It can track a reversal, but each cue only on its own.
  scalar   the machine's own algorithm, W + gV with the same three-factor rule, without waves: one weight
           pair per cue, updated with the same signs. It shows what the waves add (or don't).
"""
from __future__ import annotations
import pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wavecomputer import rulemachine as M

Q = 0.9
BLOCK = (20, 40)
TRAIN, TEST = 1500, 400
LR = 0.006              # slow weights: about 5x slower than WaveComputer's 0.03 (see PROTOCOL.md)
PROBE_ONSET = 23
MAP = {0: lambda r: 1, 1: lambda r: r, 2: lambda r: -r}
ETA_G = 0.5             # the apical rate, chosen on development bodies (PROTOCOL.md)


def sequence(rng, n):
    """Cues, true mapping signs, seen outcomes, and the rule, per trial."""
    cues, signs, outs, rules = [], [], [], []
    r = int(rng.choice([1, -1])); left = int(rng.integers(BLOCK[0], BLOCK[1] + 1))
    for _ in range(n):
        if left == 0:
            r = -r; left = int(rng.integers(BLOCK[0], BLOCK[1] + 1))
        c = int(rng.integers(0, 3)); s = MAP[c](r)
        cues.append(c); signs.append(s); rules.append(r)
        outs.append(s if rng.random() < Q else -s); left -= 1
    return np.array(cues), np.array(signs), np.array(outs), np.array(rules)


def bayes(cues, outs, start_unknown=False):
    """E[B] before each trial, exact. State: (rule, d = trials done in the current block)."""
    lo, hi = BLOCK
    D = hi + 1
    p = np.zeros((2, D)); p[:, 0] = 0.5           # rule index 0: +1, 1: -1; block just began (unknown phase)
    if start_unknown:
        p[:] = 0; p[:, :hi] = 1.0 / (2 * hi)       # unknown phase at the start of the test
    # hazard: probability that a block that has run d trials ends now
    dur = np.zeros(D + 1); dur[lo:hi + 1] = 1.0 / (hi - lo + 1)
    surv = 1 - np.cumsum(dur)
    haz = np.array([dur[d] / max(surv[d - 1] if d > 0 else 1.0, 1e-12) for d in range(D)])
    preds = []
    for c, o in zip(cues, outs):
        # transition: blocks of length d end with hazard haz[d]
        new = np.zeros_like(p)
        for d in range(D):
            h = min(1.0, haz[d]) if d >= lo else 0.0
            if d + 1 < D:
                new[:, d + 1] += p[:, d] * (1 - h)
            new[::-1, 1] += p[:, d] * h                # rule flips; the new block has its first trial
        p = new / new.sum()
        pr = p[0].sum()                                # P(rule = +1)
        m = {0: 1.0, 1: 2 * pr - 1, 2: -(2 * pr - 1)}[int(c)]
        preds.append((2 * Q - 1) * m)
        # observe
        like = np.array([[Q if MAP[int(c)](r) == o else 1 - Q] for r in (1, -1)])
        p = p * like; p /= p.sum()
    return np.array(preds)


def plastic(cues, outs, alpha, w=None):
    w = np.zeros(3) if w is None else w.copy(); preds = []
    for c, o in zip(cues, outs):
        preds.append(w[c]); w[c] += alpha * (o - w[c])
    return np.array(preds), w


def best_alpha(cues, outs):
    grid = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
    err = [np.mean((plastic(cues, outs, a)[0][100:] - outs[100:]) ** 2) for a in grid]
    return grid[int(np.argmin(err))]


def scalar_gate(cues, outs, eta_g, lr=0.01, state=None, learn=True):
    """The same algorithm with no waves: prediction w_c + g v_c, error e = pred - B."""
    w, v, g = (np.zeros(3), np.zeros(3), 1.0) if state is None else (state[0].copy(), state[1].copy(), state[2])
    preds = []
    for c, o in zip(cues, outs):
        p = w[c] + g * v[c]; preds.append(p); e = p - o
        gn = float(np.clip(g - eta_g * e * v[c] / (v[c] ** 2 + 0.1), -1, 1))   # the same normalized step
        if learn:
            w[c] -= lr * e; v[c] -= lr * g * e
        g = gn
    return np.array(preds), (w, v, g)


def body(seed, eta_g, eta_g_scalar=None):
    rng = np.random.default_rng(seed)
    U = M.Units()
    gap = int(rng.integers(28, 43)); amp = float(rng.uniform(0.9, 1.12))
    tr = sequence(rng, TRAIN); te = sequence(rng, TEST)
    jit = rng.integers(-5, 6, TRAIN + TEST); gj = rng.integers(-3, 4, TRAIN + TEST)
    W = np.zeros((U.n, 8)); V = np.zeros((U.n, 8)); g = 1.0     # g starts at +1: breaks the sign symmetry
    gtrace = []
    for i, (c, o) in enumerate(zip(tr[0], tr[2])):
        on = 20 + int(jit[i]); bo = on + gap + int(gj[i])
        r = M.run(U, W, V, g, int(c), on, int(o), bo, amp, learn=True, update_g=True, eta_g=eta_g, lr=LR)
        W, V, g = r["W"], r["V"], r["g"]; gtrace.append(g)
    b0 = PROBE_ONSET + gap
    cues, signs, outs, rules = te
    pred, pred_cut, gs = [], [], []
    for i, (c, o) in enumerate(zip(cues, outs)):
        pred.append(M.predicted_b(M.run(U, W, V, g, int(c), PROBE_ONSET, amp=amp)["prediction"], b0, amp))
        pred_cut.append(M.predicted_b(M.run(U, W, V, g, int(c), PROBE_ONSET, amp=amp, apical=False)["prediction"], b0, amp))
        j = TRAIN + i; on = 20 + int(jit[j]); bo = on + gap + int(gj[j])
        g = M.run(U, W, V, g, int(c), on, int(o), bo, amp, update_g=True, eta_g=eta_g)["g"]; gs.append(g)
    pred, pred_cut = np.array(pred), np.array(pred_cut)
    pb = bayes(cues, outs)
    alpha = best_alpha(tr[0], tr[2])
    w0 = plastic(tr[0], tr[2], alpha)[1]
    pp = plastic(cues, outs, alpha, w0)[0]
    eg = eta_g_scalar if eta_g_scalar is not None else 0.5
    _, st = scalar_gate(tr[0], tr[2], eg)
    ps = scalar_gate(cues, outs, eg, state=st, learn=False)[0]
    # per-cue weights, summed over that cue's units: where did learning put each cue?
    wsum = [float((W[U.cue == c].sum(1) / 3).sum()) for c in range(3)]
    vsum = [float((V[U.cue == c].sum(1) / 3).sum()) for c in range(3)]
    return {"seed": seed, "gap": gap, "amp": amp, "alpha": alpha, "eta_g": eta_g,
            "cues": cues.tolist(), "signs": signs.tolist(), "outs": outs.tolist(), "rules": rules.tolist(),
            "pred": pred.tolist(), "pred_cut": pred_cut.tolist(), "g": gs, "g_train_last": gtrace[-50:],
            "bayes": pb.tolist(), "plastic": pp.tolist(), "scalar": ps.tolist(),
            "W_by_cue": wsum, "V_by_cue": vsum}
