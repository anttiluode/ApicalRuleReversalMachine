"""The gated branches reduce exactly to WaveComputer's verified ones when the apical compartment is off.

With the apical compartment silenced (the cut), rulemachine.run must give the same prediction field and the same learned W as the same trial
built from wavecomputer.nodes (FadeClocks + Branches, themselves bit-identical to the vendored nodes; see
WaveComputer tests/test_equivalence.py).        python tests/test_engine.py
"""
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from wavecomputer import rulemachine as M, nodes as Nd
from wavecomputer.physics import Field, passive, PING_AMP


def reference(U, W, cue, onset, b, b_onset, amp, steps, lr):
    tb = passive([(M.SITE_B, b_onset, amp * b)], steps)
    target = np.zeros_like(tb); target[:, 3:6] = tb[:, 3:6]
    sense = M.cue_wave(amp, onset, steps); mine = U.cue == cue
    clocks = Nd.FadeClocks(U.decay, U.arm, U.rel, PING_AMP, M.REFRACTORY)
    br = Nd.Branches(W, lr, 3, 5, 0.71, 2.0, 1.0, learn=True); field = Field(steps)
    closed = np.zeros(U.n, bool); last = np.zeros(U.n)
    pp = np.zeros(8); pulse = np.zeros(U.n); po = np.zeros((U.n, 8)); P = np.zeros((steps, 8))
    for t in range(steps):
        pt = target[t - 1] if t > 0 else np.zeros(8)
        x = np.zeros(U.n)
        if t > 0:
            x[mine] = max(0.0, sense[t - 1])
        x[closed] = 0
        pred = field.step(np.where(U.mask, po, 0).sum(0)); fire = clocks.step(x)
        out = br.step(pulse, np.where(U.mask, (pp - pt)[None, :], 0.0))
        closed = np.where(fire, False, closed | (clocks.armed & (clocks.trace < last))); last = clocks.trace.copy()
        pp = pred.copy(); pulse = clocks.pulse.copy(); po = out.copy(); P[t] = pred
    return P, np.where(U.mask, br.W, 0.0)


if __name__ == "__main__":
    U = M.Units(); rng = np.random.default_rng(5); ok = True
    W = np.where(U.mask, rng.normal(0, 0.05, (U.n, 8)), 0.0)
    for cue, b in ((0, 1), (1, -1), (2, 1)):
        Pa, Wa = reference(U, W, cue, 20, b, 55, 1.03, M.STEPS, 0.03)
        V0 = np.where(U.mask, rng.normal(0, 0.05, (U.n, 8)), 0.0)
        r = M.run(U, W, V0, 0.7, cue, 20, b, 55, 1.03, learn=True, update_g=True, eta_g=0.5, lr=0.03, apical=False)
        same = np.allclose(Pa, r["prediction"], atol=1e-12) and np.allclose(Wa, r["W"], atol=1e-12)
        ok &= same and r["g"] == 0.7 and np.array_equal(r["V"], V0)
        print("cue %s, outcome %+d: field and W identical: %s; V and g untouched: %s"
              % (M.CUES[cue], b, same, r["g"] == 0.7 and np.array_equal(r["V"], V0)))
    print("REDUCES EXACTLY" if ok else "MISMATCH"); sys.exit(0 if ok else 1)
