"""experiments/rulemachine.png from experiments/results.json."""
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import measures
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
C = {"machine": "#1a7f64", "cut": "#cf6a2e", "bayes": "#1f2328", "plastic": "#8c959f", "scalar": "#5b6cc4"}

d = json.loads((HERE / "results.json").read_text())["results"]
ms = [measures(b) for b in d]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig = plt.figure(figsize=(13, 7.9)); gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1])

# 1. one body: the rule, the apical state, and what it predicts for R1
b = d[0]; c = np.array(b["cues"]); rules = np.array(b["rules"]); g = np.array(b["g"])
p, pb, pp = np.array(b["pred"]), np.array(b["bayes"]), np.array(b["plastic"])
t = np.arange(len(c)); lo, hi = 0, 200
a = fig.add_subplot(gs[0, :])
a.fill_between(t[lo:hi], -1.15, 1.15, where=rules[lo:hi] > 0, color="#eef6f3", step="mid", lw=0)
a.plot(t[lo:hi], np.r_[g[lo], g[lo:hi - 1]], color=C["machine"], lw=1.2, alpha=.5, label="apical state g (before each trial)")
for cue, mk, sgn in ((1, "o", 1), (2, "s", -1)):
    i = np.flatnonzero(c[lo:hi] == cue) + lo
    a.scatter(i, sgn * p[i], marker=mk, s=16, color=C["machine"], zorder=3, label="machine, %s" % ("R1" if cue == 1 else "R2"))
    a.scatter(i, sgn * pb[i], marker="_", s=40, color=C["bayes"], zorder=4, label="ideal observer" if cue == 1 else None)
    a.scatter(i, sgn * pp[i], marker=".", s=18, color=C["plastic"], zorder=2, label="plastic learner" if cue == 1 else None)
a.axhline(0, color=GRID, lw=.8); a.set_ylim(-1.15, 1.15); a.set_xlim(lo, hi)
a.set_xlabel("test trial (weights frozen; shaded: rule = +1)"); a.set_ylabel("prediction read as the rule\n(R2 sign-flipped)")
a.set_title("One body: the rule flips without a cue; the apical state follows, and both reversing cues follow it", loc="left", color=INK, fontsize=10)
a.legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.17))

# 2. the cut
a = fig.add_subplot(gs[1, 0]); rng = np.random.default_rng(0)
cols = [("pred_acc_F", "intact\nF", C["machine"]), ("pred_cut_acc_F", "cut\nF", C["cut"]),
        ("pred_acc_R", "intact\nR1, R2", C["machine"]), ("pred_cut_acc_R", "cut\nR1, R2", C["cut"])]
for i, (k, lab, col) in enumerate(cols):
    v = np.array([m[k] for m in ms]); a.scatter(i + rng.uniform(-.15, .15, len(v)), v, s=8, color=col, alpha=.5, lw=0)
    a.plot([i - .3, i + .3], [np.median(v)] * 2, color=col, lw=2.5)
a.axhline(.5, color=INK, lw=.8, ls="--"); a.text(3.45, .5, "chance", va="center", fontsize=8)
a.set_xticks(range(4)); a.set_xticklabels([x[1] for x in cols]); a.set_ylim(0.3, 1.04); a.set_xlim(-.5, 4)
a.set_ylabel("sign accuracy"); a.set_title("Silence the apical compartment", loc="left", color=INK, fontsize=10)

# 3. transfer
a = fig.add_subplot(gs[1, 1])
for i, (k, lab, col) in enumerate([("pred", "machine", C["machine"]), ("scalar", "scalar\n(no waves)", C["scalar"]),
                                     ("bayes", "ideal", C["bayes"]), ("plastic", "plastic", C["plastic"])]):
    v = np.array([m[k + "_transfer"] for m in ms]); a.scatter(i + rng.uniform(-.15, .15, len(v)), v, s=8, color=col, alpha=.5, lw=0)
    a.plot([i - .3, i + .3], [np.median(v)] * 2, color=col, lw=2.5)
a.axhline(.5, color=INK, lw=.8, ls="--"); a.set_xticks(range(4)); a.set_xticklabels(["machine", "scalar\n(no waves)", "ideal", "plastic"])
a.set_ylim(-0.03, 1.05); a.set_ylabel("sign accuracy on transfer trials")
a.set_title("A cue not yet seen since the switch", loc="left", color=INK, fontsize=10)

# 4. where learning put each cue
a = fig.add_subplot(gs[1, 2])
W = np.abs([b["W_by_cue"] for b in d]); V = np.abs([b["V_by_cue"] for b in d])
for k, (lab, mk) in enumerate((("F (fixed)", "o"), ("R1 (reverses)", "s"), ("R2 (reverses)", "^"))):
    a.scatter(W[:, k], V[:, k], s=16, marker=mk, color=C["plastic"] if k == 0 else C["machine"], alpha=.7, label=lab, lw=0)
m = max(W.max(), V.max()) * 1.05; a.plot([0, m], [0, m], color=GRID, lw=.8)
a.set_xlim(0, m); a.set_ylim(0, m); a.set_xlabel("|basal weight W|"); a.set_ylabel("|apical-gated weight V|")
a.set_title("Learning sorted the cues itself", loc="left", color=INK, fontsize=10); a.legend(frameon=False, fontsize=8)
fig.text(.995, .005, "32 bodies · dots: bodies · bars: medians", ha="right", fontsize=8, color=MUTED)
fig.tight_layout(); fig.savefig(HERE / "rulemachine.png", dpi=150); print("ok")
