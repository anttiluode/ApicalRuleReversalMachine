"""Per-body measures, used for development and for the batch alike."""
import numpy as np


def transfer_events(cues, rules):
    """Trials that are the first of their R cue since the last rule switch, with the OTHER R cue already
    seen (and its outcome shown) since that switch. Only a shared rule state can get these right."""
    ev = []; start = 0
    for i in range(1, len(cues) + 1):
        if i == len(cues) or rules[i] != rules[i - 1]:
            seen = set()
            for j in range(start, i):
                c = int(cues[j])
                if c in (1, 2) and c not in seen:
                    if (3 - c) in seen and start > 0:
                        ev.append(j)
                    seen.add(c)
            start = i
    return ev


def measures(b):
    c = np.array(b["cues"]); s = np.array(b["signs"]); rules = np.array(b["rules"])
    P = {k: np.array(b[k]) for k in ("pred", "pred_cut", "bayes", "plastic", "scalar")}
    R = c > 0; F = c == 0
    ev = transfer_events(c, rules)
    out = {"n_transfer": len(ev)}
    for k, x in P.items():
        out[k + "_acc_R"] = float(np.mean(np.sign(x[R]) == s[R]))
        out[k + "_acc_F"] = float(np.mean(np.sign(x[F]) == s[F]))
        out[k + "_mse_bayes"] = float(np.mean((x - P["bayes"]) ** 2))
        out[k + "_transfer"] = float(np.mean(np.sign(x[ev]) == s[ev])) if ev else float("nan")
        out[k + "_absR"] = float(np.mean(np.abs(x[R]))); out[k + "_absF"] = float(np.mean(np.abs(x[F])))
    Wc, Vc = np.abs(b["W_by_cue"]), np.abs(b["V_by_cue"])
    out["sorted"] = bool(Wc[0] > Vc[0] and Vc[1] > Wc[1] and Vc[2] > Wc[2])
    return out
