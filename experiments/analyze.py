"""Score the batch against PROTOCOL.md. Writes summary.json (or dev_summary.json).
    python experiments/analyze.py [results.json]
"""
import json, math, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics import measures


def summarize(path, eta=None):
    res = json.loads(pathlib.Path(path).read_text())["results"]
    if eta is not None:
        res = [b for b in res if b["eta_g"] == eta]
    ms = [measures(b) for b in res]; n = len(ms); need = math.ceil(0.9 * n)
    col = lambda k: np.array([m[k] for m in ms])
    med = lambda k: float(np.nanmedian(col(k)))
    S = {"n": n, "need_90pct": need, "rules": {}, "report": {}}
    R = S["rules"]
    R["S1_sorted"] = {"bodies": int(col("sorted").sum()), "pass": bool(col("sorted").sum() >= need)}
    cutR, cutF = col("pred_cut_absR") / col("pred_absR"), col("pred_cut_absF") / col("pred_absF")
    R["C1_cut"] = {"cut_R_sign_accuracy": med("pred_cut_acc_R"), "cut_F_sign_accuracy": med("pred_cut_acc_F"),
                   "cut_over_intact_R_size": float(np.median(cutR)), "cut_over_intact_F_size": float(np.median(cutF)),
                   "pass": bool(0.35 <= med("pred_cut_acc_R") <= 0.65 and med("pred_cut_acc_F") >= 0.95
                                and np.median(cutR) <= 0.2 and np.median(cutF) >= 0.9)}
    R["I1_inference"] = {"R_sign_accuracy": med("pred_acc_R"), "bayes": med("bayes_acc_R"), "pass": bool(med("pred_acc_R") >= 0.80)}
    above = int((col("pred_transfer") > col("plastic_transfer")).sum())
    R["T1_transfer"] = {"machine": med("pred_transfer"), "plastic": med("plastic_transfer"), "bayes": med("bayes_transfer"),
                        "machine_above_plastic": above,
                        "pass": bool(med("pred_transfer") >= 0.75 and med("plastic_transfer") <= 0.25 and above >= need)}
    beat = int((col("pred_mse_bayes") < col("plastic_mse_bayes")).sum())
    R["B1_vs_plastic"] = {"machine_mse": med("pred_mse_bayes"), "plastic_mse": med("plastic_mse_bayes"), "machine_better": beat,
                          "pass": bool(beat >= need)}
    for k in ("pred", "pred_cut", "bayes", "plastic", "scalar"):
        S["report"][k] = {x: med(k + x) for x in ("_acc_R", "_acc_F", "_mse_bayes", "_transfer", "_absR", "_absF")}
    S["report"]["scalar_better_than_machine_mse"] = int((col("scalar_mse_bayes") < col("pred_mse_bayes")).sum())
    S["report"]["transfer_events_median"] = med("n_transfer")
    S["all_rules_pass"] = all(v["pass"] for v in R.values())
    return S


if __name__ == "__main__":
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results.json"
    S = summarize(path, eta=0.5 if path.name == "dev_results.json" else None)
    (HERE / ("summary.json" if path.name == "results.json" else "dev_summary.json")).write_text(json.dumps(S, indent=1))
    for k, v in S["rules"].items():
        print("%-14s %s %s" % (k, "PASS" if v["pass"] else "FAIL", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "pass"}))
    print("all rules pass:", S["all_rules_pass"])
