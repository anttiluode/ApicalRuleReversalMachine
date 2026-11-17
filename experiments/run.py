"""The pre-registered batch (PROTOCOL.md): 32 bodies, seeds 400001-400032. About 9 minutes on two cores.
    python experiments/run.py
"""
import hashlib, json, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ruleworld as R

SEEDS = list(range(400001, 400033))


def job(s):
    return R.body(s, R.ETA_G, eta_g_scalar=R.ETA_G)


if __name__ == "__main__":
    files = ["PROTOCOL.md", "experiments/run.py", "experiments/ruleworld.py", "experiments/metrics.py",
             "wavecomputer/rulemachine.py", "wavecomputer/nodes.py", "wavecomputer/physics.py",
             "wavecomputer/units.json", "wavecomputer/kernel.npz"]
    rec = {"label": "ApicalRuleReversalMachine batch (pre-registered)", "seeds": SEEDS, "eta_g": R.ETA_G,
           "hashes": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}, "results": []}
    t0 = time.time()
    with ProcessPoolExecutor(2) as p:
        for n, r in enumerate(p.map(job, SEEDS), 1):
            rec["results"].append(r); rec["elapsed_seconds"] = time.time() - t0
            (HERE / "results.json").write_text(json.dumps(rec))
            print("%d/%d %.0fs" % (n, len(SEEDS), time.time() - t0), flush=True)
