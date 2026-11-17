"""Development bodies (seeds 981-986), apical rate eta_g in {0.3, 0.5, 1.0} (normalized step, slow weights). Not evidence."""
import json, pathlib, sys
from concurrent.futures import ProcessPoolExecutor
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ruleworld as R


def job(a):
    s, e = a
    return R.body(s, e, eta_g_scalar=e)


if __name__ == "__main__":
    jobs = [(s, e) for e in (0.3, 0.5, 1.0) for s in range(981, 987)]
    with ProcessPoolExecutor(2) as p:
        res = list(p.map(job, jobs))
    (HERE / "dev_results.json").write_text(json.dumps({"label": "development (not evidence)", "results": res}))
