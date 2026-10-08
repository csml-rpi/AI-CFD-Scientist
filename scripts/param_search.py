"""Ask-tell coefficient search with an evaluation budget that is enforced.

This holds the search loop and enforces the evaluation cap. You choose the
backend and supply the bounds and the budget; it decides which point to try
next and stops when the budget is spent.

CHOOSING A BACKEND, with d parameters and a budget of n evaluations

  bo      Gaussian-process surrogate plus expected improvement. For few,
          EXPENSIVE evaluations: n up to ~100, d up to ~10. It spends
          computation choosing the next point so that it needs fewer of them.
          Use this when one evaluation is a solver run.

  cmaes   Covariance Matrix Adaptation Evolution Strategy. For many parameters
          (d >= 5) AND a large budget (n > ~100), or when parameters are
          correlated or badly scaled -- it learns the shape of the landscape
          instead of assuming the axes are independent. Below ~100 evaluations
          it is still estimating its covariance matrix.

  random  Uniform sampling. Correct when n < 2d: too few points for any model
          of the landscape to mean anything.

      n < 2d           -> random
      n <= 100         -> bo
      n > 100, d >= 5  -> cmaes

USAGE -- every call takes --state, a JSON file holding the whole search

    python3 param_search.py init --state s.json --backend bo \
        --bounds '{"<coeff>": [<low>, <high>], ...}' --budget <n> --direction min

    python3 param_search.py ask  --state s.json
        -> {"id": 0, "params": {"<coeff>": 0.31, ...}, "remaining": 39}

    python3 param_search.py tell --state s.json --id 0 --score 0.00412
        a run that produced no score:  --id 0 --failed

    python3 param_search.py best   --state s.json
    python3 param_search.py status --state s.json

`ask` refuses once the budget is spent. When it does, take `best`, deploy those
values and report -- do not start a second search. Everything lives in the
state file, so an interrupted search resumes by calling `ask` again.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

def _load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"no search state at {path} -- run `init` first")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"search state at {path} is not valid JSON: {exc}")


def _save(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def _names(state: Dict[str, Any]) -> List[str]:
    return list(state["bounds"].keys())


def _lo_hi(state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    b = state["bounds"]
    lo = np.array([b[k][0] for k in b], dtype=float)
    hi = np.array([b[k][1] for k in b], dtype=float)
    return lo, hi


def _to_unit(state: Dict[str, Any], params: Dict[str, float]) -> np.ndarray:
    lo, hi = _lo_hi(state)
    x = np.array([float(params[k]) for k in _names(state)], dtype=float)
    return np.clip((x - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0, 1.0)


def _from_unit(state: Dict[str, Any], u: np.ndarray) -> Dict[str, float]:
    lo, hi = _lo_hi(state)
    x = lo + np.clip(u, 0.0, 1.0) * (hi - lo)
    return {k: float(v) for k, v in zip(_names(state), x)}


def _scored(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Trials with a finite score. A failed evaluation says nothing about the
    landscape, so it is excluded from fitting rather than given a made-up bad
    value -- an invented penalty is a data point the surrogate then believes."""
    out = []
    for t in state["trials"]:
        s = t.get("score")
        if isinstance(s, (int, float)) and math.isfinite(s):
            out.append(t)
    return out


def _sign(state: Dict[str, Any]) -> float:
    """+1 when lower is better. Everything inside minimises sign*score."""
    return 1.0 if state.get("direction", "min") == "min" else -1.0


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

def _ask_random(state: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    return rng.random(len(_names(state)))


def _ask_bo(state: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Expected improvement over a Gaussian-process surrogate.

    Falls back to random sampling while there are too few points to fit
    anything: a GP on two observations is a straight line with enormous error
    bars, and its argmax is noise dressed as a decision.
    """
    d = len(_names(state))
    done = _scored(state)
    n_init = max(4, 2 * d)
    if len(done) < n_init:
        return _ask_random(state, rng)

    try:
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    except ImportError:
        return _ask_random(state, rng)

    X = np.array([_to_unit(state, t["params"]) for t in done])
    y = np.array([_sign(state) * float(t["score"]) for t in done])

    # Matern 5/2: twice differentiable, the standard choice for a physical
    # response surface. WhiteKernel absorbs solver noise, so the GP is not
    # forced to interpolate every point exactly.
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.full(d, 0.3), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-12, 1e-1))
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
                                  random_state=int(rng.integers(0, 2 ** 31 - 1)))
    try:
        gp.fit(X, y)
    except Exception:  # noqa: BLE001 -- a failed fit is a reason to sample, not to crash
        return _ask_random(state, rng)

    best = float(np.min(y))
    # Acquisition maximised by dense random search plus a local cloud around
    # the incumbent. Cheap, and the surface is only d-dimensional.
    cand = rng.random((4096, d))
    cand = np.vstack([cand, np.clip(X[int(np.argmin(y))] + 0.05 * rng.normal(size=(256, d)), 0, 1)])
    mu, sd = gp.predict(cand, return_std=True)
    sd = np.maximum(sd, 1e-12)
    imp = best - mu - 0.01          # small exploration margin
    z = imp / sd
    ei = imp * norm.cdf(z) + sd * norm.pdf(z)
    ei[sd < 1e-10] = 0.0
    if not np.any(ei > 0):
        return _ask_random(state, rng)
    return cand[int(np.argmax(ei))]


def _cma_init(d: int, sigma0: float, seed: int) -> Dict[str, Any]:
    lam = 4 + int(3 * math.log(d))
    mu = max(1, lam // 2)
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / w.sum()
    mueff = float(1.0 / np.sum(w ** 2))
    cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
    cs = (mueff + 2) / (d + mueff + 5)
    c1 = 2 / ((d + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (d + 1)) - 1) + cs
    return {
        "d": d, "lam": lam, "mu": mu, "w": w.tolist(), "mueff": mueff,
        "cc": cc, "cs": cs, "c1": c1, "cmu": cmu, "damps": damps,
        "chiN": math.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2)),
        "mean": [0.5] * d, "sigma": float(sigma0),
        "C": np.eye(d).tolist(), "pc": [0.0] * d, "ps": [0.0] * d,
        "gen": 0, "seed": seed, "pending": [],
    }


def _ask_cmaes(state: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """One sample from the current generation.

    The ask-tell interface hands out a point at a time; CMA only learns once a
    whole generation has been scored, so a generation is buffered and closed in
    `_cma_update`.
    """
    c = state["cma"]
    d = c["d"]
    if not c["pending"]:
        C = np.array(c["C"], dtype=float)
        C = (C + C.T) / 2.0
        try:
            vals, vecs = np.linalg.eigh(C)
            A = vecs @ np.diag(np.sqrt(np.maximum(vals, 1e-20))) @ vecs.T
        except np.linalg.LinAlgError:
            A = np.eye(d)
        mean = np.array(c["mean"], dtype=float)
        pend = []
        for _ in range(c["lam"]):
            z = rng.normal(size=d)
            x = mean + c["sigma"] * (A @ z)
            pend.append({"z": z.tolist(), "x": np.clip(x, 0.0, 1.0).tolist(),
                         "raw": x.tolist(), "score": None, "handed_out": False})
        c["pending"] = pend
    for p in c["pending"]:
        if not p.get("handed_out"):
            p["handed_out"] = True
            return np.array(p["x"], dtype=float)
    # The whole generation is out for evaluation. Resample rather than stall;
    # the generation closes when its scores arrive.
    return _ask_random(state, rng)


def _cma_update(state: Dict[str, Any]) -> None:
    """Close a generation once every point in it has a score."""
    c = state["cma"]
    pend = c["pending"]
    if not pend or any(p["score"] is None for p in pend):
        return
    d, mu = c["d"], c["mu"]
    w = np.array(c["w"], dtype=float)
    finite = [p for p in pend if math.isfinite(p["score"])]
    if len(finite) < mu:
        # Too many failed evaluations to rank a generation. Drop it and keep
        # the distribution -- better than updating towards noise.
        c["pending"] = []
        return
    finite.sort(key=lambda p: p["score"])
    sel = finite[:mu]
    old_mean = np.array(c["mean"], dtype=float)
    X = np.array([p["raw"] for p in sel], dtype=float)
    mean = (w[:, None] * X).sum(axis=0)

    C = np.array(c["C"], dtype=float)
    try:
        vals, vecs = np.linalg.eigh((C + C.T) / 2.0)
        invsqrtC = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-20))) @ vecs.T
    except np.linalg.LinAlgError:
        invsqrtC = np.eye(d)

    ps = np.array(c["ps"], dtype=float)
    pc = np.array(c["pc"], dtype=float)
    sigma = float(c["sigma"])
    cs, cc, c1, cmu = c["cs"], c["cc"], c["c1"], c["cmu"]
    mueff, chiN, damps = c["mueff"], c["chiN"], c["damps"]

    ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (mean - old_mean) / sigma)
    c["gen"] += 1
    hsig = bool(np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * c["gen"])) / chiN
                < 1.4 + 2 / (d + 1))
    pc = (1 - cc) * pc + (math.sqrt(cc * (2 - cc) * mueff) if hsig else 0.0) * (mean - old_mean) / sigma

    Y = (X - old_mean) / sigma
    rank_mu = sum(w[i] * np.outer(Y[i], Y[i]) for i in range(mu))
    C = ((1 - c1 - cmu) * C
         + c1 * (np.outer(pc, pc) + (0.0 if hsig else cc * (2 - cc)) * C)
         + cmu * rank_mu)
    sigma = float(np.clip(sigma * math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1)),
                          1e-6, 1.0))

    c.update(mean=mean.tolist(), C=C.tolist(), pc=pc.tolist(), ps=ps.tolist(),
             sigma=sigma, pending=[])


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_init(a: argparse.Namespace) -> int:
    try:
        bounds = json.loads(a.bounds)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--bounds is not valid JSON: {exc}")
    if not isinstance(bounds, dict) or not bounds:
        raise SystemExit("--bounds must be a non-empty object, e.g. "
                         "'{\"<coeff>\": [<low>, <high>]}'")
    clean: Dict[str, List[float]] = {}
    for k, v in bounds.items():
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            raise SystemExit(f"bounds for {k!r} must be [low, high]")
        lo, hi = float(v[0]), float(v[1])
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            raise SystemExit(f"bounds for {k!r} must be finite with high > low")
        clean[str(k)] = [lo, hi]

    d = len(clean)
    backend = a.backend
    if backend == "auto":
        backend = "random" if a.budget < 2 * d else ("bo" if a.budget <= 100 else "cmaes")
    if backend not in ("bo", "cmaes", "random"):
        raise SystemExit("--backend must be one of: bo, cmaes, random, auto")

    state: Dict[str, Any] = {
        "backend": backend, "bounds": clean, "direction": a.direction,
        "budget": int(a.budget), "used": 0, "seed": int(a.seed), "trials": [],
    }
    if backend == "cmaes":
        state["cma"] = _cma_init(d, float(a.sigma0), int(a.seed))

    _save(Path(a.state), state)
    advice = ""
    if backend == "bo" and a.budget > 150:
        advice = " (note: past ~100 evaluations cmaes usually overtakes bo)"
    elif backend == "cmaes" and a.budget < 100:
        advice = (" (note: under ~100 evaluations cmaes is still estimating its "
                  "covariance matrix and bo is usually better)")
    print(json.dumps({
        "ok": True, "backend": backend, "parameters": d, "budget": int(a.budget),
        "direction": a.direction, "state": str(a.state),
        "note": f"{backend} over {d} parameters, {a.budget} evaluations{advice}",
    }, indent=2))
    return 0


def cmd_ask(a: argparse.Namespace) -> int:
    path = Path(a.state)
    state = _load(path)
    if state["used"] >= state["budget"]:
        print(json.dumps({
            "ok": False, "exhausted": True, "used": state["used"], "budget": state["budget"],
            "error": (f"This search has used all {state['budget']} evaluations it was given. "
                      "No further points will be handed out. Take the best result you have "
                      "(`best`), deploy it, and report."),
        }, indent=2))
        return 2
    rng = np.random.default_rng(state["seed"] + 1000 * state["used"])
    backend = state["backend"]
    if backend == "bo":
        u = _ask_bo(state, rng)
    elif backend == "cmaes":
        u = _ask_cmaes(state, rng)
    else:
        u = _ask_random(state, rng)

    params = _from_unit(state, np.asarray(u, dtype=float))
    trial = {"id": len(state["trials"]), "params": params, "score": None}
    state["trials"].append(trial)
    state["used"] += 1
    _save(path, state)
    print(json.dumps({
        "ok": True, "id": trial["id"], "params": params,
        "used": state["used"], "budget": state["budget"],
        "remaining": state["budget"] - state["used"],
    }, indent=2))
    return 0


def cmd_tell(a: argparse.Namespace) -> int:
    path = Path(a.state)
    state = _load(path)
    tid = int(a.id)
    if not (0 <= tid < len(state["trials"])):
        raise SystemExit(f"no trial with id {tid}; ids run 0..{len(state['trials']) - 1}")
    trial = state["trials"][tid]
    if a.failed or a.score is None or not math.isfinite(float(a.score)):
        trial["score"] = None
        trial["failed"] = True
    else:
        trial["score"] = float(a.score)
        trial["failed"] = False
    if state["backend"] == "cmaes":
        target = _to_unit(state, trial["params"])
        for p in state["cma"]["pending"]:
            if p.get("handed_out") and p.get("score") is None:
                if np.allclose(np.array(p["x"], dtype=float), target, atol=1e-9):
                    p["score"] = trial["score"] if trial["score"] is not None else float("inf")
                    break
        _cma_update(state)
    _save(path, state)
    done = _scored(state)
    best = min(done, key=lambda t: _sign(state) * t["score"]) if done else None
    print(json.dumps({
        "ok": True, "recorded": tid, "score": trial["score"],
        "scored_trials": len(done), "remaining": state["budget"] - state["used"],
        "best_so_far": (best or {}).get("score"),
    }, indent=2))
    return 0


def cmd_best(a: argparse.Namespace) -> int:
    state = _load(Path(a.state))
    done = _scored(state)
    if not done:
        print(json.dumps({"ok": False, "error": "no scored trials yet"}, indent=2))
        return 2
    b = min(done, key=lambda t: _sign(state) * t["score"])
    print(json.dumps({
        "ok": True, "id": b["id"], "score": b["score"], "params": b["params"],
        "scored_trials": len(done), "used": state["used"], "budget": state["budget"],
    }, indent=2))
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    state = _load(Path(a.state))
    done = _scored(state)
    b = min(done, key=lambda t: _sign(state) * t["score"]) if done else None
    print(json.dumps({
        "ok": True, "backend": state["backend"], "parameters": len(state["bounds"]),
        "direction": state["direction"], "used": state["used"], "budget": state["budget"],
        "remaining": state["budget"] - state["used"],
        "scored": len(done), "failed": state["used"] - len(done),
        "best_score": (b or {}).get("score"), "best_params": (b or {}).get("params"),
        "generation": state.get("cma", {}).get("gen"),
    }, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="param_search.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="start a search")
    i.add_argument("--state", required=True)
    i.add_argument("--backend", default="auto", help="bo | cmaes | random | auto")
    i.add_argument("--bounds", required=True, help='JSON: {"<coeff>": [<low>, <high>], ...}')
    i.add_argument("--budget", type=int, required=True, help="maximum evaluations -- enforced")
    i.add_argument("--direction", default="min", choices=("min", "max"))
    i.add_argument("--sigma0", type=float, default=0.3, help="cmaes initial step (unit cube)")
    i.add_argument("--seed", type=int, default=0)
    i.set_defaults(func=cmd_init)

    k = sub.add_parser("ask", help="get the next point to evaluate")
    k.add_argument("--state", required=True)
    k.set_defaults(func=cmd_ask)

    t = sub.add_parser("tell", help="report a point's score")
    t.add_argument("--state", required=True)
    t.add_argument("--id", required=True)
    t.add_argument("--score", type=float, default=None)
    t.add_argument("--failed", action="store_true", help="the evaluation produced no score")
    t.set_defaults(func=cmd_tell)

    b = sub.add_parser("best", help="best point so far")
    b.add_argument("--state", required=True)
    b.set_defaults(func=cmd_best)

    s = sub.add_parser("status", help="budget and progress")
    s.add_argument("--state", required=True)
    s.set_defaults(func=cmd_status)

    a = p.parse_args()
    return int(a.func(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
