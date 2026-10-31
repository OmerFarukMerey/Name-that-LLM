"""exp025: mercurydust's "Name That LLM-Submission" (public 0.660, Roman Tamrazov's 0.654 + "V31" changes), its last two
stages out of fold: the softer Sinkhorn and the reworked typography step, against Roman's.

V31's Sinkhorn: alpha 0.45 towards target marginals 0.65 x the predicted + 0.35 x the train class marginals (Roman:
alpha 1 towards even ones). V31's typography ("v2"): Roman's 34 markers + '###', '```', '> '; class rates shrunk to the
global rate, (hits + 24 p) / (n + 24), clipped to [0.01, 0.99]; each marker weighted by the spread of its class rates
(relative to the median, clipped to [0.25, 2]); evidence = weighted log-likelihood ratio against the global rates,
centred per row, x 1.5 / sum of weights, clipped to +-0.30; gate = clip((0.6 - margin) / 0.6) x clip(sqrt(words / 80),
0.4, 1) x clip(sqrt((markers present + 1) / 4), 0.45, 1). Roman's ("v38a", exp020): (hits + 2) / (n + 4), mean
log-likelihood x 2 clipped to +-0.35, gate (0.8 - margin) / 0.8. Rates are fitted on the other folds. Out-of-fold rows
carry no train-test shift, so the Sinkhorn change can only show its cost here, not the LB gain; the typography step
can show both. Base: exp020's no-post replica (core + deb).

    PYTHONPATH=src python experiments/exp025_v31_typography_oof.py
"""

import re

import numpy as np
from scipy.special import softmax

import exp020_typobayes_oof as T  # noqa: E402  (appends --plain and imports exp015 as T.E)
import exp021_typobayes_tuning as U  # noqa: E402
from ntl.data import load_train
from ntl.metrics import balanced_accuracy
from ntl.oofbank import paired_bootstrap

E = T.E
MARKS_V31 = T.MARKS + ['###', '```', '> ']


def soft_sinkhorn(P, prior, alpha=0.45, rho=0.35, iters=100):
    """V31's balancing; prior = train class marginals (rho 0: the predicted marginals, i.e. no balancing). Roman's is
    alpha 1, rho 1 with an even prior."""
    marg = np.clip(P.mean(0), 1e-6, None); marg /= marg.sum()
    target = np.clip((1 - rho) * marg + rho * prior, 1e-6, None); target /= target.sum()
    Q = P.copy()
    for _ in range(iters):
        Q *= len(Q) * target[None, :] / np.clip(Q.sum(0, keepdims=True), 1e-12, None)
        Q /= np.clip(Q.sum(1, keepdims=True), 1e-12, None)
    return (1 - alpha) * np.log(P + 1e-12) + alpha * np.log(Q + 1e-12)


def v2_evidence(Xfit, yfit, X, kappa=24.0, weights=True):
    n = np.bincount(yfit, minlength=18).astype(np.float64)
    hits = np.stack([Xfit[yfit == c].sum(0) for c in range(18)]).astype(np.float64)
    g = (Xfit.sum(0) + 2.0) / (len(Xfit) + 4.0)
    rate = np.clip((hits + kappa * g[None, :]) / (n[:, None] + kappa), 0.01, 0.99); g = np.clip(g, 0.01, 0.99)
    sig = np.sqrt(np.mean((rate - g[None, :]) ** 2, 0))
    fw = np.clip(sig / max(float(np.median(sig[sig > 0])) if np.any(sig > 0) else 1.0, 1e-6), 0.25, 2.0) if weights \
        else np.ones(X.shape[1])
    ev = (X * fw) @ np.log(rate).T + ((1 - X) * fw) @ np.log1p(-rate).T
    ev -= ((X * fw) @ np.log(g) + ((1 - X) * fw) @ np.log1p(-g))[:, None]
    ev -= ev.mean(1, keepdims=True)
    return np.clip(1.5 * ev / max(float(fw.sum()), 1e-6), -0.30, 0.30)


def oof(fn, X, y, folds, **kw):
    ev = np.zeros((len(X), 18))
    for f in np.unique(folds):
        fit = folds != f
        ev[~fit] = fn(X[fit], y[fit], X[~fit], **kw)
    return ev


def v38a_ev(Xfit, yfit, X):
    return np.clip(2.0 * U.evidence(X, U.rates(Xfit, yfit)), -0.35, 0.35)


def margin(raw):
    s = np.sort(softmax(raw, 1), 1)
    return s[:, -1] - s[:, -2]


def gate_v38a(raw, X, words):
    return np.clip((0.8 - margin(raw)) / 0.8, 0., 1.)


def gate_v2(raw, X, words, g=0.6):
    return (np.clip((g - margin(raw)) / g, 0., 1.) * np.clip(np.sqrt(np.maximum(words, 1.) / 80.), 0.40, 1.)
            * np.clip(np.sqrt((X.sum(1) + 1.) / 4.), 0.45, 1.))


def uneven_pools(y, conc, size=2000, draws=40, seed=0):
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y == c) for c in range(18)]
    for _ in range(draws):
        counts = rng.multinomial(size, rng.dirichlet([conc] * 18))
        yield np.concatenate([rng.choice(ix, k) for ix, k in zip(by_class, counts) if k])


def main():
    train = load_train()
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts, texts = train.prompt_id.to_numpy(), train.text.to_numpy()
    nlm = np.load(E.NLM_OOF); folds = nlm["folds"]
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    members["svc4"] = np.load(E.OUT / "svc4_oof.npy").astype(np.float64)
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
    assert np.array_equal(z["ids"], train.id.to_numpy(dtype=str))
    members["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    P0 = E.blend(members, {**E.CORE, "deb": 1.0})
    prior = (np.bincount(y, minlength=18) + 1.0) / (len(y) + 18.0)
    words = np.array([len(re.findall(r'\b\w+\b', t)) for t in texts], dtype=np.float64)
    X34, X37 = U.markers(texts, T.MARKS), U.markers(texts, MARKS_V31)

    steps = {   # name -> (evidence, gate)
        "none": (np.zeros((len(y), 18)), gate_v38a),
        "v38a (Roman)": (oof(v38a_ev, X34, y, folds), gate_v38a),
        "v2 (V31)": (oof(v2_evidence, X37, y, folds), gate_v2),
        "v2, Roman's gate": (oof(v2_evidence, X37, y, folds), gate_v38a),
        "v2, margin gate only (0.6)": (oof(v2_evidence, X37, y, folds),
                                       lambda r, X, w: np.clip((0.6 - margin(r)) / 0.6, 0., 1.)),
        "v2, no marker weights": (oof(v2_evidence, X37, y, folds, weights=False), gate_v2),
        "v2 on Roman's 34 markers": (oof(v2_evidence, X34, y, folds), gate_v2),
    }
    Xof = {"v2 on Roman's 34 markers": X34, "v38a (Roman)": X34}
    bal = {"Roman: Sinkhorn 1.0, even": lambda P, pr: soft_sinkhorn(P, np.full(18, 1 / 18), 1.0, 1.0),
           "V31: Sinkhorn 0.45, target mix 0.35": lambda P, pr: soft_sinkhorn(P, pr, 0.45, 0.35)}

    def predict(raw, step, rows=slice(None)):
        ev, gate = steps[step]
        X = Xof.get(step, X37)[rows]
        return (raw + gate(raw, X, words[rows])[:, None] * ev[rows]).argmax(1)

    print("all OOF rows (even mix):")
    ref = predict(bal["V31: Sinkhorn 0.45, target mix 0.35"](P0, prior), "v2 (V31)")
    for bname, b in bal.items():
        raw = b(P0, prior)
        for sname in steps:
            p = predict(raw, sname); c = paired_bootstrap(y, p, ref, prompts)
            print(f"  {bname:38s} {sname:28s} {balanced_accuracy(y, p):.4f}  vs V31 {c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]"
                  f"  changed {(p != ref).mean():.3f}")
    for conc in (100, 20, 5, 1):
        res = {}
        for r in uneven_pools(y, conc):
            for bname, b in bal.items():
                raw = b(P0[r], prior)
                for sname in ("none", "v38a (Roman)", "v2 (V31)", "v2, margin gate only (0.6)"):
                    res.setdefault((bname[:5], sname), []).append(balanced_accuracy(y[r], predict(raw, sname, r)))
        print(f"Dirichlet({conc:>3}) pools, mean of 40: " + "  ".join(f"{b}/{s[:12]} {np.mean(v):.4f}" for (b, s), v in res.items()))


if __name__ == "__main__":
    main()
