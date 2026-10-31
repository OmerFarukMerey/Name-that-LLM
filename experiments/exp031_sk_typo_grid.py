"""exp031: the two test-time knobs the public 0.664s moved, out of fold: V31's Sinkhorn strength alpha (exp028 on the LB:
0.60 0.658, 0.45 0.663, 0.30 0.664; judith007's 0.664 runs 0.60) and the scale of V31's typography evidence (1.5; judith007
uses 1.2). exp025's machinery: exp020's no-post replica (core + deb), V31's typography step with marker rates fitted on the
other folds, scored on all out-of-fold rows (even classes) and on Dirichlet-drawn uneven pools of 2,000 rows (40 draws
each), the only offline stand-in for the test's unequal class mix.

    PYTHONPATH=src:experiments python experiments/exp031_sk_typo_grid.py
"""

import json

import numpy as np

import exp025_v31_typography_oof as V   # noqa: E402  (imports exp020, which appends --plain)
from ntl.data import load_train
from ntl.metrics import balanced_accuracy
from ntl.paths import REPO_ROOT

E, U = V.E, V.U
OUT = REPO_ROOT / "outputs" / "exp031_sk_typo_grid"
ALPHAS = (0.20, 0.30, 0.45, 0.60)
SCALES = (1.0, 1.2, 1.5)


def _unclipped(Xfit, yfit, X, kappa=24.0):
    """exp025's v2_evidence before its clip to +-0.30 (it scales by 1.5 / sum of marker weights)."""
    n = np.bincount(yfit, minlength=18).astype(np.float64)
    hits = np.stack([Xfit[yfit == c].sum(0) for c in range(18)]).astype(np.float64)
    g = (Xfit.sum(0) + 2.0) / (len(Xfit) + 4.0)
    rate = np.clip((hits + kappa * g[None, :]) / (n[:, None] + kappa), 0.01, 0.99); g = np.clip(g, 0.01, 0.99)
    sig = np.sqrt(np.mean((rate - g[None, :]) ** 2, 0))
    fw = np.clip(sig / max(float(np.median(sig[sig > 0])) if np.any(sig > 0) else 1.0, 1e-6), 0.25, 2.0)
    ev = (X * fw) @ np.log(rate).T + ((1 - X) * fw) @ np.log1p(-rate).T
    ev -= ((X * fw) @ np.log(g) + ((1 - X) * fw) @ np.log1p(-g))[:, None]
    ev -= ev.mean(1, keepdims=True)
    return 1.5 * ev / max(float(fw.sum()), 1e-6)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_train()
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    texts = train.text.to_numpy()
    nlm = np.load(E.NLM_OOF, allow_pickle=False); folds = nlm["folds"]
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    members["svc4"] = np.load(E.OUT / "svc4_oof.npy").astype(np.float64)
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz", allow_pickle=False)
    members["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    P0 = E.blend(members, {**E.CORE, "deb": 1.0})
    prior = (np.bincount(y, minlength=18) + 1.0) / (len(y) + 18.0)
    words = np.array([len(V.re.findall(r'\b\w+\b', t)) for t in texts], dtype=np.float64)
    X37 = U.markers(texts, V.MARKS_V31)
    ev = {s: V.oof(lambda a, b, c: np.clip(s / 1.5 * _unclipped(a, b, c), -0.30, 0.30), X37, y, folds) for s in SCALES}
    assert np.allclose(ev[1.5], V.oof(V.v2_evidence, X37, y, folds))   # the scale-1.5 path is exp025's evidence

    def predict(raw, scale, rows):
        return (raw + V.gate_v2(raw, X37[rows], words[rows])[:, None] * ev[scale][rows]).argmax(1)

    R = {}
    pools = {"even (all OOF rows)": [np.arange(len(y))]}
    for conc in (20, 5, 1):
        pools[f"Dirichlet({conc})"] = list(V.uneven_pools(y, conc))
    for pname, rs in pools.items():
        print(f"\n{pname}: mean balanced accuracy over {len(rs)} pool(s)")
        print("            " + "  ".join(f"typo x{s:<4}" for s in SCALES))
        for a in ALPHAS:
            cells = []
            for s in SCALES:
                v = np.mean([balanced_accuracy(y[r], predict(V.soft_sinkhorn(P0[r], prior, a, 0.35), s, r)) for r in rs])
                R[f"{pname}|sk{a}|typo{s}"] = float(v); cells.append(f"{v:.4f}   ")
            print(f"  sk {a:.2f}   " + "  ".join(cells))
    json.dump(R, open(OUT / "metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
