"""exp061: the 0.704's Fable routing coefficient, out of fold.

The public 0.704 ranks the GLM stage's rows by GLM mass + 0.35 * min(P(claude-fable-5), GLM mass). On the LB the routing
is what carries its +0.002 over the soft 0.702 (#119: the other two changes without it read 0.703), and the blend weight
is flat on the routed line (W 1 / 1.25 / 1.5 all 0.704). This varies the coefficient (and drops the cap) at cover 0.15,
his head out of fold, W 1 and 1.25, non-GLM stage results reverted (exp058 revert); reference: the public 0.35.

    python -I experiments/exp061_routing_coef.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv.append("revert")
import exp058_stage_w_fable as S  # noqa: E402
import numpy as np  # noqa: E402

X = S.X
from ntl.cv import prompt_folds  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402

SETTINGS = [(0.0, True), (0.2, True), (0.35, True), (0.5, True), (0.75, True), (1.0, True), (0.35, False), (0.75, False)]


def blend(zl, ll, cover, W, h_ref, coef, cap):
    mass = np.exp(zl[:, X.GLM]).sum(1)
    fable = np.exp(zl[:, S.FABLE])
    priority = mass + coef * (np.minimum(fable, mass) if cap else fable)
    sel = np.argsort(-priority, kind="stable")[:int(cover * len(zl))]
    pred = zl.argmax(1)
    pred[sel] = (zl[sel] + W / X.match_temperature(zl, h_ref) * ll[sel]).argmax(1)
    return np.where(np.isin(pred, X.GLM), pred, zl.argmax(1)), sel


def main():
    train = load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    sf, _ = X.load_scores("flash", ids)
    sg, _ = X.load_scores("glm53", ids)
    names = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    X160, _ = X.design(X.features(sf, texts, 160), X.features(sg, texts, 160), names, X.HIS["diff_features"])
    folds = prompt_folds(prompts)
    nlm = np.load(X.E.NLM_OOF)
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    members["svc4"] = np.load(X.E.OUT / "svc4_oof.npy").astype(np.float64)
    zl = np.log(np.clip(X.E.sinkhorn(X.E.blend(members, X.E.CORE)), 1e-12, None))
    ll = X.oof_head(X160, y, folds, X.HIS["C"])
    h = X.HIS["entropy_ref"]
    score = lambda p: balanced_accuracy(train.model, np.asarray(X.LABELS)[p])
    glm = np.isin(y, X.GLM)
    print(f"stand-in stack {score(zl.argmax(1)):.4f}", flush=True)
    for W in (1.0, 1.25):
        ref, rsel = blend(zl, ll, 0.15, W, h, 0.35, True)
        print(f"\nW {W}: public routing (0.35, capped) {score(ref):.4f}", flush=True)
        for coef, cap in SETTINGS:
            p, sel = blend(zl, ll, 0.15, W, h, coef, cap)
            g = paired_bootstrap(y, p, ref, prompts, n=1000)
            print(f"  coef {coef:<4g} {'capped' if cap else 'uncapped':8s} {score(p):.4f}  vs 0.35: {g[0]:+.4f} "
                  f"[{g[1]:+.4f},{g[2]:+.4f}]  ({int((p != ref).sum())} rows differ; stage rows: GLM "
                  f"{glm[sel].mean():.2f}, fable {(y[sel] == S.FABLE).mean():.2f}, new vs public "
                  f"{len(np.setdiff1d(sel, rsel))})", flush=True)


if __name__ == "__main__":
    main()
