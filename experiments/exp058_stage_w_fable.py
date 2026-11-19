"""exp058: the GLM stage's blend weight W under the public 0.704's Fable routing, out of fold.

Two public notebooks scored 0.704 from the 0.702 soft typography by different stage changes:
- lightningv08's 0.704 ranks the stage rows by GLM mass + 0.35 * min(P(claude-fable-5), GLM mass) (Fable routing);
- bill123987's ntl706w keeps the GLM-mass order and scales the blend weight to 1.25 W (and the sanity guard to 0.32).
exp050 found W 1.25 +0.002-0.003 with estopadilla's head and about neutral with our gbmpair head (GLM-mass order).
This reruns that comparison under both orders, at cover 0.15 (the public stage) and 0.22 (our c22), against
his head at W 1 with the same order and cover (paired prompt bootstrap).

    python -I experiments/exp058_stage_w_fable.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp050_glm_ll_head as X  # noqa: E402  (sets the paths and OMP threads)
import numpy as np  # noqa: E402

from ntl.cv import prompt_folds  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402

FABLE = X.LABELS.index("claude-fable-5")
COVERS = (0.15, 0.22)


def blend(zl, ll, cover, W, h_ref, fable):
    """v1_stage.py: the top `cover` rows by priority get argmax(zl + (W / T') ll)."""
    mass = np.exp(zl[:, X.GLM]).sum(1)
    priority = mass + 0.35 * np.minimum(np.exp(zl[:, FABLE]), mass) if fable else mass
    sel = np.argsort(-priority, kind="stable")[:int(cover * len(zl))]
    w = W / X.match_temperature(zl, h_ref)
    pred = zl.argmax(1)
    pred[sel] = (zl[sel] + w * ll[sel]).argmax(1)
    return pred


def main():
    train = load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    sf, _ = X.load_scores("flash", ids)
    sg, _ = X.load_scores("glm53", ids)
    names = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    X160, cols = X.design(X.features(sf, texts, 160), X.features(sg, texts, 160), names, X.HIS["diff_features"])
    assert cols == X.HIS["features"]
    pr = X.pair_features(sf, sg, 160)
    Xp = np.hstack([X160, np.array([[r[k] for k in pr[0]] for r in pr])])

    folds = prompt_folds(prompts)
    nlm = np.load(X.E.NLM_OOF)
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    members["svc4"] = np.load(X.E.OUT / "svc4_oof.npy").astype(np.float64)
    zl = np.log(np.clip(X.E.sinkhorn(X.E.blend(members, X.E.CORE)), 1e-12, None))
    score = lambda p: balanced_accuracy(train.model, np.asarray(X.LABELS)[p])
    h = X.HIS["entropy_ref"]
    print(f"stand-in stack {score(zl.argmax(1)):.4f}", flush=True)

    heads = {"his": X.oof_head(X160, y, folds, X.HIS["C"])}
    heads["gbmpair"] = X.mix(heads["his"], X.oof_gbm(Xp, y, folds))
    for fable in (False, True):
        order = "GLM mass + Fable routing" if fable else "GLM mass"
        for c in COVERS:
            ref = blend(zl, heads["his"], c, 1.0, h, fable)
            print(f"\n{order}, cover {c}: his head W 1 = {score(ref):.4f} "
                  f"(vs the stack {paired_bootstrap(y, ref, zl.argmax(1), prompts, n=1000)[0]:+.4f})", flush=True)
            for name, ll in heads.items():
                for W in (1.0, 1.25, 1.5):
                    if name == "his" and W == 1.0:
                        continue
                    p = blend(zl, ll, c, W, h, fable)
                    g = paired_bootstrap(y, p, ref, prompts, n=1000)
                    print(f"  {name:8s} W {W:<4g} {score(p):.4f}  vs his W 1: {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}]",
                          flush=True)
    # the routing itself, his head W 1, GLM-mass order as the reference
    for c in COVERS:
        a, b = blend(zl, heads["his"], c, 1.0, h, False), blend(zl, heads["his"], c, 1.0, h, True)
        g = paired_bootstrap(y, b, a, prompts, n=1000)
        print(f"\nFable routing vs GLM-mass order (his head W 1, cover {c}): {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}]")


if __name__ == "__main__":
    main()
