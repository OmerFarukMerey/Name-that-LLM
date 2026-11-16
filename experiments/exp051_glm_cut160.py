"""exp051: the GLM stage's likelihood head trained on train scores cut the way the test-time stage cuts them.

estopadilla's head (and our exp050 refits) were fitted on per-token scores of the full train texts (his train run; ours:
ntl-glmll-{flash,glm53}, cut 512), keeping the first 160 tokens. The stage scores only the first 160 tokens of each test
text, in decodes of 8 texts, so most of its decodes are 8 x 173 rows while the train decodes are as long as their texts.
GLM-5.3 and Flash are mixture-of-experts and their scores move with the decode shape: on 120 train texts run through
the stage (kaggle/glmll/build_stagecheck.py) GLM-5.3 differs from our 512-cut run by a median 0.10 nats per token.
ntl-glmll160-{flash,glm53} (kaggle/glmll/build_glmll.py 160) rescore every train text at the stage's cut.

    python -I experiments/exp051_glm_cut160.py drift <stage-check run dir with {flash,glm53}_scores.npz>
    python -I experiments/exp051_glm_cut160.py oof
    python -I experiments/exp051_glm_cut160.py export name=C,W[,flag ...]   # flags: gbm (trees on 54 + pair), both

oof: heads trained on the 512-cut features (what every public notebook runs), on the 160-cut features, or on both,
always scored on the 160-cut features (the test's), blended into exp050's stand-in stack as v1_stage.py blends them.
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "experiments")]
import exp050_glm_ll_head as X  # noqa: E402  (sets the paths, threads and the stand-in stack's inputs)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

D512 = X.OUT   # ntl-glmll-{flash,glm53} outputs
D160 = Path(os.environ.get("EXP051_D160", _ROOT / "outputs" / "exp051_glm_cut160"))


def load(d, tag, ids):
    """Per-token scores of the train texts from a run_pipeline output dir (its npz ids are an object array, never
    read; <tag>_features.csv lists the same texts in the same order)."""
    z = np.load(d / f"{tag}_scores.npz", allow_pickle=False)
    order = pd.read_csv(d / f"{tag}_features.csv", dtype={"id": str}, usecols=["id"]).id.tolist()
    assert len(order) == len(z["offsets"]) - 1
    pos = {k: i for i, k in enumerate(order)}
    off, sc = z["offsets"], z["scores"]
    return [sc[off[pos[i]]:off[pos[i] + 1]] for i in ids]


def frame(sf, sg, texts, names):
    """His 54 columns of the first 160 tokens, then the 12 per-token flash - glm53 contrasts (exp050's pair block)."""
    A, cols = X.design(X.features(sf, texts, 160), X.features(sg, texts, 160), names, X.HIS["diff_features"])
    pr = X.pair_features(sf, sg, 160)
    P = np.array([[r[k] for k in pr[0]] for r in pr])
    return np.hstack([A, P]), cols + [f"pair:{k}" for k in pr[0]]


def drift(run, ids, texts, names):
    rid = pd.read_csv(run / "flash_features.csv", dtype={"id": str}, usecols=["id"]).id.tolist()
    pos = {k: i for i, k in enumerate(ids)}
    sd = dict(zip(X.HIS["features"], X.HIS["scale"]))
    for tag in ("flash", "glm53"):
        st = load(run, tag, rid)
        for label, d in (("512-cut train run", D512), ("160-cut train run", D160)):
            tr = load(d, tag, rid)
            g = np.concatenate([np.abs(a[:, 0] - b[:len(a), 0]) for a, b in zip(st, tr)])
            fa = pd.DataFrame(X.features(st, [texts[pos[i]] for i in rid], 160))[names]
            fb = pd.DataFrame(X.features(tr, [texts[pos[i]] for i in rid], 160))[names]
            fg = ((fa - fb) / pd.Series({n: sd[f"{tag}:{n}"] for n in names})).abs()
            print(f"{tag:5s} stage run vs {label}: per-token |lp gap| median {np.median(g):.4f} p99 "
                  f"{np.quantile(g, 0.99):.3f} max {g.max():.2f}, identical tokens {np.mean(g == 0):.3f}; "
                  f"feature |gap|/sd median {fg.median().median():.4f}, max {fg.max().max():.2f}", flush=True)


def fit_predict(kind, Xtr, ytr, Xva, n54):
    """kind lr: his recipe (scaler + LR C 0.05 on his 54 columns); gbm: that averaged 50/50 with exp050's trees on
    every column (54 + pair)."""
    sc, lr = X.fit(Xtr[:, :n54], ytr, X.HIS["C"])
    ll = lr.predict_log_proba(sc.transform(Xva[:, :n54]))
    if kind == "gbm":
        gs = StandardScaler().fit(Xtr)
        clf = HistGradientBoostingClassifier(**X.GBM).fit(gs.transform(Xtr), ytr)
        ll = X.mix(ll, np.log(np.clip(clf.predict_proba(gs.transform(Xva)), 1e-12, None)))
    return ll


def oof(kind, train_sets, Xeval, y, folds, n54):
    ll = np.zeros((len(y), 18))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        Xtr = np.vstack([T[tr] for T in train_sets])
        ll[va] = fit_predict(kind, Xtr, np.tile(y[tr], len(train_sets)), Xeval[va], n54)
    return ll


def main():
    mode = (sys.argv[1:2] or ["oof"])[0]
    train = X.load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    names = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    if mode == "drift":
        drift(Path(sys.argv[2]), ids, texts, names)
        return

    F512, cols = frame(load(D512, "flash", ids), load(D512, "glm53", ids), texts, names)
    F160, _ = frame(load(D160, "flash", ids), load(D160, "glm53", ids), texts, names)
    n54 = len(X.HIS["features"])
    assert cols[:n54] == X.HIS["features"]
    dz = np.abs(F160[:, :n54] - F512[:, :n54]) / np.asarray(X.HIS["scale"])
    print(f"160-cut vs 512-cut train features, |gap|/sd: median {np.median(dz):.4f}, p99 {np.quantile(dz, 0.99):.3f}; "
          f"worst columns {[cols[i] for i in np.argsort(-np.median(dz, 0))[:5]]}", flush=True)

    folds = X.prompt_folds(prompts)
    nlm = np.load(X.E.NLM_OOF)
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    members["svc4"] = np.load(X.E.OUT / "svc4_oof.npy").astype(np.float64)
    zl = np.log(np.clip(X.E.sinkhorn(X.E.blend(members, X.E.CORE)), 1e-12, None))
    base = zl.argmax(1)
    h = X.HIS["entropy_ref"]
    score = lambda p: X.balanced_accuracy(train.model, np.asarray(X.LABELS)[p])
    print(f"stand-in stack {score(base):.4f}", flush=True)

    if mode == "oof":
        his = lambda F: X.log_softmax(((F[:, :n54] - np.asarray(X.HIS["mean"])) / np.asarray(X.HIS["scale"]))
                                      @ np.asarray(X.HIS["coef"]).T + np.asarray(X.HIS["intercept"]), 1)
        for c in X.COVERS:
            a, b = X.blend(zl, his(F512), c, 1.0, h), X.blend(zl, his(F160), c, 1.0, h)
            print(f"his weights (in-sample) c{c:.2f}: on 512-cut feats {score(a) - score(base):+.4f}, on 160-cut feats "
                  f"{score(b) - score(base):+.4f} vs the stack", flush=True)
        ref = oof("lr", [F512], F160, y, folds, n54)   # the public setting: trained at 512, the test at 160
        p_ref = {c: X.blend(zl, ref, c, 1.0, h) for c in X.COVERS}
        print("reference = his recipe trained on 512-cut, scored on 160-cut feats: "
              + "  ".join(f"c{c:.2f} {score(p_ref[c]) - score(base):+.4f}" for c in X.COVERS) + " vs the stack")
        runs = [("lr", "512", [F512]), ("lr", "160", [F160]), ("lr", "both", [F512, F160]),
                ("gbm", "512", [F512]), ("gbm", "160", [F160]), ("gbm", "both", [F512, F160])]
        for kind, src, sets in runs:
            ll = oof(kind, sets, F160, y, folds, n54)
            for W in (1.0, 1.25):
                g = {c: X.paired_bootstrap(y, X.blend(zl, ll, c, W, h), p_ref[c], prompts, n=1000) for c in X.COVERS}
                print(f"{kind:3s} trained on {src:4s} W {W:<4g} alone {score(ll.argmax(1)):.4f} | vs reference: "
                      + "  ".join(f"c{c:.2f} {g[c][0]:+.4f} [{g[c][1]:+.4f},{g[c][2]:+.4f}]" for c in X.COVERS),
                      flush=True)
    if mode == "export":
        for spec in sys.argv[2:]:
            name, v = spec.split("=")
            C, W, *flags = v.split(",")
            sets = [F512, F160] if "both" in flags else [F160]
            Xf, yf = np.vstack(sets), np.tile(y, len(sets))
            src = "512- and 160-cut" if "both" in flags else "160-cut"
            X.export(name, Xf[:, :n54], cols[:n54], yf, float(C), float(W),
                     note=f" (exp051: trained on {src} train scores, C {C}, W {W}"
                          + (", trees on 54 + pair" if "gbm" in flags else "") + ")",
                     extra={"pair": True} if "gbm" in flags else None,
                     gbm=(Xf, cols) if "gbm" in flags else None)


if __name__ == "__main__":
    main()
