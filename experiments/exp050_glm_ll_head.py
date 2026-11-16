"""exp050: refit the likelihood head of the GLM stage that every public 0.689-0.694 notebook runs (estopadilla's
ll_model.json: StandardScaler + multinomial logistic regression, C 0.05, on 54 likelihood features of the first 160
tokens under GLM-5.3-Flash and GLM-5.3, all 2,699 train rows).

Inputs: outputs/exp050_glm_ll_head/{flash,glm53}_scores.npz, the outputs of our kernels ntl-glmll-flash and
ntl-glmll-glm53 (kaggle/glmll/build_glmll.py: his run_pipeline.py on every train passage, cut at 512 tokens).

1. check: his features rebuilt at 160 tokens + his recipe on all rows must give his scaler and coefficients
2. out of fold (ntl.cv prompt folds): the head alone, and blended the way v1_stage.py blends it, into a stand-in stack
   (exp015's core OOF with Sinkhorn): the top `cover` texts by P(glm-5.3) + P(glm-5.3-flash), argmax(T' zl + W ll)
   with T' matching the stack's mean entropy to the head's entropy_ref
3. candidates: C, W, cut, extra features

    python -I experiments/exp050_glm_ll_head.py [check|grid|ref|gbm|export name=C,W[,flag] ...]
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "kaggle" / "glmll")]
sys.argv.append("--plain")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import log_softmax  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import exp015_nlm_core_oof as E  # noqa: E402
from extra_features import gbm_export, gbm_logp, one, pair_features  # noqa: E402
from ntl.cv import prompt_folds  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402

OUT = Path(os.environ.get("EXP050_OUT", _ROOT / "outputs" / "exp050_glm_ll_head"))
SRC = _ROOT / "kaggle" / "glmll" / "v1src"
HIS = json.loads((SRC / "ll_model.json").read_text())
LABELS = HIS["classes"]   # alphabetical, = exp015's NLM_LABELS
assert LABELS == E.NLM_LABELS
GLM = [LABELS.index("glm-5.3"), LABELS.index("glm-5.3-flash")]
COVERS = (0.15, 0.20, 0.25)
RDIFF = ["lp_mean", "lp_q50", "ent_mean", "fdg", "z_mean", "logrank_mean", "top1", "bpc"]


def load_scores(tag, ids):
    z = np.load(OUT / f"{tag}_scores.npz", allow_pickle=False)
    # the npz's ids are an object array (never unpickled); <tag>_features.csv lists the same texts in the same order
    order = pd.read_csv(OUT / f"{tag}_features.csv", dtype={"id": str}, usecols=["id"]).id.tolist()
    assert len(order) == len(z["offsets"]) - 1
    pos = {k: i for i, k in enumerate(order)}
    assert set(ids) <= set(pos), f"{tag}: {len(set(ids) - set(pos))} train ids missing"
    off, sc, tok = z["offsets"], z["scores"], z["tokens"]
    scores = [sc[off[pos[i]]:off[pos[i] + 1]] for i in ids]
    # the tokens are stored text-only; they line up with the scores when there is one score per text token
    toks = [tok[off[pos[i]]:off[pos[i] + 1]] for i in ids] if len(tok) == len(sc) else None
    return scores, toks


def load_ref(tag, ids):
    """ntl-refll's per-token scores under a Qwen3 base model (ids saved as a plain string array)."""
    z = np.load(OUT / f"{tag}_scores.npz", allow_pickle=False)
    pos = {k: i for i, k in enumerate(z["ids"].tolist())}
    off, sc = z["offsets"], z["scores"]
    assert np.isfinite(sc).all()
    return [sc[off[pos[i]]:off[pos[i] + 1]] for i in ids]


def features(scores, texts, cut, extra=False):
    """His per-text features of the first `cut` tokens (n_chars stays the full text's length, as at test time)."""
    rows = [one(s[:cut], len(t)) for s, t in zip(scores, texts)]
    if extra:
        for r, s in zip(rows, scores):
            lp = s[:cut, 0].astype(np.float64)
            r.update({f"lp_bin{i}": float(((lp >= a) & (lp < b)).mean())
                      for i, (a, b) in enumerate([(-99, -8), (-8, -4), (-4, -2), (-2, -1), (-1, -0.25), (-0.25, 1)])})
    return rows


def design(flash, glm53, names, diff):
    """His layout: flash:<f>, glm53:<f>, then d:<f> = flash - glm53 for the diff features."""
    cols, X = [], []
    for tag, rows in (("flash", flash), ("glm53", glm53)):
        for n in names:
            cols.append(f"{tag}:{n}")
            X.append([r[n] for r in rows])
    for n in diff:
        cols.append(f"d:{n}")
        X.append([a[n] - b[n] for a, b in zip(flash, glm53)])
    return np.asarray(X, dtype=np.float64).T, cols


def fit(X, y, C, w=None):
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y, sample_weight=w)
    return sc, lr


def oof_head(X, y, folds, C, w=None):
    ll = np.zeros((len(y), 18))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        sc, lr = fit(X[tr], y[tr], C, None if w is None else w[tr])
        ll[va] = lr.predict_log_proba(sc.transform(X[va]))
    return ll


def export(name, X, cols, y, C, W, w=None, note="", extra=None, gbm=None):
    """A head fitted on every train row, in ll_model.json's format (v1_stage.py / the 0.694's inline stage read it).
    extra: the evidence beyond his 54 features the stage must compute (build_lv694.py), with his head as fallback."""
    sc, lr = fit(X, y, C, w)
    assert list(lr.classes_) == list(range(18))
    model = dict(HIS, features=cols, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_.tolist(),
                 intercept=lr.intercept_.tolist(), C=C, W=W, train_rows=len(y),
                 source=f"Name-that-LLM experiments/exp050_glm_ll_head.py: {name}{note}")
    if extra:
        model.update(extra=extra, fallback=HIS)
    if gbm is not None:   # boosted trees on gbm = (X, cols), averaged with the logistic head (build_lv694 stage patch)
        Xg, cg = gbm
        gs = StandardScaler().fit(Xg)
        clf = HistGradientBoostingClassifier(**GBM).fit(gs.transform(Xg), y)
        g = gbm_export(clf)
        assert np.abs(gbm_logp(g, gs.transform(Xg)) - np.log(clf.predict_proba(gs.transform(Xg)))).max() < 1e-5
        model["gbm"] = dict(g, features=cg, mean=gs.mean_.tolist(), scale=gs.scale_.tolist(), mix=GBM_MIX, params=GBM)
    (OUT / "heads").mkdir(parents=True, exist_ok=True)
    (OUT / "heads" / f"{name}.json").write_text(json.dumps(model), encoding="utf-8")
    print(f"exported heads/{name}.json", flush=True)


GBM = dict(max_depth=2, learning_rate=0.1, max_iter=150, l2_regularization=1.0, random_state=0)
GBM_MIX = 0.5   # the trees' share of the probability mix with the logistic head


def oof_gbm(X, y, folds):
    ll = np.zeros((len(y), 18))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        sc = StandardScaler().fit(X[tr])
        clf = HistGradientBoostingClassifier(**GBM).fit(sc.transform(X[tr]), y[tr])
        ll[va] = np.log(np.clip(clf.predict_proba(sc.transform(X[va])), 1e-12, None))
    return ll


def mix(ll_lr, ll_gbm, share=GBM_MIX):
    return np.log((1 - share) * np.exp(ll_lr) + share * np.exp(ll_gbm))


def entropy(z):
    return float(-(np.exp(z) * z).sum(1).mean())


def match_temperature(z, h_ref):
    lo, hi = 1e-3, 1e3
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if entropy(log_softmax(z * mid, 1)) > h_ref else (lo, mid)
    return float(np.sqrt(lo * hi))


def blend(zl, ll, cover, W, h_ref):
    """v1_stage.py on one test: the top `cover` rows by GLM mass get argmax(zl + (W / T') ll)."""
    mass = np.exp(zl[:, GLM]).sum(1)
    n = int(cover * len(zl))
    sel = np.argsort(-mass, kind="stable")[:n]
    w = W / match_temperature(zl, h_ref)
    pred = zl.argmax(1)
    pred[sel] = (zl[sel] + w * ll[sel]).argmax(1)
    return pred


def main():
    mode = (sys.argv[1:2] or ["grid"])[0]
    train = load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    sf, tf = load_scores("flash", ids)
    sg, tg = load_scores("glm53", ids)
    same = np.mean([np.array_equal(a, b) for a, b in zip(tf, tg)]) if tf and tg else float("nan")
    print(f"scored tokens per text: median {np.median([len(s) for s in sf]):.0f}, max {max(len(s) for s in sf)}; "
          f"identical flash/glm53 tokens {same:.3f}")
    names = [c.split(":", 1)[1] for c in HIS["features"] if c.startswith("flash:")]
    diff = HIS["diff_features"]

    X160, cols = design(features(sf, texts, 160), features(sg, texts, 160), names, diff)
    assert cols == HIS["features"], (cols[:5], HIS["features"][:5])
    sc, lr = fit(X160, y, HIS["C"])
    dm = np.abs(sc.mean_ - HIS["mean"]) / np.asarray(HIS["scale"])
    print(f"check vs ll_model.json: scaler mean max |diff|/scale {dm.max():.4f} (median {np.median(dm):.4f}), "
          f"coef corr {np.corrcoef(lr.coef_.ravel(), np.ravel(HIS['coef']))[0, 1]:.4f}, "
          f"max |coef diff| {np.abs(lr.coef_ - np.asarray(HIS['coef'])).max():.4f}")
    worst = np.argsort(-dm)[:5]
    print("  largest scaler gaps:", [(cols[i], round(float(dm[i]), 3)) for i in worst])
    if mode == "check":
        return

    folds = prompt_folds(prompts)
    nlm = np.load(E.NLM_OOF)
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    members["svc4"] = np.load(E.OUT / "svc4_oof.npy").astype(np.float64)
    zl = np.log(np.clip(E.sinkhorn(E.blend(members, E.CORE)), 1e-12, None))
    base = zl.argmax(1)
    score = lambda p: balanced_accuracy(train.model, np.asarray(LABELS)[p])
    print(f"stand-in stack: {score(base):.4f}, mean entropy {entropy(zl):.3f} (his entropy_ref {HIS['entropy_ref']:.3f})")

    h = HIS["entropy_ref"]
    mass = np.exp(zl[:, GLM]).sum(1)
    top = mass >= np.quantile(mass, 0.70)   # the train rows the stage would score (focus weighting)
    # his recipe refit out of fold is the reference every candidate is compared with (his own weights saw every row)
    ref = oof_head(X160, y, folds, HIS["C"])
    p_ref = {c: blend(zl, ref, c, 1.0, h) for c in COVERS}
    print("his recipe (L160, C 0.05, W 1), out of fold, vs the stack: "
          + "  ".join(f"c{c:.2f} {paired_bootstrap(y, p_ref[c], base, prompts, n=1000)[0]:+.4f}" for c in COVERS))
    his_ll = log_softmax(((X160 - np.asarray(HIS["mean"])) / np.asarray(HIS["scale"])) @ np.asarray(HIS["coef"]).T
                         + np.asarray(HIS["intercept"]), 1)
    print("his weights in-sample (optimistic) vs the stack: "
          + "  ".join(f"c{c:.2f} {paired_bootstrap(y, blend(zl, his_ll, c, 1.0, h), base, prompts, n=1000)[0]:+.4f}"
                      for c in COVERS))
    table = []

    def report(tag, ll, Ws=(1.0,)):
        alone = score(ll.argmax(1))
        for W in Ws:
            g = {c: paired_bootstrap(y, blend(zl, ll, c, W, h), p_ref[c], prompts, n=1000) for c in COVERS}
            table.append((tag, W, g))
            print(f"{tag:26s} W {W:<4g} alone {alone:.4f} | vs his recipe: "
                  + "  ".join(f"c{c:.2f} {g[c][0]:+.4f} [{g[c][1]:+.4f},{g[c][2]:+.4f}]" for c in COVERS), flush=True)

    REF = ("q4b", "q8b")

    def build(flags):
        """His 54 columns plus the blocks named in flags (pair, q4b / q8b, rd = glm53 - reference differences)."""
        Xs, cs, extra = [X160], list(cols), {}
        if "pair" in flags:
            pr = pair_features(sf, sg, 160)
            Xs.append(np.array([[r[k] for k in pr[0]] for r in pr])); cs += [f"pair:{k}" for k in pr[0]]
            extra["pair"] = True
        for tag in REF:
            if tag in flags:
                rr = features(load_ref(tag, ids), texts, 160)
                Xs.append(np.array([[r[n] for n in names] for r in rr])); cs += [f"{tag}:{n}" for n in names]
                extra["ref"] = {"tag": tag, "size": {"q4b": "4b-base", "q8b": "8b-base"}[tag]}
                if "rd" in flags:
                    g = features(sg, texts, 160)
                    Xs.append(np.array([[a[n] - b[n] for n in RDIFF] for a, b in zip(g, rr)]))
                    cs += [f"r:{n}" for n in RDIFF]
                    extra["ref_diff"] = RDIFF
        return np.hstack(Xs), cs, extra

    if mode == "grid":
        Ws = (0.75, 1.0, 1.25, 1.5, 2.0)
        for C in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
            report(f"L160 C {C}", oof_head(X160, y, folds, C), Ws)
        # focus: the train rows the stage would score (the top 30% by the stand-in's out-of-fold GLM mass) weigh more
        for a in (1.0, 3.0):
            for C in (0.05, 0.2):
                report(f"L160 C {C} focus x{1 + a:g}", oof_head(X160, y, folds, C, 1 + a * top), (1.0, 1.5))
        Xc, _ = design(features(sf, texts, 512), features(sg, texts, 512), names, diff)
        for C in (0.05, 0.2):
            report(f"L512 C {C} (runtime cut change)", oof_head(Xc, y, folds, C), (1.0, 1.5))
        ex_names = names + [f"lp_bin{i}" for i in range(6)]
        Xe, _ = design(features(sf, texts, 160, True), features(sg, texts, 160, True), ex_names, diff)
        report("L160 + lp bins C 0.05 (features.py change)", oof_head(Xe, y, folds, 0.05), (1.0, 1.5))
        pair = pair_features(sf, sg, 160)
        P = np.array([[r[k] for k in pair[0]] for r in pair])
        Xp = np.hstack([X160, P])
        for C in (0.05, 0.2):
            report(f"L160 + per-token pair C {C} (stage change)", oof_head(Xp, y, folds, C), (1.0, 1.5))
        mean_gain = lambda g: np.mean([g[c][0] for c in COVERS])
        print("\nbest by mean gain over covers:")
        for tag, W, g in sorted(table, key=lambda r: -mean_gain(r[2]))[:8]:
            print(f"  {tag:40s} W {W:<4g} {mean_gain(g):+.4f}")
    if mode == "ref":
        for flags in (("q4b",), ("q4b", "rd"), ("pair", "q4b"), ("pair", "q4b", "rd")):
            Xf, _, _ = build(flags)
            for C in (0.05, 0.1):
                report(f"L160 + {'+'.join(flags)} C {C}", oof_head(Xf, y, folds, C), (1.0, 1.25, 1.5))
        mean_gain = lambda g: np.mean([g[c][0] for c in COVERS])
        print("\nbest by mean gain over covers:")
        for tag, W, g in sorted(table, key=lambda r: -mean_gain(r[2]))[:8]:
            print(f"  {tag:40s} W {W:<4g} {mean_gain(g):+.4f}")
    if mode == "gbm":
        for flags in ((), ("pair",), ("pair", "q4b")):
            Xf, _, _ = build(flags)
            lg = oof_gbm(Xf, y, folds)
            report(f"gbm on 54{''.join('+' + f for f in flags)}", lg, (1.0,))
            report(f"his recipe/2 + gbm 54{''.join('+' + f for f in flags)}/2", mix(ref, lg), (1.0, 1.25))
    if mode == "export":
        for spec in sys.argv[2:]:   # name=C,W[,flag ...]: flags pair, q4b, q8b, rd, fx<a> (focus weight 1 + a)
            name, v = spec.split("=")
            C, W, *flags = v.split(",")
            C, W = float(C), float(W)
            fx = [float(f[2:]) for f in flags if f.startswith("fx")]
            Xf, cf, extra = build(flags)
            gbm = (Xf, cf) if "gbm" in flags else None
            if gbm is not None:   # the logistic part stays on his 54 columns, the trees read every block
                Xf, cf = X160, cols
            export(name, Xf, cf, y, C, W, (1 + fx[0] * top) if fx else None,
                   f" (L160, C {C}, W {W}" + (f", focus x{1 + fx[0]:g} on the top 30% GLM mass" if fx else "")
                   + (f", extra {'+'.join(f for f in flags if not f.startswith('fx'))}" if extra else "") + ")",
                   extra, gbm)

if __name__ == "__main__":
    main()
