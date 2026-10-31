"""exp030: zbigniewkirchner's public 0.664 swaps the 4-view SVC's pool (Dhruv's nlm_pool_v5, 120k passages) for
crtproduction's open pool (198k, minus passages regenerated from train prompts) and raises --svc_w from 1.5 to 12. This
fits infer.py's SVC recipe out of fold on exp015's prompt-grouped folds, with its source weights (official rows 1.0, pool
0.35) and its exact-text de-duplication (pool rows equal to any official text or to an earlier row are dropped), once per
pool, and scores the SVC alone and inside exp015's --plain replica (core + DeBERTa stand-in) at svc weights 1.5 / 6 / 12 /
20, with V31's soft Sinkhorn at 0.45 and 0.30. infer.py's matched-clean external rows (weight 0.5) are not in the replica.

    python -I experiments/exp030_open_pool_svc.py svc open <ntl_open_pool.parquet>   # fit the folds (CPU, ~2 h)
    python -I experiments/exp030_open_pool_svc.py svc v5                             # the same on nlm_pool_v5
    python -I experiments/exp030_open_pool_svc.py                                    # compare

-I because the open pool is a downloaded file; the repo's src/ and experiments/ are put on the path below.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
sys.argv.append("--plain")
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp015_nlm_core_oof as E  # noqa: E402  (reads --plain at import)
from exp029_atlas_ext import soft_sinkhorn  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402
from ntl.paths import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "outputs" / "exp030_open_pool_svc"
SVC_WEIGHTS = (1.5, 6.0, 12.0, 20.0)


def key(t):
    """infer.py's _norm_text_key."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(t or ""))).strip()


def load_pool(which, path=None):
    if which == "v5":
        p = pd.read_parquet(E.POOL, columns=["text", "model"])
    else:
        p = pd.read_parquet(path, columns=["text", "model", "flag_train_prompt_regen"])
        p = p[~p.flag_train_prompt_regen.astype(bool)][["text", "model"]]
    return p[p.model.isin(E.NLM_LABELS)].reset_index(drop=True)


def svc_folds(folds, pool):
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    fw = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
    shape = lambda t: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
    func = lambda t: " ".join(w.lower() if w.lower() in fw or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))
    train = load_train()
    ttext = train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    seen, keep = {key(t) for t in ttext}, []
    for t in pool.text.fillna("").astype(str):
        k = key(t); ok = bool(k) and k not in seen; keep.append(ok)
        if ok:
            seen.add(k)
    pool = pool[np.array(keep)]
    ptext, py = pool.text.astype(str).tolist(), pool.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    print(f"pool rows after de-duplication: {len(pool)}", flush=True)
    oof = np.zeros((len(train), len(E.NLM_LABELS)), np.float32)
    for k in np.unique(folds):
        t0 = time.time()
        fit, valid = folds != k, folds == k
        fit_text = [t for t, f in zip(ttext, fit) if f] + ptext
        fit_y = np.r_[y[fit], py]
        sw = np.r_[np.ones(fit.sum(), np.float32), np.full(len(ptext), 0.35, np.float32)]
        A, B = [], []
        for view, lo, hi, mf, wv in [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7),
                                     ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)]:
            prep = shape if view == "shape" else func if view == "function" else (lambda s: s)
            an = "char" if view == "shape" else "word" if view == "function" else view
            o = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf,
                     dtype=np.float32)
            if an == "word":
                o["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
            v = TfidfVectorizer(**o)
            A.append(v.fit_transform([prep(t) for t in fit_text]) * wv)
            B.append(v.transform([prep(t) for t, f in zip(ttext, valid) if f]) * wv)
        d = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced").fit(
            hstack(A, format="csr"), fit_y, sample_weight=sw).decision_function(hstack(B, format="csr"))
        del A, B
        oof[valid] = E.lsm(d)
        print(f"fold {k}: {valid.sum()} rows, accuracy {(d.argmax(1) == y[valid]).mean():.3f} ({time.time() - t0:.0f}s)",
              flush=True)
    return oof


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nlm = np.load(E.NLM_OOF, allow_pickle=False)
    folds = nlm["folds"]
    args = [a for a in sys.argv[1:] if a != "--plain"]
    if args[:1] == ["svc"]:
        which = args[1]
        np.save(OUT / f"svc4_{which}_oof.npy", svc_folds(folds, load_pool(which, args[2] if len(args) > 2 else None)))
        return
    train = load_train()
    labels = np.array(E.NLM_LABELS)
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    score = lambda pred: balanced_accuracy(train.model, labels[pred])
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    d = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz", allow_pickle=False)
    members["deb"] = E.lsm(d["logits"][:, [list(d["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    svcs = {"v5_exp015": np.load(E.OUT / "svc4_oof.npy").astype(np.float64)}
    for which in ("v5", "open"):
        f = OUT / f"svc4_{which}_oof.npy"
        if f.exists():
            svcs[which] = np.load(f).astype(np.float64)
    R = {}
    for name, s in svcs.items():
        R[f"alone_{name}"] = score(s.argmax(1))
        print(f"SVC alone, {name}: {R[f'alone_{name}']:.4f}")
    ref = {}
    for alpha in (0.45, 0.30):
        base = soft_sinkhorn(E.blend({**members, "svc4": svcs["v5_exp015"]}, {**E.CORE, "deb": 1.0}), alpha=alpha).argmax(1)
        ref[alpha] = base
        R[f"replica_v5_exp015_w1.5_sk{alpha}"] = score(base)
        print(f"\nSinkhorn {alpha}: replica with exp015's SVC at 1.5: {score(base):.4f}")
        for name, s in svcs.items():
            line = []
            for w in SVC_WEIGHTS:
                pred = soft_sinkhorn(E.blend({**members, "svc4": s}, {**E.CORE, "svc4": w, "deb": 1.0}), alpha=alpha).argmax(1)
                g = paired_bootstrap(y, pred, base, prompts)
                R[f"replica_{name}_w{w}_sk{alpha}"] = {"score": score(pred), "gain": g}
                line.append(f"w {w:g}: {score(pred):.4f} ({g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}])")
            print(f"  {name}: " + "; ".join(line))
    json.dump(R, open(OUT / "metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
