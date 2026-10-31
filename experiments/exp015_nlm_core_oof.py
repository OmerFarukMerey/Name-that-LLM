"""exp015: an out-of-fold replica of nlm-v27's core, to score additions to it offline.

nlm-v27 sums weighted log-probabilities of its members. Three of them ship fold adapters, so kaggle/nlm/nlm_oof.py
(kernel ntl-nlm-oof) gives their out-of-fold predictions on train.csv: q14v5 (weight 3.0), q8v5 (2.0) and ll8v5
(1.0). The 4-view n-gram SVC (1.5) is refitted here per fold on train + Dhruv's 120k pool with infer.py's exact
recipe. The members without fold models (Nishant's two LoRAs, DeBERTa, ModernBERT), the prompt-group heads, the
complementary stack, ATLAS and the guards are not in the replica, so it is a yardstick, not the 0.651 ensemble.

The core blend is infer.py's: sum of weight x log-probability over members, divided by the number of members, then
softmax, then (optionally) Sinkhorn to equal class mass. A candidate member (out-of-fold probabilities on any
prompt-grouped folds) enters the sum like any infer.py member, at the weights in WEIGHTS. Gains are paired prompt
bootstraps (ntl.oofbank.paired_bootstrap), with the gain on each fold alongside.

nlm-v27's two encoders (Nishant's DeBERTa-v3-large, Dhruv's ModernBERT-large) have no fold models either, so kernel
ntl-member-enc-rep trains stand-ins with the candidates' recipe (REPLICAS). When their out-of-fold logits are in
ENC_DIR, they join the core at nlm-v27's weights and the candidates are scored against that fuller base, which is the
fairer test for another encoder of the same kind.

    PYTHONPATH=src python experiments/exp015_nlm_core_oof.py svc      # fit the SVC folds (CPU, about an hour)
    PYTHONPATH=src python experiments/exp015_nlm_core_oof.py          # replica + candidates
"""

import json
import re
import sys
import time

import numpy as np
import pandas as pd

from ntl.data import LABELS, load_train
from ntl.metrics import balanced_accuracy
from ntl.oofbank import paired_bootstrap
from ntl.paths import REPO_ROOT

OUT = REPO_ROOT / "outputs" / "exp015_nlm_core_oof"
POOL = REPO_ROOT / "data" / "external" / "nlm" / "nlm_pool_v5.parquet"
NLM_OOF = OUT / "nlm_oof.npz"                  # kernel output of ntl-nlm-oof
ENC_DIR = OUT / "enc"                          # kernel output of ntl-member-enc-pool (<name>/oof.npz)
L24_DIR = OUT / "l24"                          # LoRA24 candidates (ntl-nlm-oof-q25, ntl-l24-*-s2): <name>/oof.npz
MISTRAL_OOF = REPO_ROOT / "outputs" / "exp009_member_pilot"   # optional, exp009's pilot OOF
# infer.py's label order (alphabetical); ntl.data.LABELS is the Evaluation-page order
NLM_LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
              'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3',
              'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3',
              'qwen3.8-max']
CORE = {"q14v5": 3.0, "q8v5": 2.0, "ll8v5": 1.0, "svc4": 1.5}
# our replicas of nlm-v27's two encoder members (kernel ntl-member-enc-rep), entered at nlm-v27's weights
REPLICAS = {"deberta_v3_large_pool": ("deb", 1.0), "modernbert_large_pool": ("mb", 1.5)}
WEIGHTS = [0.5, 1.0, 1.5, 2.0, 3.0]
# LP entries infer.py counts in ntl-nlm27-mb's hidden run: svc4, five L24 members with a twin each, two Gaussian heads,
# deb and mb (the zero-weight TF-IDF member is not counted)
HIDDEN_TERMS = 15
if "--plain" in sys.argv:   # the 0.651 recipe itself (no ModernBERT, which lost on the LB: exp013 0.645): 14 entries
    sys.argv.remove("--plain")
    SKIP = {REPLICAS.pop("modernbert_large_pool")[0], "modernbert_large_pool"}
    HIDDEN_TERMS = 14
else:
    SKIP = set()
TERMS = {}   # LP entries a candidate adds on the hidden test (filled in main: 2 for a LoRA24 member, else 1)


def lsm(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def svc_folds(folds):
    """infer.py's 4-view n-gram LinearSVC, refitted per fold on (train minus the fold) + the pool."""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    fw = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
    shape = lambda t: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
    func = lambda t: " ".join(w.lower() if w.lower() in fw or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))
    train = load_train()
    y = train.model.map({l: i for i, l in enumerate(NLM_LABELS)}).to_numpy()
    pool = pd.read_parquet(POOL, columns=["text", "model"])
    pool = pool[pool.model.isin(NLM_LABELS)]
    ptext, py = list(pool.text.astype(str)), pool.model.map({l: i for i, l in enumerate(NLM_LABELS)}).to_numpy()
    oof = np.zeros((len(train), len(NLM_LABELS)), np.float32)
    for k in np.unique(folds):
        t0 = time.time()
        fit, valid = folds != k, folds == k
        fit_text = list(train.text[fit]) + ptext
        fit_y = np.r_[y[fit], py]
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
            B.append(v.transform([prep(t) for t in train.text[valid]]) * wv)
        d = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced").fit(
            hstack(A, format="csr"), fit_y).decision_function(hstack(B, format="csr"))
        oof[valid] = lsm(d)
        print(f"svc4 fold {k}: {valid.sum()} rows, accuracy {(d.argmax(1) == y[valid]).mean():.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return oof


def sinkhorn(P, iters=100):
    Q = P.copy()
    for _ in range(iters):
        Q *= (len(Q) / Q.shape[1]) / Q.sum(0, keepdims=True)
        Q /= Q.sum(1, keepdims=True)
    return Q


def blend(members, weights):
    """infer.py: sum of w x log-probability, divided by the number of non-zero LP entries, softmax.

    On the hidden test that count is HIDDEN_TERMS whatever the replica holds (twin and Gaussian entries are zero for a
    singleton passage but still counted), plus one for each candidate encoder and two for each candidate LoRA24 member
    (its plain head and its centred twin)."""
    known = set(CORE) | {alias for alias, _ in REPLICAS.values()}
    z = sum(weights[k] * members[k] for k in weights) / (HIDDEN_TERMS + sum(TERMS.get(k, 1) for k in weights if k not in known))
    P = np.exp(z - z.max(1, keepdims=True))
    return P / P.sum(1, keepdims=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nlm = np.load(NLM_OOF) if NLM_OOF.exists() else None
    folds = nlm["folds"] if nlm is not None else np.load(OUT / "gkf_folds.npy")
    if sys.argv[1:] == ["svc"]:
        np.save(OUT / "svc4_oof.npy", svc_folds(folds))
        np.save(OUT / "svc4_folds.npy", folds)
        return
    train = load_train()
    to_nlm = [list(LABELS).index(l) for l in NLM_LABELS]   # ntl.data.LABELS order -> infer.py order
    y = train.model.map({l: i for i, l in enumerate(NLM_LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    # the kernel saved ids as an object array, which np.load will not unpickle; train.csv order is checked through the
    # labels and the prompt-grouped folds instead
    assert list(nlm["labels"]) == NLM_LABELS and np.array_equal(nlm["y"], y)
    assert pd.Series(folds).groupby(prompts).nunique().max() == 1
    svc_saved = np.load(OUT / "svc4_folds.npy")
    if not np.array_equal(svc_saved, folds):
        print("note: the SVC folds differ from the adapters' inferred folds (both are prompt-grouped)")
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    members["svc4"] = np.load(OUT / "svc4_oof.npy").astype(np.float64)
    labels = np.array(NLM_LABELS)
    score = lambda P: balanced_accuracy(train.model, labels[P.argmax(1)])

    def report(tag, P, ref):
        gain = paired_bootstrap(y, P.argmax(1), ref.argmax(1), prompts)
        per_fold = [fast_gain(P, ref, folds == k) for k in np.unique(folds)]
        print(f"  {tag}: {score(P):.4f} (gain {gain[0]:+.4f} [{gain[1]:+.4f}, {gain[2]:+.4f}]; per fold "
              + " ".join(f"{g:+.3f}" for g in per_fold) + ")")
        return gain

    def fast_gain(P, ref, rows):
        return (balanced_accuracy(train.model[rows], labels[P[rows].argmax(1)])
                - balanced_accuracy(train.model[rows], labels[ref[rows].argmax(1)]))

    encs = {}
    for d in sorted(ENC_DIR.glob("*/oof.npz")) + sorted(L24_DIR.glob("*/oof.npz")):
        if d.parent.name in SKIP:
            continue
        if d.parent.parent == L24_DIR:
            TERMS[d.parent.name] = 2
        z = np.load(d)
        assert np.array_equal(z["ids"], train.id.to_numpy(dtype=str))
        order = [list(z["labels"]).index(l) for l in NLM_LABELS]
        encs[d.parent.name] = lsm(z["logits"][:, order].astype(np.float64))
    for k, v in {**members, **encs}.items():
        print(f"{k:22s} alone: {score(v):.4f}")
    base_s = sinkhorn(blend(members, CORE))
    print(f"core {CORE}: {score(blend(members, CORE)):.4f}, with Sinkhorn {score(base_s):.4f}")
    # stand-ins for nlm-v27's own encoders, at its weights: the core they extend is the fairer base for a candidate
    reps = {alias: encs.pop(name) for name, (alias, _) in REPLICAS.items() if name in encs}
    rep_w = {alias: w for name, (alias, w) in REPLICAS.items() if alias in reps}
    for alias in reps:
        report(f"core + {alias} replica at w {rep_w[alias]}",
               sinkhorn(blend({**members, alias: reps[alias]}, {**CORE, alias: rep_w[alias]})), base_s)
    if len(reps) == 2:
        other = {a: reps[a] for a in reps if a != "mb"}
        ref = sinkhorn(blend({**members, **other}, {**CORE, **{a: rep_w[a] for a in other}}))
        report("mb replica on core + deb replica (Dhruv: +0.6 points in his OOF)",
               sinkhorn(blend({**members, **reps}, {**CORE, **rep_w})), ref)
    full, full_w = {**members, **reps}, {**CORE, **rep_w}
    full_s = sinkhorn(blend(full, full_w))
    if reps:
        print(f"core + replicas {sorted(reps)}: {score(full_s):.4f} (the base for the candidates below)")
    results = {}
    for name, lp in encs.items():
        print(f"\n{name}:")
        for w in WEIGHTS:
            results[f"{name}@{w}"] = report(f"base + {name} at w {w}",
                                            sinkhorn(blend({**full, name: lp}, {**full_w, name: w})), full_s)
    if len(encs) > 1:
        for w in WEIGHTS:
            report(f"base + all {len(encs)} candidates at w {w} each",
                   sinkhorn(blend({**full, **encs}, {**full_w, **{n: w for n in encs}})), full_s)
    json.dump({"core": CORE, "replicas": rep_w, "candidates": sorted(encs), "gains": results},
              open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
