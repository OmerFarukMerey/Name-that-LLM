"""exp002_pseudo_label_sim: does refitting the stack's n-gram SVC on confidently
pseudo-labelled test passages help the final blend?

Simulation of the public 0.639 stack's SVC member (4 TF-IDF views, LinearSVC C=0.5,
train + public pool_v4). Host prompts are split: 2 folds act as train, 3 folds act as
the unlabeled test. The stack is simulated as the SVC's log-probabilities plus
independent evidence for the truth, tuned to the stack's ~0.64 balanced accuracy.
Pseudo labels: the stack's argmax on the most confident half of each predicted class.
Then the SVC is refit with those rows added and swapped into the stack for the rest.
"""

import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from ntl.cv import prompt_folds
from ntl.data import LABELS, load_train
from ntl.metrics import balanced_accuracy
from ntl.paths import REPO_ROOT

SMALL_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("small", 1, 3, 50000, 0.5)]


def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def small_words_text(t):
    return " ".join(w.lower() if w.lower() in SMALL_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", t))


def svc_proba(fit_texts, fit_y, test_texts, sample_weight=None):
    A, B = [], []
    for view, lo, hi, max_features, weight in VIEWS:
        prep = shape_text if view == "shape" else small_words_text if view == "small" else (lambda s: s)
        analyzer = "char" if view == "shape" else "word" if view == "small" else view
        opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                    max_features=max_features, dtype=np.float32)
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vec = TfidfVectorizer(**opts)
        A.append(vec.fit_transform([prep(t) for t in fit_texts]) * weight)
        B.append(vec.transform([prep(t) for t in test_texts]) * weight)
    svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")
    svc.fit(hstack(A, format="csr"), fit_y, sample_weight=sample_weight)
    d = svc.decision_function(hstack(B, format="csr"))
    e = np.exp(d - d.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main(test_folds=(2, 3, 4), keep_frac=0.5, target=0.64, seed=0):
    t0 = time.time()
    train = load_train()
    pool = pd.read_parquet(REPO_ROOT / "data/external/handwriting-blend/pool_v4.parquet")
    lab = {l: i for i, l in enumerate(LABELS)}
    folds = prompt_folds(train["prompt_id"])
    is_te = np.isin(folds, test_folds)
    tr, te = train[~is_te], train[is_te].reset_index(drop=True)
    y_te = te["model"].map(lab).values
    fit_texts = tr.text.tolist() + pool.text.tolist()
    fit_y = np.r_[tr.model.map(lab).values, pool.model.map(lab).values]

    P_svc = svc_proba(fit_texts, fit_y, te.text.tolist())
    labels = np.array(LABELS)
    print(f"[{time.time()-t0:.0f}s] SVC base on held-out host rows: {balanced_accuracy(te.model, labels[P_svc.argmax(1)]):.4f}", flush=True)

    # Simulated stack: SVC evidence + independent evidence for the truth, tuned to `target` balanced accuracy.
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=P_svc.shape)
    onehot = np.eye(len(LABELS))[y_te]
    log_svc = np.log(np.clip(P_svc, 1e-9, 1))
    for beta in np.arange(0.5, 6, 0.1):
        S = log_svc + beta * onehot + 1.5 * noise
        if balanced_accuracy(te.model, labels[S.argmax(1)]) >= target:
            break
    stack_pred = S.argmax(1)
    print(f"simulated stack beta {beta:.1f}: {balanced_accuracy(te.model, labels[stack_pred]):.4f}")

    # Pseudo rows: the most confident `keep_frac` of each predicted class.
    conf = np.exp(S - S.max(1, keepdims=True))
    conf = (conf / conf.sum(1, keepdims=True)).max(1)
    pseudo = np.zeros(len(te), bool)
    for c in range(len(LABELS)):
        idx = np.flatnonzero(stack_pred == c)
        if len(idx):
            pseudo[idx[np.argsort(-conf[idx])[: int(np.ceil(keep_frac * len(idx)))]]] = True
    print(f"pseudo rows {pseudo.sum()} of {len(te)}, label accuracy {np.mean(stack_pred[pseudo] == y_te[pseudo]):.3f}")

    P_pl = svc_proba(fit_texts + te.text[pseudo].tolist(), np.r_[fit_y, stack_pred[pseudo]], te.text.tolist())
    rest = ~pseudo
    print(f"[{time.time()-t0:.0f}s] SVC on non-pseudo rows: base {balanced_accuracy(te.model[rest], labels[P_svc[rest].argmax(1)]):.4f}"
          f" -> refit {balanced_accuracy(te.model[rest], labels[P_pl[rest].argmax(1)]):.4f}")
    S2 = S.copy()
    S2[rest] += np.log(np.clip(P_pl[rest], 1e-9, 1)) - log_svc[rest]
    print(f"stack: {balanced_accuracy(te.model, labels[stack_pred]):.4f} -> with swapped SVC "
          f"{balanced_accuracy(te.model, labels[S2.argmax(1)]):.4f}")


if __name__ == "__main__":
    main(keep_frac=float(sys.argv[1]) if len(sys.argv) > 1 else 0.5)
