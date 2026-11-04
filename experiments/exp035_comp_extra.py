"""exp035: exp034's idea for the complementary stack. Its stylegram SVC (four TF-IDF views, LinearSVC) and its GPT / Gemini /
GLM family gates learn from the official rows plus public extra passages: one passage per (prompt, model) from prompts
answered by >= 12 models, 30-300 words (V31's load_extra_passages; with the 2026-10-30 batches, P30, 10,743 rows). The
open pool (crtproduction, release v11 = Kaggle version 13) groups its passages by prompt too (`prompt_group`): 2,560 of
its prompts have >= 12 models' answers. This fits the stack's plain stylegram SVC out of fold on exp015's folds with the
P30 rows and with those plus the open pool's multi-model prompts (build_nlm.py's CX), with the stack's weighting (the
extra rows of a class share half its weight), and scores it alone. With `gates` first it fits the GPT / Gemini / GLM family
gates instead (each its own StyleGrams + balanced LinearSVC on the family's official rows at weight 1 and its extra rows at
0.25, as comp_stack.py does) and scores each within its family on the held-out official rows. The centred twin is not refitted.

    python -I experiments/exp035_comp_extra.py [gates] <dl_dir> <built notebook.ipynb> [<built notebook.ipynb> ...]

Each built notebook supplies its load_extra_passages verbatim (the first is the reference); <dl_dir> holds the passage
files exp033 lists and ntl_open_pool.parquet. -I because the inputs are downloads; src/ and experiments/ go on the path.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import glob  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402
from ntl.paths import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "outputs" / "exp035_comp_extra"
NLM_OOF = REPO_ROOT / "outputs" / "exp015_nlm_core_oof" / "nlm_oof.npz"
EXTRA_MIN_LABELS, EXTRA_SHARE = 12, 0.5
GATE_C = [('gpt', 1.0), ('gemini', 1.0), ('glm', 0.5)]   # comp_stack.py's gates on the extra passages (the Claude gate uses API rows)
FUNCTION_WORDS = frozenset("a an the and or but if while although because as so than that which who whom whose this these those it its "
                           "they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with "
                           "without for of by over under between through into about is are was were be been being have has had do does did "
                           "can could will would shall should may might must not no nor all any some each every both either neither other "
                           "another such only also just very too more most less least here there when where why how what whether then now "
                           "still yet even much many few little".split())
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def shape_view(text):
    return re.sub(r"[^\W\d_]+", lambda m: 'X' if m[0].isupper() else 'x', text)


def function_view(text):
    return ' '.join(t.lower() if (t.lower() in FUNCTION_WORDS or not t.isalnum()) else 'CONTENT' for t in TOKEN_RE.findall(text))


def ident(text):
    return text


VIEWS = [('char', ident, (3, 5), 250_000, 1.0), ('word', ident, (1, 3), 180_000, 0.7),
         ('char', shape_view, (2, 6), 60_000, 0.5), ('word', function_view, (1, 3), 50_000, 0.5)]   # comp_stack.py's StyleGrams


def loader_rows(nb, dl_dir, labels):
    """The built notebook's load_extra_passages, run on local copies of its files (exp033's run_loader.py)."""
    src = "\n".join((c["source"] if isinstance(c["source"], str) else "".join(c["source"]))
                    for c in json.load(open(nb))["cells"] if c["cell_type"] == "code")
    i = src.index("def load_extra_passages():")
    d = dl_dir.rstrip("/") + "/"
    body = src[i:src.index("\nEXTRA = load_extra_passages()", i)].replace("'/kaggle/input/' + pattern", repr(d) + " + pattern") \
        .replace("'/kaggle/input/**/ntl_open_pool.parquet'", repr(d + "**/ntl_open_pool.parquet"))
    ns = dict(glob=glob, os=os, re=re, pd=pd, np=np, LABELS=labels, EXTRA_MIN_LABELS=EXTRA_MIN_LABELS)
    exec(body, ns)
    return ns["load_extra_passages"]()


def fit_fold(args):
    from scipy.sparse import vstack, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    tr_t, tr_y, ex_t, ex_y, va_t, k = args
    t0 = time.time()
    vecs = []
    for analyzer, view, ngr, maxf, w in VIEWS:
        opts = dict(analyzer=analyzer, ngram_range=ngr, lowercase=False, sublinear_tf=True, min_df=2, max_features=maxf, dtype=np.float64)
        if analyzer == 'word':
            opts['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
        vecs.append((TfidfVectorizer(**opts).fit([view(t) for t in list(tr_t) + list(ex_t)]), view, w))
    tf = lambda texts: hstack([v.transform([view(t) for t in texts]) * w for v, view, w in vecs], format='csr')
    n_tr, n_ex = np.bincount(tr_y, minlength=k).astype(float), np.bincount(ex_y, minlength=k).astype(float)
    sw = np.concatenate([np.where(n_ex[tr_y] > 0, 1.0 - EXTRA_SHARE, 1.0), EXTRA_SHARE * n_tr[ex_y] / np.maximum(n_ex[ex_y], 1.0)])
    svc = LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(vstack([tf(tr_t), tf(ex_t)]).tocsr(), np.concatenate([tr_y, ex_y]),
                                                                 sample_weight=sw)
    d = svc.decision_function(tf(va_t))
    assert list(svc.classes_) == list(range(k))
    return d, time.time() - t0


def fit_gate(args):
    """comp_stack.py's family_gate, fitted on one fold: StyleGrams on the family's texts, balanced LinearSVC; within-family
    predictions for the held-out official rows of the family."""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    texts, labels, sw, va_t, C = args
    t0 = time.time()
    vecs = []
    for analyzer, view, ngr, maxf, w in VIEWS:
        opts = dict(analyzer=analyzer, ngram_range=ngr, lowercase=False, sublinear_tf=True, min_df=2, max_features=maxf, dtype=np.float64)
        if analyzer == 'word':
            opts['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
        vecs.append((TfidfVectorizer(**opts).fit([view(t) for t in texts]), view, w))
    tf = lambda xs: hstack([v.transform([view(t) for t in xs]) * w for v, view, w in vecs], format='csr')
    svc = LinearSVC(C=C, class_weight='balanced', random_state=42, max_iter=4000).fit(tf(texts), labels, sample_weight=sw)
    d = svc.decision_function(tf(va_t))   # one column per class, or one signed score for two classes (GLM)
    return svc.classes_[d.argmax(1) if d.ndim == 2 else (d > 0).astype(int)], time.time() - t0


def gates(dl_dir, *nbs):
    train = load_train()
    labels = sorted(train.model.unique())
    l2i = {l: i for i, l in enumerate(labels)}
    folds = np.load(NLM_OOF)["folds"]
    texts, y = train.text.fillna("").astype(str).to_numpy(), train.model.map(l2i).to_numpy()
    fams = {f: np.array([i for i, l in enumerate(labels) if l.startswith(f)]) for f, _ in GATE_C}
    jobs, slots = [], []
    for nb in nbs:
        x = loader_rows(nb, dl_dir, labels)
        ex_t, ex_y = x.text.to_numpy(), x.model.map(l2i).to_numpy()
        for fam, C in GATE_C:
            me = np.isin(ex_y, fams[fam])
            for k in np.unique(folds):
                m = np.isin(y, fams[fam]) & (folds != k)
                jobs.append((list(texts[m]) + list(ex_t[me]), np.concatenate([y[m], ex_y[me]]),
                             np.concatenate([np.ones(int(m.sum())), np.full(int(me.sum()), 0.25)]),
                             list(texts[np.isin(y, fams[fam]) & (folds == k)]), C))
                slots.append((Path(nb).stem, fam, k, int(me.sum())))
    pred = {}
    with ProcessPoolExecutor(max_workers=int(os.environ.get("EXP035_WORKERS", "5"))) as pool:
        for (n, fam, k, n_ex), (p, secs) in zip(slots, pool.map(fit_gate, jobs)):
            pred[n, fam, k] = p
            print(f"{n} {fam} gate fold {k}: {n_ex} extra rows, {secs:.0f}s", flush=True)
    for fam, _ in GATE_C:
        for n in dict.fromkeys(s_[0] for s_ in slots):
            rows = np.concatenate([np.flatnonzero(np.isin(y, fams[fam]) & (folds == k)) for k in np.unique(folds)])
            p = np.concatenate([pred[n, fam, k] for k in np.unique(folds)])
            print(f"{fam} gate, {n}: within-family balanced accuracy {balanced_accuracy(train.model.to_numpy()[rows], np.array(labels)[p]):.4f}", flush=True)


def main(dl_dir, *nbs):
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_train()
    labels = sorted(train.model.unique())
    l2i = {l: i for i, l in enumerate(labels)}
    folds = np.load(NLM_OOF)["folds"]
    extras = {Path(nb).stem: loader_rows(nb, dl_dir, labels) for nb in nbs}
    for n, x in extras.items():
        print(f"{n}: {len(x)} rows, {x.prompt_id.nunique()} prompts, per label {x.model.value_counts().min()}-{x.model.value_counts().max()}", flush=True)
    texts, y = train.text.fillna("").astype(str).to_numpy(), train.model.map(l2i).to_numpy()
    jobs, slots = [], []
    for n, x in extras.items():
        ex_t, ex_y = x.text.to_numpy(), x.model.map(l2i).to_numpy()
        for k in np.unique(folds):
            jobs.append((texts[folds != k], y[folds != k], ex_t, ex_y, texts[folds == k], len(labels)))
            slots.append((n, folds == k))
    oof = {n: np.zeros((len(train), len(labels))) for n in extras}
    with ProcessPoolExecutor(max_workers=int(os.environ.get("EXP035_WORKERS", "5"))) as pool:
        for (n, valid), (d, secs) in zip(slots, pool.map(fit_fold, jobs)):
            oof[n][valid] = d
            print(f"{n}: fold done in {secs:.0f}s", flush=True)
    ref = None
    for n, d in oof.items():
        np.save(OUT / f"{n}_stylegram.npy", d)
        pred = d.argmax(1)
        line = f"{n} stylegram SVC alone: {balanced_accuracy(train.model, np.array(labels)[pred]):.4f}"
        if ref is not None:
            g = paired_bootstrap(y, pred, ref, train.prompt_id.to_numpy())
            line += f" (vs the first {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}])"
        else:
            ref = pred
        print(line, flush=True)


if __name__ == "__main__":
    gates(*sys.argv[2:]) if sys.argv[1] == "gates" else main(*sys.argv[1:])
