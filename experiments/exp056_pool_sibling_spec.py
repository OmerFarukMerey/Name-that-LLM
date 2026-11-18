"""exp056: areebbb's Open Pool sibling specialists (NTL-40, "0.701 | Read the Fingerprints", 11-17) on our stand-in.
Inside Google, OpenAI and Anthropic a char_wb(1-5) + word(1-2) TF-IDF, SGD log-loss model trained on the open pool
only (<= 12,000 passages per model) re-splits the lab's mass between its siblings: L[:, lab] = lse + log_softmax(
L[:, lab] + w * centred spec log-probs). The pool holds no train-prompt rows, so its scores on train are out of fold
by construction. Our transcription of spec_worker.py / spec_apply (their code is never run).

Out of fold (ntl.cv prompt folds) on exp050's stand-in stack, through the 0.702's tail: the GLM stage at cover 0.15
(his head and our gbm head, shuffled GLM-5.3 view), then apex_postprocess (sibling residual strength 0.10, GLM rows
kept, soft typography penalty 0.05) reading the specialised logits, as NTL-40's section 19 feeds section 20.

Second base: lightningv08's public out-of-fold bank (ntl.oofbank, nested_calibrated, scenario 3: 0.544), the kind of
weak reader OOF NTL-40's table was measured on. Specialist variants (VARIANTS) are scored on both bases.

    python -I experiments/exp056_pool_sibling_spec.py <base.npz> <ntl_open_pool.parquet> [variant ...]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import hashlib  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "kaggle" / "glmll")]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.special import log_softmax, logsumexp, softmax  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression, SGDClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import exp051_glm_cut160 as Y  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402

X = Y.X
OUT = Path(os.environ.get("EXP056_OUT", _ROOT / "outputs" / "exp056_pool_sibling_spec"))
LABS = ("gemini", "gpt", "claude")
SPEC_REF = {"gemini": 0.9243, "gpt": 0.9058, "claude": 1.1947}   # NTL-40: mean within-lab std of its OOF base


VARIANTS = {   # name: fit_specialists options; "ntl40" is NTL-40's worker as published
    "ntl40": {},
    "nocap": {"cap": 10 ** 9},
    "noproxy": {"drop_proxy": True},
    "mw": {"weighted": True},
    "a1e-5": {"alpha": 1e-5},
    "a5e-7": {"alpha": 5e-7},
    "train5": {"train_w": 5.0},   # + the lab's train rows of the other prompt folds, weight 5 each
}


# ---- transcription of NTL-40's spec_worker.py (cap, alpha: its constants), with our variant options ----
def fit_specialists(pool_path, texts, labs=LABS, cap=12000, alpha=2e-6, drop_proxy=False, weighted=False,
                    train_w=0.0, train=None, folds=None):
    """lab -> (sibling labels, centred sibling log-probs of `texts`) from models trained on the open pool (plus, with
    train_w > 0, the lab's train rows outside each prompt fold, so the scores stay out of fold)."""
    t0 = time.time()
    pool = pd.read_parquet(pool_path, columns=["text", "model", "flag_train_prompt_regen", "flag_near_train_prompt",
                                               "flag_kb_proxy_after_ruling", "match_weight"])
    pool["text"] = pool["text"].fillna("").astype(str)
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    seen = set(hashlib.md5(norm(s).encode()).hexdigest() for s in texts)
    keep = (~pool.flag_train_prompt_regen.astype(bool) & ~pool.flag_near_train_prompt.astype(bool)
            & ~pool.text.map(lambda s: hashlib.md5(norm(s).encode()).hexdigest() in seen))
    if drop_proxy:
        keep &= ~pool.flag_kb_proxy_after_ruling.astype(bool)
    pool = pool[keep & pool.model.isin(X.LABELS)].drop_duplicates("text")
    out = {}
    for lab in labs:
        sib = [m for m in X.LABELS if m.startswith(lab + "-")]
        P = pool[pool.model.isin(sib)]
        P = pd.concat([g.sample(min(len(g), cap), random_state=0) for _, g in P.groupby("model")])
        parts = [(np.ones(len(texts), bool), np.zeros(len(texts), bool))] if train_w <= 0 else \
            [(folds == k, (folds != k) & train.model.isin(sib).to_numpy()) for k in np.unique(folds)]
        S = np.zeros((len(texts), len(sib)))
        for va, tr in parts:
            T = pd.concat([P[["text", "model"]], train.loc[tr, ["text", "model"]]]) if tr.any() else P
            sw = np.r_[np.clip(P.match_weight.to_numpy(), 0.05, 5.0) if weighted else np.ones(len(P)),
                       np.full(int(tr.sum()), train_w)]
            ch = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), sublinear_tf=True, min_df=3,
                                 max_features=600000, lowercase=False)
            wd = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=3,
                                 max_features=300000, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
            A = sparse.hstack([ch.fit_transform(T.text), wd.fit_transform(T.text)]).tocsr()
            clf = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=20, tol=None, class_weight="balanced",
                                random_state=0).fit(A, T.model.values, sample_weight=sw)
            vt = [texts[i] for i in np.flatnonzero(va)]
            B = sparse.hstack([ch.transform(vt), wd.transform(vt)]).tocsr()
            cls = list(clf.classes_)
            S[va] = np.log(np.clip(clf.predict_proba(B)[:, [cls.index(s) for s in sib]], 1e-9, None))
        out[lab] = (sib, S - S.mean(1, keepdims=True))
        print(f"specialist {lab}: {len(P)} pool rows, {time.time() - t0:.0f}s", flush=True)
    return out


def spec_apply(L, spec, w, rescale=True):
    """NTL-40's spec_apply on logits L (labels in X.LABELS order); rescale=False uses w as is."""
    L = np.asarray(L, dtype=np.float64).copy()
    for lab, (sib, S) in spec.items():
        idx = [X.LABELS.index(m) for m in sib]
        wl = w
        if rescale:
            inlab = np.isin(L.argmax(1), idx)
            spread = L[inlab][:, idx].std(1).mean() if inlab.sum() >= 20 else SPEC_REF[lab]
            wl = w * float(np.clip(spread / SPEC_REF[lab], 0.5, 2.5))
        lse = np.logaddexp.reduce(L[:, idx], axis=1, keepdims=True)
        L[:, idx] = lse + log_softmax(L[:, idx] + wl * S, axis=1)
    return L


# ---- transcription of pngy_residual.py and the 0.702's apex_postprocess.py ----
MARKERS = (",", ".", ";", ":", "!", "?", "—", "–", "’", "“", "”",
           "'", '"', "(", ")", "[", "]", "*", "", "**", " - ", "  ")
FUNCTIONS = ("the", "and", "but", "which", "however", "therefore", "moreover", "because", "perhaps",
             "might", "would", "could", "should", "also", "although", "overall", "finally", "we", "you", "i")
WORDS = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
FAMS = [np.array([i for i, l in enumerate(X.LABELS) if l.startswith(f + "-")]) for f in ("claude", "gpt", "gemini", "glm")]


def style_features(texts):
    rows = []
    for v in texts:
        counts = np.array([v.count(m) for m in MARKERS], dtype=np.float64)
        tokens = WORDS.findall(v)
        n, chars = max(1, len(tokens)), max(1, len(v))
        lengths = np.array([len(t) for t in tokens] or [0], dtype=float)
        lower = [t.lower() for t in tokens]
        shape = [np.log1p(len(tokens)), np.log1p(len(v)), chars / n, len(set(lower)) / n, lengths.mean(), lengths.std(),
                 sum(c.isupper() for c in v) / chars, sum(c.isdigit() for c in v) / chars,
                 sum(ord(c) > 127 for c in v) / chars, sum(c.isspace() for c in v) / chars]
        rows.append(np.r_[shape, counts / n, [lower.count(w) / n for w in FUNCTIONS]])
    return np.asarray(rows, dtype=np.float64)


def refine(z, evidence, strength=0.10):
    out = z.copy()
    for cols in FAMS:
        p = softmax(z[:, cols], axis=1)
        q = np.maximum(evidence[:, cols], 1e-15)
        q /= q.sum(1, keepdims=True)
        top = np.partition(p, -2, axis=1)[:, -2:]
        gate = np.clip((0.65 - (top[:, 1] - top[:, 0])) / 0.65, 0, 1) * strength
        mixed = (1 - gate[:, None]) * p + gate[:, None] * q
        out[:, cols] = logsumexp(z[:, cols], axis=1, keepdims=True) + np.log(mixed)
    return out


def residual_evidence(Str, ytr, Sva):
    ev = np.ones((len(Sva), 18))
    for cols in FAMS:
        sel = np.isin(ytr, cols)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=500, class_weight="balanced",
                                                               random_state=166)).fit(Str[sel], ytr[sel])
        ev[:, m.classes_] = m.predict_proba(Sva)
    return ev


SPACED_OK = np.isin(X.LABELS, ["claude-opus-4-6", "claude-opus-5", "glm-5.3", "glm-5.3-flash"])
RULES = ((" — ", ~SPACED_OK), ("’", np.array([l.startswith(("claude-", "glm-")) for l in X.LABELS])),
         ("—", np.array([l.startswith("gemini-") for l in X.LABELS])))
GLM = np.isin(np.arange(18), X.GLM)


def apex(stage, refined, texts, penalty=0.05):
    """GLM rows keep the stage's call, the rest take argmax(refined); then the soft typography."""
    out = np.where(GLM[stage], stage, refined.argmax(1))
    P = softmax(refined, axis=1)
    for i, t in enumerate(texts):
        for marker, forbidden in RULES:
            if marker in t and forbidden[out[i]]:
                P[i, forbidden] *= penalty
                if P[i].sum() > 0:
                    P[i] /= P[i].sum()
                    out[i] = P[i].argmax()
    return out


def load_spec(name, pool_path, texts, train, folds):
    cache = OUT / f"spec_{name}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: (z[k + "_labels"].tolist(), z[k]) for k in LABS}
    print(f"fitting {name}", flush=True)
    spec = fit_specialists(pool_path, texts, train=train, folds=folds, **VARIANTS[name])
    np.savez(cache, **{k: v[1] for k, v in spec.items()}, **{k + "_labels": np.array(v[0]) for k, v in spec.items()})
    return spec


def main(base_path, pool_path, *names):
    names = names or tuple(VARIANTS)
    train = X.load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    tt = pd.DataFrame({"text": texts, "model": train.model.to_numpy()})
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.astype(str).to_numpy()
    folds = X.prompt_folds(prompts)
    lab = np.asarray(X.LABELS)
    score = lambda p: X.balanced_accuracy(train.model, lab[p])
    base = np.load(base_path)
    assert (base["y"] == y).all() and base["ids"].tolist() == ids
    bank = np.load(_ROOT / "data" / "external" / "oofbank" / "oof_predictions.npz", allow_pickle=False)
    assert bank["ids"].tolist() == ids
    cls = bank["classes"].tolist()
    bases = {"stand-in": base["zl"],
             "bank": log_softmax(bank["nested_calibrated"][1][:, [cls.index(l) for l in X.LABELS]], axis=1)}
    OUT.mkdir(parents=True, exist_ok=True)

    names54 = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    n54 = len(X.HIS["features"])
    FB, _ = Y.frame(Y.load(Y.D160, "flash", ids), Y.load(Y.D160 / "s1", "glm53", ids), texts, names54)
    h = X.HIS["entropy_ref"]
    S_ = style_features(texts)
    heads = {k: np.zeros((len(y), 18)) for k in ("his", "gbm")}
    ev = np.zeros((len(y), 18))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        sc, lr = X.fit(FB[tr][:, :n54], y[tr], X.HIS["C"])
        gs = StandardScaler().fit(FB[tr])
        clf = HistGradientBoostingClassifier(**X.GBM).fit(gs.transform(FB[tr]), y[tr])
        l_lr = lr.predict_log_proba(sc.transform(FB[va][:, :n54]))
        heads["his"][va] = l_lr
        heads["gbm"][va] = X.mix(l_lr, np.log(np.clip(clf.predict_proba(gs.transform(FB[va])), 1e-12, None)))
        ev[va] = residual_evidence(S_[tr], y[tr], S_[va])
    tails = {}
    for b, z in bases.items():
        for hd in ("his", "gbm"):
            st = X.blend(z, heads[hd], 0.15, 1.0, h)
            tails[b, hd] = st, apex(st, refine(z, ev), texts)
            print(f"{b:8s} {hd}: stack {score(z.argmax(1)):.4f}, GLM stage (cover 0.15) {score(st):.4f}, "
                  f"0.702 tail {score(tails[b, hd][1]):.4f}", flush=True)

    ws = (0.10, 0.20, 0.30)
    for name in names:
        spec = load_spec(name, pool_path, texts, tt, folds)
        acc = []
        for k, (sib, S) in spec.items():
            idx = np.array([X.LABELS.index(m) for m in sib])
            rows = np.isin(y, idx)
            acc.append(f"{k} {np.mean(idx[S[rows].argmax(1)] == y[rows]):.3f}")
        print(f"\n{name}: specialist alone inside the lab: " + ", ".join(acc), flush=True)
        for b, z in bases.items():
            print(f"  {b:8s} stack + spec: " + ", ".join(
                f"w {w:.2f} {score(spec_apply(z, spec, w, False).argmax(1)) - score(z.argmax(1)):+.4f}" for w in ws),
                flush=True)
            for hd in ("his", "gbm"):
                st, ref = tails[b, hd]
                cells = []
                for w in ws:
                    p = apex(st, refine(spec_apply(z, spec, w, False), ev), texts)
                    g = paired_bootstrap(y, p, ref, prompts, n=1000)
                    cells.append(f"w {w:.2f} {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}]")
                print(f"  {b:8s} {hd} tail + spec: " + "; ".join(cells), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
