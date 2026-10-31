"""exp020: Roman Tamrazov's "Who am I?" step (public 0.654 = nlm-v27 0.651 + this), out of fold on the exp015 replica.

His step ("v38a", appended to infer.py) takes the ensemble's log-probabilities after Sinkhorn, before nlm-v27's post
step (em-dash guards, then a one-to-one Hungarian assignment inside each estimated test prompt cluster of 12-18
passages), and instead of that post step adds a "typography" log-likelihood: 34 binary markers (dashes, curly quotes,
colons, bullets, "However," ...), per-class Bernoulli rates from train ((hits + 2) / (n + 4), clipped to
[0.015, 0.985]), averaged over the markers, centred per row, doubled and clipped to +-0.35, then gated by
(0.8 - top-2 margin) / 0.8. So it changes two things at once: the post step goes, the marker evidence comes in.

Here the rates are fitted on the other folds (prompt-grouped, the adapters' folds), the base is exp015's `--plain`
replica (core + DeBERTa stand-in, Sinkhorn), and the post step uses upstream's clustering of the passages (TF-IDF
similarity, capped average linkage, threshold 0.15) run on train's texts, which come in 150 prompts of 18 like the
hidden test is assumed to. Optional: harshitakumari256's Qwen3-4B ATLAS OOF (exp019) mixed at 10% on low-margin rows.

    PYTHONPATH=src python experiments/exp020_typobayes_oof.py [path/to/atlas4b/oof_all.parquet]
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax

sys.argv.append("--plain")
import exp015_nlm_core_oof as E  # noqa: E402  (reads --plain at import)
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402
from ntl.paths import REPO_ROOT  # noqa: E402

UPSTREAM = REPO_ROOT / "kaggle" / "nlm" / "upstream_nlm_v27.ipynb"
# Roman's markers, verbatim
MARKS = [' — ', '—', '–', '’', '‘', '“', '”', '«', '»', '"', ';', ':', '(', ')', '[', ']', '**', '`', '…', '•', '\n',
         '\n\n', 'e.g.', 'i.e.', 'However,', 'Overall,', 'In conclusion', 'Therefore,', 'Moreover,', '\t', ' - ', ' = ',
         '\n- ', '\n1.']


def upstream_guards():
    """nlm-v27's GUARDS list, read from the pinned upstream infer.py."""
    nb = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    src = "".join(nb["cells"][2]["source"]).splitlines()
    i = next(i for i, l in enumerate(src) if l.lstrip().startswith("GUARDS = ["))
    j = next(j for j in range(i, len(src)) if src[j].rstrip().endswith(")]"))
    return eval("\n".join(l.strip() for l in src[i:j + 1]).split("=", 1)[1], {"LABELS": E.NLM_LABELS})


def capped_agglo(S, cap=18, thr=0.02):
    """upstream's average-linkage clustering, verbatim."""
    n = len(S); A = S.astype(np.float32).copy(); np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32); alive = np.ones(n, bool); lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf); k = int(np.argmax(M)); i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr: break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf; A[j] = -np.inf; A[:, j] = -np.inf
        size[i] += size[j]; alive[j] = False; lab[lab == j] = i
    return lab


def post(tot, texts, groups, guards):
    """upstream's post step: em-dash guards, then dense-cluster (12-18) Hungarian."""
    tot = tot.copy()
    for mark, allowed in guards:
        forb = np.array([l not in allowed for l in E.NLM_LABELS])
        for i, t in enumerate(texts):
            if mark in t and E.NLM_LABELS[int(tot[i].argmax())] not in allowed:
                tot[i, forb] = -1e9
    sz = np.bincount(groups)[groups]
    for g in np.unique(groups[(sz >= 12) & (sz <= 18)]):
        ii = np.where(groups == g)[0]; r, c = linear_sum_assignment(-tot[ii])
        tot[ii[r]] = -1e9; tot[ii[r], c] = 0.0
    return tot


def typo_evidence(texts, y, folds):
    """Roman's marker evidence, with the class rates fitted on the other folds."""
    X = np.array([[float(m in t) for m in MARKS] for t in texts], dtype=np.float32)
    ev = np.zeros((len(texts), len(E.NLM_LABELS)))
    for k in np.unique(folds):
        fit, val = folds != k, folds == k
        freq = np.bincount(y[fit], minlength=len(E.NLM_LABELS)).astype(float)
        hits = np.array([X[fit][y[fit] == c].sum(0) for c in range(len(E.NLM_LABELS))], dtype=np.float64)
        rate = np.clip((hits + 2.0) / (freq[:, None] + 4.0), 0.015, 0.985)
        e = (X[val] @ np.log(rate).T + (1 - X[val]) @ np.log(1 - rate).T) / len(MARKS)
        e -= e.mean(1, keepdims=True)
        ev[val] = np.clip(2.0 * e, -.35, .35)
    return ev


def main():
    train = load_train()
    labels = np.array(E.NLM_LABELS)
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    texts = train.text.to_numpy()
    nlm = np.load(E.NLM_OOF); folds = nlm["folds"]
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    members["svc4"] = np.load(E.OUT / "svc4_oof.npy").astype(np.float64)
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
    assert np.array_equal(z["ids"], train.id.to_numpy(dtype=str))
    members["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    score = lambda pred: balanced_accuracy(train.model, labels[pred])

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import adjusted_rand_score
    Xc = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05).fit_transform(texts)
    groups = capped_agglo((Xc @ Xc.T).toarray(), 18, 0.15)
    sizes = np.bincount(groups)[np.bincount(groups) > 0]
    print(f"clusters on train texts: {len(sizes)} (true 150), ARI {adjusted_rand_score(prompts, groups):.4f}, "
          f"rows in 12-18 clusters {np.isin(groups, np.where((np.bincount(groups) >= 12) & (np.bincount(groups) <= 18))[0]).mean():.3f}")
    guards = upstream_guards()
    ev = typo_evidence(texts, y, folds)
    print(f"marker evidence alone: {score(ev.argmax(1)):.4f}")

    P0 = E.blend(members, {**E.CORE, "deb": 1.0})
    bases = {"replica": P0}
    if len(sys.argv) > 1 and sys.argv[1].endswith(".parquet"):
        a = pd.read_parquet(sys.argv[1]).set_index("id").loc[train.id.astype(str)]
        LL = a[[f"LL_{l}" for l in E.NLM_LABELS]].to_numpy(np.float64)
        A = np.exp(14.820207057988586 * (LL - LL.max(1, keepdims=True))); A /= A.sum(1, keepdims=True)
        s = np.sort(P0, 1); rows = (s[:, -1] - s[:, -2]) < 0.3
        P = P0.copy(); P[rows] = 0.9 * P0[rows] + 0.1 * A[rows]
        bases["replica + atlas4b 10%"] = P
    ref = None
    for name, P in bases.items():
        raw = np.log(E.sinkhorn(P) + 1e-12)      # infer.py's tot after Sinkhorn (sinkhorn 1.0)
        q = softmax(raw, 1); s = np.sort(q, 1); gate = np.clip((0.8 - (s[:, -1] - s[:, -2])) / 0.8, 0., 1.)
        out = {"no post": raw.argmax(1),
               "upstream post (0.651)": post(raw, texts, groups, guards).argmax(1),
               "Hungarian only": post(raw, texts, groups, []).argmax(1),
               "guards only": post(raw, texts, np.arange(len(raw)), guards).argmax(1),
               "v38a safe (0.654)": (raw + gate[:, None] * ev).argmax(1),
               "v38a full": (raw + ev).argmax(1),
               "v38a safe, then upstream post": post(raw + gate[:, None] * ev, texts, groups, guards).argmax(1)}
        if ref is None:
            ref = out["upstream post (0.651)"]
        print(f"\n{name} (gate mean {gate.mean():.3f}):")
        for tag, pred in out.items():
            g = paired_bootstrap(y, pred, ref, prompts)
            print(f"  {tag:32s} {score(pred):.4f}  vs 0.651 replica {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}]  "
                  f"changed {(pred != ref).mean():.3f}")


if __name__ == "__main__":
    main()
