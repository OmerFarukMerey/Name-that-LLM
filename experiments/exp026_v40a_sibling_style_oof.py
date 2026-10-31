"""exp026: Roman Tamrazov's "V40A" (Who am I? v33, pushed 2026-10-28 15:57 UTC) out of fold: mercurydust's 0.660 plus a
family-only stylometry residual. Within each of the claude, gemini, gpt and glm families a balanced logistic regression
on handcrafted style features (punctuation/markdown markers, function words, length statistics) is fitted on the train
texts; its reliability is clip((5-fold GroupKFold balanced accuracy - chance) / 0.15, 0, 1); on test rows its sibling
probabilities are mixed into the family's renormalised probabilities with strength max_alpha x reliability x
clip((0.65 - within-family margin) / 0.65), keeping each family's total mass. Roman's max_alpha is 0.08. The residual
is applied after the typography step, on exp025's replica (core + deb, V31's Sinkhorn, typography v2 with V31's gate
or the margin gate only), fitted on the other folds (nested: its own GroupKFold runs inside the fit folds).

    PYTHONPATH=src:experiments python experiments/exp026_v40a_sibling_style_oof.py
"""

import re
from functools import lru_cache

import numpy as np
from scipy.special import softmax

import exp025_v31_typography_oof as V
from ntl.data import load_train
from ntl.metrics import balanced_accuracy
from ntl.oofbank import paired_bootstrap

E, U, T = V.E, V.U, V.T

# Roman's sibling_style.py (v33, cell 3), verbatim apart from the diagnostics file
MARKERS = [',','.', ';', ':', '!', '?','—','–','-','’','‘','“','”','"',"'",'(',')','[',']','{','}','*','`','•','…','/','\\',':','\n','\n\n','\n-','\n1.','\n2.','##','```','**','> ','\t',' = ', ' e.g.', ' i.e.', 'However,','Overall,','In conclusion','Therefore,','Moreover,','First,','Finally,',' - ',': ', ' ; ', '  ']
WORDS = ['the','and','but','that','which','with','into','from','for','you','your','our','we','I','it','its','they','their','there','this','these','those','however','therefore','moreover','although','because','perhaps','might','would','could','should','like','such','just','very','also','not','if','or','as','so','by','of','to','in','on','at','is','are','be','was','were','have','has','had','may','must','all','any','each','every','many','some','more','most','only','even','still','then','when','where','why','how','what','who']
WRE = re.compile(r"[A-Za-z]+(?:[’'][A-Za-z]+)?")
SRE = re.compile(r'(?<=[.?!])\s+')


@lru_cache(maxsize=15000)
def row(t):
    t = str(t)
    s = WRE.findall(t); lo = [z.lower() for z in s]; n = max(len(s), 1); nt = max(len(t), 1); ns = max(len(t.splitlines()), 1)
    lens = np.array([len(z) for z in s] or [0]); sent_lens = np.array([len(WRE.findall(z)) for z in SRE.split(t) if z.strip()] or [0])
    base = [np.log1p(n), np.log1p(nt), nt/n, ns/n, len(set(lo))/n, float(np.mean(lens)), float(np.std(lens)), float(np.mean(sent_lens)), float(np.std(sent_lens)), float(np.max(sent_lens)), float(np.mean(lens >= 8)), sum(c.isupper() for c in t)/nt, sum(c.isdigit() for c in t)/nt, sum(ord(c) > 127 for c in t)/nt, sum(c.isspace() for c in t)/nt, sum(c == '\n' for c in t)/nt]
    base += [min(t.count(m)/n, 3) for m in MARKERS]
    cnt = {}
    for w in lo: cnt[w] = cnt.get(w, 0)+1
    base += [cnt.get(w, 0)/n for w in WORDS]
    return np.array(base, dtype=np.float32)


def features(texts):
    return np.stack([row(str(t)) for t in texts])


def apply_grouped_style_residual(logits, train_text, test_text, labels, y, groups, max_alpha=0.08):
    from scipy.special import logsumexp
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import balanced_accuracy_score
    Xt = features(train_text)
    Xq = features(test_text)
    out = np.asarray(logits, dtype=np.float64).copy()
    lab = np.asarray(labels)
    yy = np.asarray(y)
    g = np.asarray(groups)
    diag = {}
    for family, C in [('claude', 0.003), ('gemini', 0.003), ('gpt', 0.003), ('glm', 0.03)]:
        cols = np.flatnonzero(np.char.startswith(lab.astype(str), family))
        if len(cols) < 2:
            continue
        train_mask = np.isin(yy, cols)
        inds = np.flatnonzero(train_mask)
        X = Xt[inds]
        target = np.searchsorted(cols, yy[inds])
        grp = g[inds]
        oof = np.empty(len(target), dtype=np.int32)
        for a, b in GroupKFold(n_splits=5).split(X, target, groups=grp):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=400, class_weight='balanced'))
            model.fit(X[a], target[a])
            oof[b] = model.predict(X[b])
        bacc = float(balanced_accuracy_score(target, oof))
        chance = 1.0 / len(cols)
        reliability = np.clip((bacc - chance) / 0.15, 0.0, 1.0)
        if reliability <= 0:
            diag[family] = dict(oof_balanced_accuracy=bacc, random_level=chance, alpha_mean=0.)
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=400, class_weight='balanced'))
        model.fit(X, target)
        style_p = np.maximum(model.predict_proba(Xq), 1e-8)
        old_logits = out[:, cols].copy()
        old_p = softmax(old_logits, axis=1)
        margin = np.partition(old_p, -2, axis=1)[:, -1] - np.partition(old_p, -2, axis=1)[:, -2]
        gate = np.clip((0.65 - margin) / 0.65, 0., 1.)
        strength = (max_alpha * reliability) * gate
        mix = (1. - strength[:, None]) * old_p + strength[:, None] * style_p
        out[:, cols] = logsumexp(old_logits, axis=1, keepdims=True) + np.log(np.maximum(mix, 1e-12))
        assert np.allclose(logsumexp(out[:, cols], axis=1), logsumexp(old_logits, axis=1), atol=1e-8)
        diag[family] = dict(oof_balanced_accuracy=round(bacc, 4), random_level=round(chance, 4),
                            alpha_mean=round(float(strength.mean()), 4), changed_within_family=int(np.sum(old_p.argmax(1) != mix.argmax(1))))
    return out, diag


def main():
    train = load_train()
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts, texts = train.prompt_id.to_numpy(), train.text.to_numpy()
    nlm = np.load(E.NLM_OOF); folds = nlm["folds"]
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    members["svc4"] = np.load(E.OUT / "svc4_oof.npy").astype(np.float64)
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
    assert np.array_equal(z["ids"], train.id.to_numpy(dtype=str))
    members["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    P0 = E.blend(members, {**E.CORE, "deb": 1.0})
    prior = (np.bincount(y, minlength=18) + 1.0) / (len(y) + 18.0)
    words = np.array([len(re.findall(r'\b\w+\b', t)) for t in texts], dtype=np.float64)
    X37 = U.markers(texts, V.MARKS_V31)
    raw = V.soft_sinkhorn(P0, prior, 0.45, 0.35)
    ev = V.oof(V.v2_evidence, X37, y, folds)
    gates = {"V31 gate": V.gate_v2, "margin gate only": lambda r, X, w: np.clip((0.6 - V.margin(r)) / 0.6, 0., 1.)}
    ref = (raw + V.gate_v2(raw, X37, words)[:, None] * ev).argmax(1)
    print(f"V31 replica (V31 gate, no residual): {balanced_accuracy(y, ref):.4f}")
    for gname, gate in gates.items():
        safe = raw + gate(raw, X37, words)[:, None] * ev
        p = safe.argmax(1); c = paired_bootstrap(y, p, ref, prompts)
        print(f"  {gname:17s} no residual      {balanced_accuracy(y, p):.4f}  vs V31 {c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]")
        for alpha in (0.08, 0.2, 0.5):
            out, diags = np.empty_like(safe), []
            for f in np.unique(folds):
                fit = folds != f
                out[~fit], d = apply_grouped_style_residual(safe[~fit], texts[fit], texts[~fit], E.NLM_LABELS, y[fit],
                                                            prompts[fit], max_alpha=alpha)
                diags.append(d)
            p = out.argmax(1); c = paired_bootstrap(y, p, ref, prompts)
            print(f"  {gname:17s} V40A alpha {alpha:<5} {balanced_accuracy(y, p):.4f}  vs V31 {c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}]"
                  f"  changed {(p != ref).mean():.3f}")
            if alpha == 0.08 and gname == "V31 gate":
                for fam in ("claude", "gemini", "gpt", "glm"):
                    print(f"      {fam}: " + "  ".join(f"bacc {d[fam]['oof_balanced_accuracy']:.3f} a {d[fam]['alpha_mean']:.4f} "
                                                     f"chg {d[fam].get('changed_within_family', 0)}" for d in diags if fam in d))


if __name__ == "__main__":
    main()
