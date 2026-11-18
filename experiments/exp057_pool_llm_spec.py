"""exp057: an LLM trained on the open pool only, used like NTL-40's sibling specialists (exp056). Our TPU kernel
ntl-tpu-op8 (11-04, kaggle/member/tpu_member.py) trained a Qwen3-8B-Base LoRA24 classifier on open pool v13 (190,588
rows, stage 1) before its official-row stage 2; its stage-1 model's log-probs on every train row (s1_on_train.npy, 0.602
balanced accuracy alone) never saw an official row, so they are out of fold by construction, like exp056's TF-IDF.

Out of fold (ntl.cv prompt folds) on exp056's two bases (stand-in stack, public OOF bank), through the 0.702's tail
(GLM stage cover 0.15, apex residual + soft typography): op8 s1 as a within-lab specialist (centred log-probs inside
Google / OpenAI / Anthropic, lab mass kept) and as an 18-class member (z + a * log p, renormalised), against the TF-IDF
specialists of exp056.

Caveat: op8's driver dropped only flag_train_prompt_regen rows; pool v13 may hold passages near train prompts that
later pool releases flag (flag_near_train_prompt), which would flatter s1 on train.

    python -I experiments/exp057_pool_llm_spec.py <base.npz> [combo]   # combo: on top of NTL-40 as published
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "kaggle" / "glmll")]
import numpy as np  # noqa: E402
from scipy.special import log_softmax  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import exp056_pool_sibling_spec as P  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402

X, Y = P.X, P.Y
OUT = Path(os.environ.get("EXP057_OUT", _ROOT / "outputs" / "exp057_pool_llm_spec"))
LABELS18 = ["claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
            "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3",
            "glm-5.3-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "kimi-k3",
            "qwen3.8-max"]   # op8's summary.json label order


def as_spec(L):
    """18-class log-probs (X.LABELS order) -> exp056's specialist format: centred log-probs inside each lab."""
    out = {}
    for lab in P.LABS:
        sib = [m for m in X.LABELS if m.startswith(lab + "-")]
        S = log_softmax(L[:, [X.LABELS.index(m) for m in sib]], axis=1)
        out[lab] = (sib, S - S.mean(1, keepdims=True))
    return out


def main(base_path):
    train = X.load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.astype(str).to_numpy()
    folds = X.prompt_folds(prompts)
    lab = np.asarray(X.LABELS)
    score = lambda p: X.balanced_accuracy(train.model, lab[p])
    base = np.load(base_path)
    assert (base["y"] == y).all() and base["ids"].tolist() == ids
    bank = np.load(_ROOT / "data" / "external" / "oofbank" / "oof_predictions.npz", allow_pickle=False)
    cls = bank["classes"].tolist()
    bases = {"stand-in": base["zl"],
             "bank": log_softmax(bank["nested_calibrated"][1][:, [cls.index(l) for l in X.LABELS]], axis=1)}

    s1 = np.load(OUT / "op8" / "s1_on_train.npy", allow_pickle=False).astype(np.float64)
    assert s1.shape == (len(y), 18)
    s1 = log_softmax(s1[:, [LABELS18.index(l) for l in X.LABELS]], axis=1)
    print(f"op8 s1 alone {score(s1.argmax(1)):.4f} (summary.json 0.6020)", flush=True)
    members = {"op8 s1 (LLM, pool only)": as_spec(s1),
               "TF-IDF (NTL-40)": P.load_spec("ntl40", None, texts, None, folds)}
    for nm, spec in members.items():
        acc = []
        for k, (sib, S) in spec.items():
            idx = np.array([X.LABELS.index(m) for m in sib])
            rows = np.isin(y, idx)
            acc.append(f"{k} {np.mean(idx[S[rows].argmax(1)] == y[rows]):.3f}")
        print(f"{nm}: alone inside the lab: " + ", ".join(acc), flush=True)
    for b, z in bases.items():
        acc = []
        for k, (sib, _) in members["TF-IDF (NTL-40)"].items():
            idx = np.array([X.LABELS.index(m) for m in sib])
            rows = np.isin(y, idx)
            acc.append(f"{k} {np.mean(idx[z[rows][:, idx].argmax(1)] == y[rows]):.3f}")
        print(f"{b} stack inside the lab: " + ", ".join(acc), flush=True)

    names54 = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    n54 = len(X.HIS["features"])
    FB, _ = Y.frame(Y.load(Y.D160, "flash", ids), Y.load(Y.D160 / "s1", "glm53", ids), texts, names54)
    h = X.HIS["entropy_ref"]
    S_ = P.style_features(texts)
    head = np.zeros((len(y), 18))
    ev = np.zeros((len(y), 18))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        sc, lr = X.fit(FB[tr][:, :n54], y[tr], X.HIS["C"])
        head[va] = lr.predict_log_proba(sc.transform(FB[va][:, :n54]))
        ev[va] = P.residual_evidence(S_[tr], y[tr], S_[va])

    for b, z in bases.items():
        st = X.blend(z, head, 0.15, 1.0, h)
        ref = P.apex(st, P.refine(z, ev), texts)
        print(f"\n{b}: stack {score(z.argmax(1)):.4f}, GLM stage {score(st):.4f}, 0.702 tail {score(ref):.4f}", flush=True)
        cands = {}
        for w in (0.10, 0.20, 0.30, 0.50):
            for nm, spec in members.items():
                cands[f"{nm} specialist w {w:.2f}"] = P.spec_apply(z, spec, w, False)
        for a in (0.10, 0.20, 0.30, 0.50):
            cands[f"op8 s1 18-class member a {a:.2f}"] = log_softmax(z + a * s1, axis=1)
        cands["op8 s1 spec w 0.20 + TF-IDF w 0.20"] = P.spec_apply(P.spec_apply(z, members["op8 s1 (LLM, pool only)"],
                                                                                0.20, False),
                                                                   members["TF-IDF (NTL-40)"], 0.20, False)
        for nm, zs in cands.items():
            # an 18-class member also moves lab mass, so it also changes which rows the GLM stage picks
            st2 = X.blend(zs, head, 0.15, 1.0, h) if "member" in nm else st
            p = P.apex(st2, P.refine(zs, ev), texts)
            g = paired_bootstrap(y, p, ref, prompts, n=1000)
            print(f"  {nm:40s} stack {score(zs.argmax(1)) - score(z.argmax(1)):+.4f}  tail {score(p):.4f} "
                  f"{g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}]", flush=True)




def combo(base_path):
    """op8 s1 specialist on top of NTL-40 as the notebook runs it (TF-IDF w 0.20 rescaled to its OOF base's spread),
    op8's w on the stand-in's own scale; gain per prompt fold."""
    train = X.load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.astype(str).to_numpy()
    folds = X.prompt_folds(prompts)
    lab = np.asarray(X.LABELS)
    score = lambda p, m=slice(None): X.balanced_accuracy(train.model[m], lab[p[m]])
    base = np.load(base_path)
    bank = np.load(_ROOT / "data" / "external" / "oofbank" / "oof_predictions.npz", allow_pickle=False)
    cls = bank["classes"].tolist()
    bases = {"stand-in": base["zl"],
             "bank": log_softmax(bank["nested_calibrated"][1][:, [cls.index(l) for l in X.LABELS]], axis=1)}
    s1 = np.load(OUT / "op8" / "s1_on_train.npy", allow_pickle=False).astype(np.float64)
    s1 = log_softmax(s1[:, [LABELS18.index(l) for l in X.LABELS]], axis=1)
    op8, tfidf = as_spec(s1), P.load_spec("ntl40", None, texts, None, folds)
    ll8 = np.load(OUT / "ll8op1" / "ll8op1_s1_on_train.npy", allow_pickle=False).astype(np.float64)
    ll8 = log_softmax(ll8[:, [LABELS18.index(l) for l in X.LABELS]], axis=1)
    print(f"ll8op1 s1 (Llama-3.1-8B, pool v14 only) alone {score(ll8.argmax(1)):.4f}", flush=True)
    ll8s, avg = as_spec(ll8), as_spec((s1 + ll8) / 2)
    names54 = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    n54 = len(X.HIS["features"])
    FB, _ = Y.frame(Y.load(Y.D160, "flash", ids), Y.load(Y.D160 / "s1", "glm53", ids), texts, names54)
    h = X.HIS["entropy_ref"]
    S_ = P.style_features(texts)
    head, ev = np.zeros((len(y), 18)), np.zeros((len(y), 18))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        sc, lr = X.fit(FB[tr][:, :n54], y[tr], X.HIS["C"])
        head[va] = lr.predict_log_proba(sc.transform(FB[va][:, :n54]))
        ev[va] = P.residual_evidence(S_[tr], y[tr], S_[va])
    for b, z in bases.items():
        st = X.blend(z, head, 0.15, 1.0, h)
        z40 = P.spec_apply(z, tfidf, 0.20, True)            # NTL-40 as published
        ref = P.apex(st, P.refine(z40, ev), texts)
        p702 = P.apex(st, P.refine(z, ev), texts)
        print(f"\n{b}: 0.702 tail {score(p702):.4f}, NTL-40 {score(ref):.4f}", flush=True)
        cands = {f"op8 w {w:.2f}": [(op8, w)] for w in (0.10, 0.20, 0.30, 0.40, 0.50, 0.70)}
        cands.update({f"ll8 w {w:.2f}": [(ll8s, w)] for w in (0.20, 0.40)})
        cands.update({f"mean(op8,ll8) w {w:.2f}": [(avg, w)] for w in (0.30, 0.40, 0.50, 0.70)})
        cands.update({f"op8 w 0.40 + ll8 w {w:.2f}": [(op8, 0.40), (ll8s, w)] for w in (0.10, 0.20, 0.30)})
        for nm, steps in cands.items():
            zz = z40
            for sp_, w in steps:
                zz = P.spec_apply(zz, sp_, w, False)
            p = P.apex(st, P.refine(zz, ev), texts)
            g = paired_bootstrap(y, p, ref, prompts, n=1000)
            per = [score(p, folds == k) - score(ref, folds == k) for k in np.unique(folds)]
            print(f"  NTL-40 + {nm:26s}: {score(p):.4f} {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}]  folds "
                  + " ".join(f"{d:+.3f}" for d in per) + f"  rows changed {np.mean(p != ref):.3f}", flush=True)


def q14(base_path):
    """ntl-tpu-q14v16's stage-1 Qwen3-14B-Base LoRA (keep 27, pool v16 223,840 rows match-weighted, 1.46 epochs; pool
    v16/v17 have no rows near train prompts) as the sibling specialist, alone and with op8, on top of NTL-40."""
    train = X.load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.astype(str).to_numpy()
    folds = X.prompt_folds(prompts)
    lab = np.asarray(X.LABELS)
    score = lambda p, m=slice(None): X.balanced_accuracy(train.model[m], lab[p[m]])
    base = np.load(base_path)
    bank = np.load(_ROOT / "data" / "external" / "oofbank" / "oof_predictions.npz", allow_pickle=False)
    cls = bank["classes"].tolist()
    bases = {"stand-in": base["zl"],
             "bank": log_softmax(bank["nested_calibrated"][1][:, [cls.index(l) for l in X.LABELS]], axis=1)}
    load = lambda f: log_softmax(np.load(OUT / f, allow_pickle=False).astype(np.float64)[
        :, [LABELS18.index(l) for l in X.LABELS]], axis=1)
    s1, q = load("op8/s1_on_train.npy"), load("q14op16/q14op16_s1_on_train.npy")
    print(f"q14op16 s1 alone {score(q.argmax(1)):.4f} (summary.json 0.6146), op8 s1 {score(s1.argmax(1)):.4f}, "
          f"argmax agreement {np.mean(q.argmax(1) == s1.argmax(1)):.3f}", flush=True)
    op8, q14s, avg = as_spec(s1), as_spec(q), as_spec((s1 + q) / 2)
    tfidf = P.load_spec("ntl40", None, texts, None, folds)
    for nm, spec in {"q14op16 s1": q14s, "op8 s1": op8, "mean": avg}.items():
        acc = []
        for k, (sib, S) in spec.items():
            idx = np.array([X.LABELS.index(m) for m in sib])
            rows = np.isin(y, idx)
            acc.append(f"{k} {np.mean(idx[S[rows].argmax(1)] == y[rows]):.3f}")
        print(f"{nm}: alone inside the lab: " + ", ".join(acc), flush=True)
    names54 = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    n54 = len(X.HIS["features"])
    FB, _ = Y.frame(Y.load(Y.D160, "flash", ids), Y.load(Y.D160 / "s1", "glm53", ids), texts, names54)
    h = X.HIS["entropy_ref"]
    S_ = P.style_features(texts)
    head, ev = np.zeros((len(y), 18)), np.zeros((len(y), 18))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        sc, lr = X.fit(FB[tr][:, :n54], y[tr], X.HIS["C"])
        head[va] = lr.predict_log_proba(sc.transform(FB[va][:, :n54]))
        ev[va] = P.residual_evidence(S_[tr], y[tr], S_[va])
    for b, z in bases.items():
        st = X.blend(z, head, 0.15, 1.0, h)
        z40 = P.spec_apply(z, tfidf, 0.20, True)
        ref = P.apex(st, P.refine(z40, ev), texts)
        print(f"\n{b}: NTL-40 {score(ref):.4f}", flush=True)
        cands = {"op8 w 0.40": [(op8, 0.40)]}
        cands.update({f"q14 w {w:.2f}": [(q14s, w)] for w in (0.20, 0.30, 0.40, 0.50, 0.70)})
        cands.update({f"mean(op8,q14) w {w:.2f}": [(avg, w)] for w in (0.40, 0.50, 0.70)})
        cands.update({f"op8 w 0.40 + q14 w {w:.2f}": [(op8, 0.40), (q14s, w)] for w in (0.10, 0.20, 0.30)})
        for nm, steps in cands.items():
            zz = z40
            for sp_, w in steps:
                zz = P.spec_apply(zz, sp_, w, False)
            p = P.apex(st, P.refine(zz, ev), texts)
            g = paired_bootstrap(y, p, ref, prompts, n=1000)
            per = [score(p, folds == k) - score(ref, folds == k) for k in np.unique(folds)]
            print(f"  NTL-40 + {nm:26s}: {score(p):.4f} {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}]  folds "
                  + " ".join(f"{d:+.3f}" for d in per) + f"  rows changed {np.mean(p != ref):.3f}", flush=True)


def perlab(base_path):
    """op8 specialist on top of NTL-40: one lab at a time off the w 0.40 default, and the GLM pair as a fourth lab."""
    train = X.load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.astype(str).to_numpy()
    folds = X.prompt_folds(prompts)
    lab = np.asarray(X.LABELS)
    score = lambda p, m=slice(None): X.balanced_accuracy(train.model[m], lab[p[m]])
    base = np.load(base_path)
    bank = np.load(_ROOT / "data" / "external" / "oofbank" / "oof_predictions.npz", allow_pickle=False)
    cls = bank["classes"].tolist()
    bases = {"stand-in": base["zl"],
             "bank": log_softmax(bank["nested_calibrated"][1][:, [cls.index(l) for l in X.LABELS]], axis=1)}
    s1 = log_softmax(np.load(OUT / "op8" / "s1_on_train.npy", allow_pickle=False).astype(np.float64)[
        :, [LABELS18.index(l) for l in X.LABELS]], axis=1)
    op8 = as_spec(s1)
    sib = ["glm-5.3", "glm-5.3-flash"]
    S = log_softmax(s1[:, [X.LABELS.index(m) for m in sib]], axis=1)
    glm = {"glm": (sib, S - S.mean(1, keepdims=True))}
    gi = np.array([X.LABELS.index(m) for m in sib])
    rows = np.isin(y, gi)
    print(f"op8 s1 inside the GLM pair: {np.mean(gi[S[rows].argmax(1)] == y[rows]):.3f}", flush=True)
    tfidf = P.load_spec("ntl40", None, texts, None, folds)
    names54 = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    n54 = len(X.HIS["features"])
    FB, _ = Y.frame(Y.load(Y.D160, "flash", ids), Y.load(Y.D160 / "s1", "glm53", ids), texts, names54)
    h = X.HIS["entropy_ref"]
    S_ = P.style_features(texts)
    head, ev = np.zeros((len(y), 18)), np.zeros((len(y), 18))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        sc, lr = X.fit(FB[tr][:, :n54], y[tr], X.HIS["C"])
        head[va] = lr.predict_log_proba(sc.transform(FB[va][:, :n54]))
        ev[va] = P.residual_evidence(S_[tr], y[tr], S_[va])
    for b, z in bases.items():
        st = X.blend(z, head, 0.15, 1.0, h)
        z40 = P.spec_apply(z, tfidf, 0.20, True)
        ref = P.apex(st, P.refine(z40, ev), texts)
        z8 = P.spec_apply(z40, op8, 0.40, False)
        p8 = P.apex(st, P.refine(z8, ev), texts)
        print(f"\n{b}: NTL-40 {score(ref):.4f}, + op8 w 0.40 {score(p8):.4f}", flush=True)
        cands = {f"{l} w {w:.2f}": (z8, {l: op8[l]}, w - 0.40) for l in P.LABS for w in (0.20, 0.60)}
        cands.update({f"+ glm pair w {w:.2f}": (z8, glm, w) for w in (0.20, 0.40, 0.70)})
        for nm, (zb, sp_, dw) in cands.items():
            # the lab's own weight moves by dw: applying the same centred scores again adds linearly
            p = P.apex(st, P.refine(P.spec_apply(zb, sp_, dw, False), ev), texts)
            g = paired_bootstrap(y, p, p8, prompts, n=1000)
            per = [score(p, folds == k) - score(p8, folds == k) for k in np.unique(folds)]
            print(f"  op8 0.40, {nm:20s}: {score(p):.4f} {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}] vs op8 0.40  folds "
                  + " ".join(f"{d:+.3f}" for d in per), flush=True)


LAB_OF = lambda m: "glm" if m.startswith("glm") else m.split("-")[0].split("3.8")[0]


def lab_shift(L, s, a):
    """Move each lab's mass by a * (the pool-only model's centred lab log-prob); the split inside each lab is kept."""
    labs = sorted({LAB_OF(m) for m in X.LABELS})
    col = np.array([labs.index(LAB_OF(m)) for m in X.LABELS])
    lp = np.stack([np.logaddexp.reduce(s[:, col == j], axis=1) for j in range(len(labs))], 1)
    return log_softmax(L + a * (lp - lp.mean(1, keepdims=True))[:, col], axis=1)


def labmass(base_path):
    """op8 w 0.40 inside the lab, plus op8 / q14 / their mean moving the lab masses (which changes the GLM stage's
    row selection too, so the stage is recomputed)."""
    train = X.load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.astype(str).to_numpy()
    folds = X.prompt_folds(prompts)
    lab = np.asarray(X.LABELS)
    score = lambda p, m=slice(None): X.balanced_accuracy(train.model[m], lab[p[m]])
    base = np.load(base_path)
    bank = np.load(_ROOT / "data" / "external" / "oofbank" / "oof_predictions.npz", allow_pickle=False)
    cls = bank["classes"].tolist()
    bases = {"stand-in": base["zl"],
             "bank": log_softmax(bank["nested_calibrated"][1][:, [cls.index(l) for l in X.LABELS]], axis=1)}
    load = lambda f: log_softmax(np.load(OUT / f, allow_pickle=False).astype(np.float64)[
        :, [LABELS18.index(l) for l in X.LABELS]], axis=1)
    s1, q = load("op8/s1_on_train.npy"), load("q14op16/q14op16_s1_on_train.npy")
    print("labs:", sorted({LAB_OF(m) for m in X.LABELS}), flush=True)
    labs_y = np.array([LAB_OF(m) for m in X.LABELS])[y]
    for nm, s in {"op8": s1, "q14": q, "mean": (s1 + q) / 2}.items():
        print(f"{nm} lab accuracy alone {np.mean(np.array([LAB_OF(m) for m in X.LABELS])[s.argmax(1)] == labs_y):.4f}",
              flush=True)
    op8 = as_spec(s1)
    tfidf = P.load_spec("ntl40", None, texts, None, folds)
    names54 = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    n54 = len(X.HIS["features"])
    FB, _ = Y.frame(Y.load(Y.D160, "flash", ids), Y.load(Y.D160 / "s1", "glm53", ids), texts, names54)
    h = X.HIS["entropy_ref"]
    S_ = P.style_features(texts)
    head, ev = np.zeros((len(y), 18)), np.zeros((len(y), 18))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        sc, lr = X.fit(FB[tr][:, :n54], y[tr], X.HIS["C"])
        head[va] = lr.predict_log_proba(sc.transform(FB[va][:, :n54]))
        ev[va] = P.residual_evidence(S_[tr], y[tr], S_[va])
    for b, z in bases.items():
        st = X.blend(z, head, 0.15, 1.0, h)
        z8 = P.spec_apply(P.spec_apply(z, tfidf, 0.20, True), op8, 0.40, False)
        p8 = P.apex(st, P.refine(z8, ev), texts)
        zl = np.array([LAB_OF(m) for m in X.LABELS])
        print(f"\n{b}: NTL-40 + op8 w 0.40 {score(p8):.4f}, lab accuracy {np.mean(zl[p8] == labs_y):.4f}", flush=True)
        for nm, s in {"op8": s1, "q14": q, "mean": (s1 + q) / 2}.items():
            for a in (0.05, 0.10, 0.20, 0.30):
                zz = lab_shift(z8, s, a)
                st2 = X.blend(lab_shift(z, s, a), head, 0.15, 1.0, h)
                p = P.apex(st2, P.refine(zz, ev), texts)
                g = paired_bootstrap(y, p, p8, prompts, n=1000)
                per = [score(p, folds == k) - score(p8, folds == k) for k in np.unique(folds)]
                print(f"  + {nm:4s} lab mass a {a:.2f}: {score(p):.4f} {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}] "
                      f"lab acc {np.mean(zl[p] == labs_y):.4f}  folds " + " ".join(f"{d:+.3f}" for d in per), flush=True)


def stage2(base_path):
    """The same TPU models after their official-row stage 2 (fold models, out of fold) as the sibling specialist."""
    train = X.load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.astype(str).to_numpy()
    folds = X.prompt_folds(prompts)
    lab = np.asarray(X.LABELS)
    score = lambda p, m=slice(None): X.balanced_accuracy(train.model[m], lab[p[m]])
    base = np.load(base_path)
    bank = np.load(_ROOT / "data" / "external" / "oofbank" / "oof_predictions.npz", allow_pickle=False)
    cls = bank["classes"].tolist()
    bases = {"stand-in": base["zl"],
             "bank": log_softmax(bank["nested_calibrated"][1][:, [cls.index(l) for l in X.LABELS]], axis=1)}
    load = lambda f: log_softmax(np.load(OUT / f, allow_pickle=False).astype(np.float64)[
        :, [LABELS18.index(l) for l in X.LABELS]], axis=1)

    def oof(f):
        d = np.load(OUT / f, allow_pickle=False)
        pos = {i: n for n, i in enumerate(d["ids"].tolist())}
        lb = d["labels"].tolist()
        return log_softmax(d["logits"].astype(np.float64)[[pos[i] for i in ids]][:, [lb.index(l) for l in X.LABELS]],
                           axis=1)
    sp = {"op8 s1": load("op8/s1_on_train.npy"), "op8 stage 2": oof("op8/oof.npz"),
          "q14 s1": load("q14op16/q14op16_s1_on_train.npy"), "q14 stage 2": oof("q14op16/q14op16_oof.npz")}
    for nm, s in sp.items():
        acc = []
        for k, (sib, S) in as_spec(s).items():
            idx = np.array([X.LABELS.index(m) for m in sib])
            rows = np.isin(y, idx)
            acc.append(f"{k} {np.mean(idx[S[rows].argmax(1)] == y[rows]):.3f}")
        print(f"{nm}: alone {score(s.argmax(1)):.4f}, inside the lab: " + ", ".join(acc), flush=True)
    tfidf = P.load_spec("ntl40", None, texts, None, folds)
    names54 = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    n54 = len(X.HIS["features"])
    FB, _ = Y.frame(Y.load(Y.D160, "flash", ids), Y.load(Y.D160 / "s1", "glm53", ids), texts, names54)
    h = X.HIS["entropy_ref"]
    S_ = P.style_features(texts)
    head, ev = np.zeros((len(y), 18)), np.zeros((len(y), 18))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        sc, lr = X.fit(FB[tr][:, :n54], y[tr], X.HIS["C"])
        head[va] = lr.predict_log_proba(sc.transform(FB[va][:, :n54]))
        ev[va] = P.residual_evidence(S_[tr], y[tr], S_[va])
    for b, z in bases.items():
        st = X.blend(z, head, 0.15, 1.0, h)
        z40 = P.spec_apply(z, tfidf, 0.20, True)
        ref = P.apex(st, P.refine(z40, ev), texts)
        print(f"\n{b}: NTL-40 {score(ref):.4f}", flush=True)
        for nm, s in sp.items():
            for w in (0.20, 0.40, 0.60):
                p = P.apex(st, P.refine(P.spec_apply(z40, as_spec(s), w, False), ev), texts)
                g = paired_bootstrap(y, p, ref, prompts, n=1000)
                per = [score(p, folds == k) - score(ref, folds == k) for k in np.unique(folds)]
                print(f"  NTL-40 + {nm:12s} w {w:.2f}: {score(p):.4f} {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}]  folds "
                      + " ".join(f"{d:+.3f}" for d in per), flush=True)


def both(base_path):
    """op8 and q14 together, inside the lab and on the lab masses, against op8 w 0.40 alone (the #111 notebook)."""
    train = X.load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.astype(str).to_numpy()
    folds = X.prompt_folds(prompts)
    lab = np.asarray(X.LABELS)
    score = lambda p, m=slice(None): X.balanced_accuracy(train.model[m], lab[p[m]])
    base = np.load(base_path)
    bank = np.load(_ROOT / "data" / "external" / "oofbank" / "oof_predictions.npz", allow_pickle=False)
    cls = bank["classes"].tolist()
    bases = {"stand-in": base["zl"],
             "bank": log_softmax(bank["nested_calibrated"][1][:, [cls.index(l) for l in X.LABELS]], axis=1)}
    load = lambda f: log_softmax(np.load(OUT / f, allow_pickle=False).astype(np.float64)[
        :, [LABELS18.index(l) for l in X.LABELS]], axis=1)
    s1, q = load("op8/s1_on_train.npy"), load("q14op16/q14op16_s1_on_train.npy")
    op8, q14s, avg = as_spec(s1), as_spec(q), as_spec((s1 + q) / 2)
    tfidf = P.load_spec("ntl40", None, texts, None, folds)
    names54 = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    n54 = len(X.HIS["features"])
    FB, _ = Y.frame(Y.load(Y.D160, "flash", ids), Y.load(Y.D160 / "s1", "glm53", ids), texts, names54)
    h = X.HIS["entropy_ref"]
    S_ = P.style_features(texts)
    head, ev = np.zeros((len(y), 18)), np.zeros((len(y), 18))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        sc, lr = X.fit(FB[tr][:, :n54], y[tr], X.HIS["C"])
        head[va] = lr.predict_log_proba(sc.transform(FB[va][:, :n54]))
        ev[va] = P.residual_evidence(S_[tr], y[tr], S_[va])
    for b, z in bases.items():
        st = X.blend(z, head, 0.15, 1.0, h)
        z40 = P.spec_apply(z, tfidf, 0.20, True)
        ref = P.apex(st, P.refine(z40, ev), texts)
        p8 = P.apex(st, P.refine(P.spec_apply(z40, op8, 0.40, False), ev), texts)
        print(f"\n{b}: NTL-40 {score(ref):.4f}, + op8 w 0.40 {score(p8):.4f}", flush=True)
        cands = {}
        for wq in (0.0, 0.20, 0.30):
            for a in (0.05, 0.10):
                cands[f"op8 0.40 + q14 {wq:.2f}, q14 lab {a:.2f}"] = ([(op8, 0.40), (q14s, wq)], q, a)
        for w in (0.50, 0.70):
            for a in (0.05, 0.10):
                cands[f"mean {w:.2f}, mean lab {a:.2f}"] = ([(avg, w)], (s1 + q) / 2, a)
        for nm, (steps, s, a) in cands.items():
            zz = z40
            for sp_, w in steps:
                zz = P.spec_apply(zz, sp_, w, False)
            if os.environ.get("EXP057_AFTER_STAGE"):   # the shift runs after the GLM stage: its rows keep their call
                p = P.apex(st, P.refine(lab_shift(zz, s, a), ev), texts)
            else:
                p = P.apex(X.blend(lab_shift(z, s, a), head, 0.15, 1.0, h), P.refine(lab_shift(zz, s, a), ev), texts)
            for nm0, r in (("NTL-40", ref), ("op8", p8)):
                g = paired_bootstrap(y, p, r, prompts, n=1000)
                per = [score(p, folds == k) - score(r, folds == k) for k in np.unique(folds)]
                print(f"  {nm:34s} vs {nm0:6s}: {score(p):.4f} {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}]  folds "
                      + " ".join(f"{d:+.3f}" for d in per), flush=True)


def lab_margin(L):
    """Gap between the two largest lab log-masses of logits L (labels in X.LABELS order)."""
    labs = sorted({LAB_OF(m) for m in X.LABELS})
    col = np.array([labs.index(LAB_OF(m)) for m in X.LABELS])
    lp = np.sort(np.stack([np.logaddexp.reduce(log_softmax(L, axis=1)[:, col == j], axis=1)
                           for j in range(len(labs))], 1), axis=1)
    return lp[:, -1] - lp[:, -2]


def labgate(base_path):
    """op8 w 0.40, then q14's lab shift (a 0.05 / 0.10) after the GLM stage on the rows whose lab margin is below m
    only: how many rows the 14B pass must score to keep the gain."""
    train = X.load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.astype(str).to_numpy()
    folds = X.prompt_folds(prompts)
    lab = np.asarray(X.LABELS)
    score = lambda p, m=slice(None): X.balanced_accuracy(train.model[m], lab[p[m]])
    base = np.load(base_path)
    load = lambda f: log_softmax(np.load(OUT / f, allow_pickle=False).astype(np.float64)[
        :, [LABELS18.index(l) for l in X.LABELS]], axis=1)
    s1, q = load("op8/s1_on_train.npy"), load("q14op16/q14op16_s1_on_train.npy")
    tfidf = P.load_spec("ntl40", None, texts, None, folds)
    names54 = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    n54 = len(X.HIS["features"])
    FB, _ = Y.frame(Y.load(Y.D160, "flash", ids), Y.load(Y.D160 / "s1", "glm53", ids), texts, names54)
    h = X.HIS["entropy_ref"]
    S_ = P.style_features(texts)
    head, ev = np.zeros((len(y), 18)), np.zeros((len(y), 18))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        sc, lr = X.fit(FB[tr][:, :n54], y[tr], X.HIS["C"])
        head[va] = lr.predict_log_proba(sc.transform(FB[va][:, :n54]))
        ev[va] = P.residual_evidence(S_[tr], y[tr], S_[va])
    z = base["zl"]
    st = X.blend(z, head, 0.15, 1.0, h)
    z8 = P.spec_apply(P.spec_apply(z, tfidf, 0.20, True), as_spec(s1), 0.40, False)
    p8 = P.apex(st, P.refine(z8, ev), texts)
    mg = lab_margin(z8)
    print(f"stand-in: NTL-40 + op8 {score(p8):.4f}; lab margin quantiles (10/25/50/75/90%) "
          + " ".join(f"{v:.2f}" for v in np.quantile(mg, [.1, .25, .5, .75, .9])), flush=True)
    for a in (0.05, 0.10):
        full = lab_shift(z8, q, a)
        for m in (0.5, 1.0, 2.0, 3.0, 5.0, np.inf):
            gate = mg < m
            zz = np.where(gate[:, None], full, z8)
            p = P.apex(st, P.refine(zz, ev), texts)
            g = paired_bootstrap(y, p, p8, prompts, n=1000)
            per = [score(p, folds == k) - score(p8, folds == k) for k in np.unique(folds)]
            print(f"  q14 lab a {a:.2f}, margin < {m:4.1f} ({gate.mean():.2f} of rows): {score(p):.4f} "
                  f"{g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}]  folds " + " ".join(f"{d:+.3f}" for d in per), flush=True)


def postmortem(base_path):
    """#111 (NTL-40 + op8 w 0.40) scored 0.684 against NTL-40's 0.702. What op8 would cost on the stand-in if its test
    log-probs carried no information (rows shuffled, within the true lab or across all rows), or if the notebook's
    spread rescaling hit its clip (w 0.20 / 1.00)."""
    train = X.load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.astype(str).to_numpy()
    folds = X.prompt_folds(prompts)
    lab = np.asarray(X.LABELS)
    score = lambda p: X.balanced_accuracy(train.model, lab[p])
    z = np.load(base_path)["zl"]
    s1 = np.load(OUT / "op8" / "s1_on_train.npy", allow_pickle=False).astype(np.float64)
    s1 = log_softmax(s1[:, [LABELS18.index(l) for l in X.LABELS]], axis=1)
    tfidf = P.load_spec("ntl40", None, texts, None, folds)
    names54 = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    n54 = len(X.HIS["features"])
    FB, _ = Y.frame(Y.load(Y.D160, "flash", ids), Y.load(Y.D160 / "s1", "glm53", ids), texts, names54)
    h = X.HIS["entropy_ref"]
    S_ = P.style_features(texts)
    head, ev = np.zeros((len(y), 18)), np.zeros((len(y), 18))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        sc, lr = X.fit(FB[tr][:, :n54], y[tr], X.HIS["C"])
        head[va] = lr.predict_log_proba(sc.transform(FB[va][:, :n54]))
        ev[va] = P.residual_evidence(S_[tr], y[tr], S_[va])
    st = X.blend(z, head, 0.15, 1.0, h)
    z40 = P.spec_apply(z, tfidf, 0.20, True)
    ref = P.apex(st, P.refine(z40, ev), texts)
    print(f"NTL-40 tail {score(ref):.4f}", flush=True)
    labs = np.array([LAB_OF(m) for m in X.LABELS])[y]
    cands = {f"op8 w {w:.2f}": (s1, w) for w in (0.20, 0.40, 1.00)}
    for seed in range(3):
        rng = np.random.default_rng(seed)
        cands[f"shuffled rows (all) w 0.40 s{seed}"] = (s1[rng.permutation(len(y))], 0.40)
        perm = np.arange(len(y))
        for g in np.unique(labs):   # shuffled among rows of the same true lab: lab-typical but uninformative
            r = np.flatnonzero(labs == g)
            perm[r] = rng.permutation(r)
        cands[f"shuffled rows (in lab) w 0.40 s{seed}"] = (s1[perm], 0.40)
    for nm, (s, w) in cands.items():
        p = P.apex(st, P.refine(P.spec_apply(z40, as_spec(s), w, False), ev), texts)
        g = paired_bootstrap(y, p, ref, prompts, n=1000)
        print(f"  NTL-40 + {nm:34s}: {score(p):.4f} {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}]  rows changed "
              f"{np.mean(p != ref):.3f}", flush=True)


if __name__ == "__main__":
    {"combo": combo, "q14": q14, "perlab": perlab, "labmass": labmass, "both": both, "labgate": labgate,
     "stage2": stage2, "postmortem": postmortem}.get((sys.argv[2:3] or [""])[0], main)(sys.argv[1])
