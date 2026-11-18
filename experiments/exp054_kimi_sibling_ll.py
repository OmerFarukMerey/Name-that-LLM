"""exp054: a sibling likelihood for kimi-k3. Every train passage scored under the open Kimi-Linear-48B-A3B-Instruct
(kernel ntl-kimill, kaggle/glmll/build_kimill.py: Q3_K_M GGUF on llama.cpp, empty-user assistant turn, first 160
tokens). Kimi-K3's text model is the same kimi_linear architecture with the same vocabulary, as GLM-5.3-Flash is to
GLM-5.3. Same per-text features and protocol as exp052, with contrasts against Qwen3-4B-Base (ntl-refll),
GLM-5.3-Flash and Qwen3.8-27B (exp052).

Out of fold (ntl.cv prompt folds) on exp050's stand-in stack after the GLM stage (gbm head, cover 0.20): heads on
these columns, blended like v1_stage.py (argmax(zl + W / T' ll)) on the rows the stack gives the most kimi-k3 mass
(or kimi + gpt-5.5 mass), outside the GLM stage's rows. A partial ntl-kimill run (time cap) is evaluated on the
texts it scored.

    python -I experiments/exp054_kimi_sibling_ll.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "kaggle" / "glmll")]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import exp051_glm_cut160 as Y  # noqa: E402
import exp052_qwen_sibling_ll as Q52  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402

X = Y.X
OUT = Path(os.environ.get("EXP054_OUT", _ROOT / "outputs" / "exp054_kimi_sibling_ll"))
DIFF = Q52.DIFF


def load_scores(path=OUT / "kimi_scores.npz"):
    z = np.load(path, allow_pickle=False)
    off, sc = z["offsets"], z["scores"]
    assert np.isfinite(sc).all()
    return {k: sc[off[i]:off[i + 1]] for i, k in enumerate(z["ids"].tolist())}


def main(kimi=None, tag="kimi", extra=None):
    """kimi: id -> per-token scores of the model under test (default: ntl-kimill's); extra: name -> other such
    scores, added as one more feature set (rows scored by all of them)."""
    train = X.load_train()
    kimi = load_scores() if kimi is None else kimi
    extra = extra or {}
    keep = train.id.astype(str).isin(kimi).to_numpy().copy()
    for v in extra.values():
        keep &= train.id.astype(str).isin(v).to_numpy()
    print(f"{tag} scored {keep.sum()} / {len(keep)} train texts", flush=True)
    base = np.load(Path(os.environ.get("EXP054_BASE", "base.npz")))
    zl_all, ll_all = base["zl"], base["ll"]
    train = train[keep].reset_index(drop=True)
    zl, ll = zl_all[keep], ll_all[keep]
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    assert (base["y"][keep] == y).all()
    prompts = train.prompt_id.astype(str).to_numpy()
    folds = X.prompt_folds(prompts)
    names = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    n54 = len(X.HIS["features"])
    lab = np.asarray(X.LABELS)
    score = lambda p: X.balanced_accuracy(train.model, lab[p])
    K = pd.DataFrame(X.features([kimi[i] for i in ids], texts, 160))[names]
    R = pd.DataFrame(X.features(X.load_ref("q4b", ids), texts, 160))[names]
    fl = pd.DataFrame(X.features(Y.load(Y.D512, "flash", ids), texts, 160))[names]
    q38 = pd.DataFrame(X.features(Q52.load_q38(ids), texts, 160))[names]
    F512, _ = Y.frame(Y.load(Y.D512, "flash", ids), Y.load(Y.D512, "glm53", ids), texts, names)
    k = K.to_numpy(np.float64)
    dq4 = (K[DIFF] - R[DIFF]).to_numpy(np.float64)
    dfl = (K[DIFF] - fl[DIFF]).to_numpy(np.float64)
    dq38 = (K[DIFF] - q38[DIFF]).to_numpy(np.float64)
    ki, g55 = list(lab).index("kimi-k3"), list(lab).index("gpt-5.5")
    print(f"{tag} lp_mean by class: " + ", ".join(f"{l[:14]} {K.lp_mean[y == c].mean():.3f}"
                                                         for c, l in enumerate(lab)), flush=True)
    pair = np.isin(y, [ki, g55])
    gpt = [i for i, l in enumerate(lab) if l.startswith("gpt")]
    for nm, col in (("lp_mean", K.lp_mean), ("lp_mean - q4b", K.lp_mean - R.lp_mean),
                    ("lp_mean - flash", K.lp_mean - fl.lp_mean), ("lp_mean - q38", K.lp_mean - q38.lp_mean)):
        a = col.to_numpy()
        print(f"AUC kimi-k3 vs rest from {nm}: {roc_auc_score(y == ki, a):.3f}; vs gpt-5.5 only: "
              f"{roc_auc_score(y[pair] == ki, a[pair]):.3f}; OpenAI vs rest: {roc_auc_score(np.isin(y, gpt), a):.3f}",
              flush=True)
    h = X.HIS["entropy_ref"]
    p_stage = X.blend(zl, ll, 0.20, 1.0, h)
    print(f"stack {score(zl.argmax(1)):.4f}, + GLM stage {score(p_stage):.4f}; kimi recall after the stage "
          f"{np.mean(p_stage[y == ki] == ki):.3f}", flush=True)
    sets = {"kimi": k, "kimi+d(q4b)": np.hstack([k, dq4]), "kimi+d(q4b)+d(flash)": np.hstack([k, dq4, dfl]),
            "kimi+d(q4b,flash,q38)": np.hstack([k, dq4, dfl, dq38]),
            "kimi+q4b+glm": np.hstack([k, R.to_numpy(np.float64), F512[:, :n54]])}
    sets = {nm.replace("kimi", tag): v for nm, v in sets.items()}
    if extra:
        E = [pd.DataFrame(X.features([v[i] for i in ids], texts, 160))[names] for v in extra.values()]
        sets[f"{tag}+d(q4b)+" + "+".join(extra)] = np.hstack([k, dq4] + [e.to_numpy(np.float64) for e in E]
                                                            + [(K[DIFF] - e[DIFF]).to_numpy(np.float64) for e in E])
    heads = {}
    for nm, A in sets.items():
        heads[f"{nm} lr"] = X.oof_head(A, y, folds, 0.05)
        heads[f"{nm} gbm"] = X.mix(heads[f"{nm} lr"], X.oof_gbm(A, y, folds))
    for nm, v in heads.items():
        pk = v.argmax(1)
        print(f"{nm:28s} head alone {score(pk):.4f}, kimi recall {np.mean(pk[y == ki] == ki):.3f}, gpt-5.5 recall "
              f"{np.mean(pk[y == g55] == g55):.3f}", flush=True)
    P = np.exp(zl)
    stage_rows = np.zeros(len(y), bool)
    stage_rows[np.argsort(-P[:, X.GLM].sum(1), kind="stable")[:int(0.20 * len(y))]] = True
    w0 = 1.0 / X.match_temperature(zl, h)
    rules = {"kimi mass": P[:, ki], "kimi+gpt mass": P[:, [ki] + gpt].sum(1), "gpt mass": P[:, gpt].sum(1)}
    for nm, v in heads.items():
        for rn, m in rules.items():
            for cover in (0.10, 0.20, 0.40):
                for W in (0.5, 1.0, 1.5):
                    cand = np.where(~stage_rows)[0]
                    sel = cand[np.argsort(-m[cand], kind="stable")[:int(cover * len(y))]]
                    p = p_stage.copy()
                    p[sel] = (zl[sel] + W * w0 * v[sel]).argmax(1)
                    g = paired_bootstrap(y, p, p_stage, prompts, n=500)
                    print(f"{nm:28s} {rn:13s} cover {cover:.2f} W {W:.1f}: {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}] "
                          f"kimi recall {np.mean(p[y == ki] == ki):.3f}, gpt-5.5 {np.mean(p[y == g55] == g55):.3f}",
                          flush=True)


if __name__ == "__main__":
    main()
