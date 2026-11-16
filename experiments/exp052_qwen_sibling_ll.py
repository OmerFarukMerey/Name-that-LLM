"""exp052: a sibling likelihood for qwen3.8-max. Every train passage scored under the open Qwen3.8-27B chat model
(kernel ntl-q38ll, kaggle/glmll/build_q38ll.py: NF4, no-think assistant turn, first 160 tokens) gives the same 24
per-text features as the GLM stage; contrasts against a neutral reference (Qwen3-4B-Base, ntl-refll) and GLM-5.3-Flash
play the role GLM-5.3 vs Flash plays in the GLM stage.

Out of fold (ntl.cv prompt folds) on exp050's stand-in stack after the GLM stage (gbm head, cover 0.20): heads on
these columns, blended like v1_stage.py (argmax(zl + W / T' ll)) on the rows the stack gives the most qwen3.8-max
mass (or GPT + Kimi + Qwen mass), outside the GLM stage's rows.

    python -I experiments/exp052_qwen_sibling_ll.py [oof]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "kaggle" / "glmll")]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp051_glm_cut160 as Y  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402

X = Y.X
OUT = Path(os.environ.get("EXP052_OUT", _ROOT / "outputs" / "exp052_qwen_sibling_ll"))
DIFF = ["lp_mean", "lp_q50", "ent_mean", "fdg", "z_mean", "logrank_mean", "top1", "bpc"]


def load_q38(ids):
    z = np.load(OUT / "q38_scores.npz", allow_pickle=False)
    pos = {k: i for i, k in enumerate(z["ids"].tolist())}
    off, sc = z["offsets"], z["scores"]
    assert np.isfinite(sc).all()
    return [sc[off[pos[i]]:off[pos[i] + 1]] for i in ids]


def main():
    train = X.load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.astype(str).to_numpy()
    folds = X.prompt_folds(prompts)
    names = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    n54 = len(X.HIS["features"])
    lab = np.asarray(X.LABELS)
    score = lambda p: X.balanced_accuracy(train.model, lab[p])
    Q = pd.DataFrame(X.features(load_q38(ids), texts, 160))[names]
    R = pd.DataFrame(X.features(X.load_ref("q4b", ids), texts, 160))[names]
    fl = pd.DataFrame(X.features(Y.load(Y.D512, "flash", ids), texts, 160))[names]
    F512, _ = Y.frame(Y.load(Y.D512, "flash", ids), Y.load(Y.D512, "glm53", ids), texts, names)
    q, r = Q.to_numpy(np.float64), R.to_numpy(np.float64)
    dq4 = (Q[DIFF] - R[DIFF]).to_numpy(np.float64)
    dfl = (Q[DIFF] - fl[DIFF]).to_numpy(np.float64)
    qi = list(lab).index("qwen3.8-max")
    print(f"q38 lp_mean by class (mean over texts): "
          + ", ".join(f"{l[:12]} {Q.lp_mean[y == k].mean():.3f}" for k, l in enumerate(lab)), flush=True)
    for nm, col in (("lp_mean", Q.lp_mean), ("lp_mean - q4b", Q.lp_mean - R.lp_mean), ("lp_mean - flash", Q.lp_mean - fl.lp_mean)):
        a = col.to_numpy()
        from sklearn.metrics import roc_auc_score
        print(f"AUC qwen3.8-max vs rest from {nm}: {roc_auc_score(y == qi, a):.3f}", flush=True)
    base = np.load(Path(os.environ.get("EXP052_BASE", "base.npz")))
    zl, ll = base["zl"], base["ll"]
    assert (base["y"] == y).all()
    h = X.HIS["entropy_ref"]
    p_stage = X.blend(zl, ll, 0.20, 1.0, h)
    print(f"stack {score(zl.argmax(1)):.4f}, + GLM stage {score(p_stage):.4f}", flush=True)
    sets = {"q38": q, "q38+d(q4b)": np.hstack([q, dq4]), "q38+d(q4b)+d(flash)": np.hstack([q, dq4, dfl]),
            "q38+q4b+glm": np.hstack([q, r, F512[:, :n54]])}
    heads = {}
    for nm, A in sets.items():
        heads[f"{nm} lr"] = X.oof_head(A, y, folds, 0.05)
        heads[f"{nm} gbm"] = X.mix(heads[f"{nm} lr"], X.oof_gbm(A, y, folds))
    for k, v in heads.items():
        pk = v.argmax(1)
        print(f"{k:26s} head alone {score(pk):.4f}, qwen recall {np.mean(pk[y == qi] == qi):.3f}", flush=True)
    P = np.exp(zl)
    fam = lambda f: [i for i, l in enumerate(lab) if l.startswith(f)]
    stage_rows = np.zeros(len(y), bool)
    stage_rows[np.argsort(-P[:, X.GLM].sum(1), kind="stable")[:int(0.20 * len(y))]] = True
    w0 = 1.0 / X.match_temperature(zl, h)
    rules = {"qwen mass": P[:, qi], "gpt+kimi+qwen mass": P[:, fam("gpt") + fam("kimi") + fam("qwen")].sum(1)}
    for k, v in heads.items():
        for rn, m in rules.items():
            for cover in (0.10, 0.20, 0.40):
                for W in (0.5, 1.0, 1.5):
                    cand = np.where(~stage_rows)[0]
                    sel = cand[np.argsort(-m[cand], kind="stable")[:int(cover * len(y))]]
                    p = p_stage.copy()
                    p[sel] = (zl[sel] + W * w0 * v[sel]).argmax(1)
                    g = paired_bootstrap(y, p, p_stage, prompts, n=500)
                    print(f"{k:26s} {rn:18s} cover {cover:.2f} W {W:.1f}: {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}] "
                          f"qwen recall {np.mean(p[y == qi] == qi):.3f}", flush=True)


if __name__ == "__main__":
    main()
