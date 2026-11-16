"""exp053: more training rows for the GLM stage's likelihood head. exp051's learning curve: the trees of the gbmpair
head gain ~+0.002 balanced accuracy per extra 20% of train prompts and have not flattened (0.6715 / 0.6739 / 0.6760 /
0.6780 at 40 / 60 / 80 / 100% of each training fold, shuffled GLM-5.3 view), the logistic half is flat. The open pool
(crtproduction/name-that-llm-open-pool, Kaggle version 17 = pool release v15, 226,191 passages, no train-prompt rows)
has thousands of passages for the 8 labels that fill 97% of the stage's rows (top 25% by GLM mass, out of fold):
glm-5.3, glm-5.3-flash, claude-fable-5, claude-opus-4-8, claude-sonnet-5, claude-opus-5, kimi-k3, claude-opus-4-6.

select: 250 passages per label, one per prompt, proxy routes after the pool's ruling left out, drawn in proportion to
        the pool's typography match_weight (seed 0) -> kaggle/glmll/pool_rows_v17.csv (id, model, prompt_canon); the
        kernels ntl-glmllpool-{flash,glm53} (build_glmll.py pool) score them like the train runs (cut 160).
oof:    out of fold on the official rows (ntl.cv prompt folds; the pool rows join every training fold, weighted so
        that each label's total weight stays equal), scored on the shuffled GLM-5.3 view, cover 0.20, against gbmpair.

    python -I experiments/exp053_glm_head_pool.py select <ntl_open_pool.parquet>
    python -I experiments/exp053_glm_head_pool.py oof <ntl_open_pool.parquet>
    python -I experiments/exp053_glm_head_pool.py export <ntl_open_pool.parquet> <name> <alpha> [lr]
                                            # a head in gbmpair's format, fitted on every official row + the pool rows
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "kaggle" / "glmll")]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABELS8 = ["glm-5.3", "glm-5.3-flash", "claude-fable-5", "claude-opus-4-8", "claude-sonnet-5", "claude-opus-5",
           "kimi-k3", "claude-opus-4-6"]
PER_LABEL = 250
ROWS = _ROOT / "kaggle" / "glmll" / "pool_rows_v17.csv"
OUT = Path(os.environ.get("EXP053_OUT", _ROOT / "outputs" / "exp053_glm_head_pool"))


def select(pool_path):
    p = pd.read_parquet(pool_path)
    q = p[p.model.isin(LABELS8) & ~p.flag_kb_proxy_after_ruling & ~p.flag_near_train_prompt & ~p.flag_train_prompt_regen
          & p.match_weight.notna() & (p.match_weight > 0)]
    rng = np.random.default_rng(0)
    out = []
    for lab in LABELS8:
        g = q[q.model == lab].sample(frac=1.0, random_state=0).drop_duplicates("prompt_canon")
        w = g.match_weight.to_numpy(np.float64)
        pick = rng.choice(len(g), PER_LABEL, replace=False, p=w / w.sum())
        out.append(g.iloc[np.sort(pick)][["id", "model", "prompt_canon"]])
        print(f"{lab:16s} candidates {len(g):6d} picked {PER_LABEL}", flush=True)
    rows = pd.concat(out).sample(frac=1.0, random_state=0)   # labels mixed, so decode neighbours are mixed as on the test
    rows.to_csv(ROWS, index=False)
    print(f"wrote {ROWS}: {len(rows)} rows, {rows.prompt_canon.nunique()} prompts")


def oof(pool_path):
    import exp051_glm_cut160 as Y
    from ntl.oofbank import paired_bootstrap
    X = Y.X
    train = X.load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.astype(str).to_numpy()
    folds = X.prompt_folds(prompts)
    names = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    n54 = len(X.HIS["features"])
    rows = pd.read_csv(ROWS)
    pool = pd.read_parquet(pool_path, columns=["id", "text"]).set_index("id").loc[rows.id]
    pid, ptexts = rows.id.astype(str).tolist(), pool.text.fillna("").astype(str).tolist()
    py = rows.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    FP, _ = Y.frame(Y.load(OUT, "flash", pid), Y.load(OUT, "glm53", pid), ptexts, names)
    F512, _ = Y.frame(Y.load(Y.D512, "flash", ids), Y.load(Y.D512, "glm53", ids), texts, names)
    FB, _ = Y.frame(Y.load(Y.D160, "flash", ids), Y.load(Y.D160 / "s1", "glm53", ids), texts, names)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fit_predict(Xtr, ytr, wtr, off, Xva, lr_pool):
        if lr_pool:
            sc = StandardScaler().fit(Xtr[:, :n54])
            lr = LogisticRegression(C=X.HIS["C"], max_iter=5000).fit(sc.transform(Xtr[:, :n54]), ytr, sample_weight=wtr)
        else:   # his recipe on the official rows only
            sc, lr = X.fit(Xtr[off][:, :n54], ytr[off], X.HIS["C"])
        ll = lr.predict_log_proba(sc.transform(Xva[:, :n54]))
        gs = StandardScaler().fit(Xtr)
        clf = HistGradientBoostingClassifier(**X.GBM).fit(gs.transform(Xtr), ytr, sample_weight=wtr)
        return X.mix(ll, np.log(np.clip(clf.predict_proba(gs.transform(Xva)), 1e-12, None)))

    base = np.load(Path(os.environ.get("EXP053_BASE", "base.npz")))
    zl = base["zl"]
    h = X.HIS["entropy_ref"]
    lab = np.asarray(X.LABELS)
    score = lambda p: X.balanced_accuracy(train.model, lab[p])
    res = {}
    for name, alpha, lr_pool in (("gbmpair", 0.0, False), ("pool a0.5 trees", 0.5, False), ("pool a1 trees", 1.0, False),
                                 ("pool a0.5 trees+lr", 0.5, True), ("pool a1 trees+lr", 1.0, True)):
        ll = {v: np.zeros((len(y), 18)) for v in ("clean", "shuffled")}
        for k in np.unique(folds):
            tr, va = folds != k, folds == k
            if alpha == 0:
                Xtr, ytr, wtr, off = F512[tr], y[tr], np.ones(tr.sum()), np.ones(tr.sum(), bool)
            else:
                w_off, wp = pool_weights(y[tr], py, alpha)
                Xtr, ytr = np.vstack([F512[tr], FP]), np.concatenate([y[tr], py])
                wtr = np.concatenate([w_off, wp])
                off = np.r_[np.ones(tr.sum(), bool), np.zeros(len(py), bool)]
            for v, Fe in (("clean", F512), ("shuffled", FB)):
                ll[v][va] = fit_predict(Xtr, ytr, wtr, off, Fe[va], lr_pool)
        res[name] = ll
    for v in ("clean", "shuffled"):
        ref = X.blend(zl, res["gbmpair"][v], 0.20, 1.0, h)
        print(f"{v}: gbmpair {score(ref):.4f}", flush=True)
        for name in res:
            if name == "gbmpair":
                continue
            p = X.blend(zl, res[name][v], 0.20, 1.0, h)
            g = paired_bootstrap(y, p, ref, prompts, n=1000)
            print(f"  {name:20s} {score(p):.4f}  vs gbmpair {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}]", flush=True)


def pool_weights(y_off, py, alpha):
    """Pool rows at alpha x (official rows per label / pool rows per label); the labels without pool rows get their
    official rows' weight raised by alpha, so every label's total weight stays equal."""
    n_off = np.bincount(y_off, minlength=18).astype(float)
    has = np.isin(np.arange(18), np.unique(py))
    wp = alpha * n_off[py] / np.bincount(py, minlength=18)[py]
    return 1.0 + np.where(has, 0.0, alpha)[y_off], wp


def export(pool_path, name, alpha, lr_flag=""):
    """gbmpair's recipe (his scaler + LR on the 54 columns, 50/50 with depth-2 trees on the 54 + 12 pair columns),
    the trees (and with `lr` the LR too) fitted on all official rows + the pool rows at weight alpha."""
    import json
    import exp051_glm_cut160 as Y
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from extra_features import gbm_export, gbm_logp
    X = Y.X
    alpha = float(alpha)
    train = X.load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    names = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    n54 = len(X.HIS["features"])
    rows = pd.read_csv(ROWS)
    pool = pd.read_parquet(pool_path, columns=["id", "text"]).set_index("id").loc[rows.id]
    pid, ptexts = rows.id.astype(str).tolist(), pool.text.fillna("").astype(str).tolist()
    py = rows.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    FP, _ = Y.frame(Y.load(OUT, "flash", pid), Y.load(OUT, "glm53", pid), ptexts, names)
    F, cols = Y.frame(Y.load(Y.D512, "flash", ids), Y.load(Y.D512, "glm53", ids), texts, names)
    assert cols[:n54] == X.HIS["features"]
    w_off, wp = pool_weights(y, py, alpha)
    A, yA, wA = np.vstack([F, FP]), np.concatenate([y, py]), np.concatenate([w_off, wp])
    if lr_flag == "lr":
        sc = StandardScaler().fit(A[:, :n54])
        lr = LogisticRegression(C=X.HIS["C"], max_iter=5000).fit(sc.transform(A[:, :n54]), yA, sample_weight=wA)
    else:
        sc, lr = X.fit(F[:, :n54], y, X.HIS["C"])
    assert list(lr.classes_) == list(range(18))
    gs = StandardScaler().fit(A)
    clf = HistGradientBoostingClassifier(**X.GBM).fit(gs.transform(A), yA, sample_weight=wA)
    g = gbm_export(clf)
    assert np.abs(gbm_logp(g, gs.transform(A)) - np.log(clf.predict_proba(gs.transform(A)))).max() < 1e-5
    model = dict(X.HIS, features=cols[:n54], mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_.tolist(),
                 intercept=lr.intercept_.tolist(), C=X.HIS["C"], W=1.0, train_rows=len(y) + len(py),
                 source=f"Name-that-LLM experiments/exp053_glm_head_pool.py: {name} (gbmpair recipe + {len(py)} open-pool "
                        f"rows at alpha {alpha:g}, {'trees and LR' if lr_flag == 'lr' else 'trees only'})",
                 extra={"pair": True}, fallback=X.HIS)
    model["gbm"] = dict(g, features=cols, mean=gs.mean_.tolist(), scale=gs.scale_.tolist(), mix=X.GBM_MIX, params=X.GBM)
    dest = _ROOT / "outputs" / "exp050_glm_ll_head" / "heads" / f"{name}.json"
    dest.write_text(json.dumps(model), encoding="utf-8")
    print(f"exported {dest} ({dest.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    {"select": select, "oof": oof, "export": export}[sys.argv[1]](*sys.argv[2:])
