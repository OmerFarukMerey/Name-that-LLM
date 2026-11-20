"""exp060: a longer likelihood cut for the 0.704's GLM stage, out of fold.

estopadilla's head (ll_model.json, "ll_trunc/L160") reads the first 160 tokens of each passage under GLM-5.3-Flash and
GLM-5.3. Our train scores cover the full texts (median 130 tokens; 38% are longer than 160, 11% longer than 256), so a
head refitted with his recipe (54 features, StandardScaler + logistic regression C 0.05) on a longer cut can be compared
with his on the same stage rows. At test time the stage scores min(cover, 90k token rows); a 256-token cut at cover 0.15
needs ~100k token rows on ~4,000 passages, within the 130k our c22 builds ran in time.
Stage as exp058's: the stand-in stack, Fable routing, cover 0.15, non-GLM stage results reverted (as apex_postprocess
does); W 1 and 1.25. Reference: his recipe at L160 (out of fold), same W.

    python -I experiments/exp060_glm_ll_cut.py
    python -I experiments/exp060_glm_ll_cut.py export 256   # his recipe on every train row at that cut -> heads/his_L256.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv.append("revert")
import exp058_stage_w_fable as S  # noqa: E402
import numpy as np  # noqa: E402

X = S.X
from ntl.cv import prompt_folds  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402

CUTS = (160, 192, 224, 256, 320, 512)


def main():
    train = load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    sf, _ = X.load_scores("flash", ids)
    sg, _ = X.load_scores("glm53", ids)
    names = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    folds = prompt_folds(prompts)
    nlm = np.load(X.E.NLM_OOF)
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    members["svc4"] = np.load(X.E.OUT / "svc4_oof.npy").astype(np.float64)
    zl = np.log(np.clip(X.E.sinkhorn(X.E.blend(members, X.E.CORE)), 1e-12, None))
    score = lambda p: balanced_accuracy(train.model, np.asarray(X.LABELS)[p])
    h = X.HIS["entropy_ref"]
    n = np.array([len(s) for s in sf])
    print(f"stand-in stack {score(zl.argmax(1)):.4f}", flush=True)

    heads = {}
    for cut in CUTS:
        Xc, cols = X.design(X.features(sf, texts, cut), X.features(sg, texts, cut), names, X.HIS["diff_features"])
        assert cols == X.HIS["features"]
        heads[cut] = X.oof_head(Xc, y, folds, X.HIS["C"])
        alone = (heads[cut].argmax(1) == y)
        print(f"L{cut}: head alone acc {alone.mean():.4f} (texts > 160 tokens: {alone[n > 160].mean():.4f})", flush=True)
    mass = np.exp(zl[:, X.GLM]).sum(1)
    pri = mass + 0.35 * np.minimum(np.exp(zl[:, S.FABLE]), mass)
    sel = np.argsort(-pri, kind="stable")[:int(0.15 * len(zl))]
    print(f"stage rows: {len(sel)}, longer than 160 tokens {np.mean(n[sel] > 160):.2f}, "
          f"mean tokens at cut 160 / 256: {np.minimum(n[sel], 160).mean():.0f} / {np.minimum(n[sel], 256).mean():.0f}")
    for W in (1.0, 1.25):
        ref = S.blend(zl, heads[160], 0.15, W, h, True)
        print(f"\nW {W}: L160 {score(ref):.4f}", flush=True)
        for cut in CUTS[1:]:
            p = S.blend(zl, heads[cut], 0.15, W, h, True)
            g = paired_bootstrap(y, p, ref, prompts, n=1000)
            print(f"  L{cut:<4d} {score(p):.4f}  vs L160: {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}]  "
                  f"({int((p != ref).sum())} rows differ)", flush=True)


def export(cut):
    """his recipe (his features, C 0.05, W 1) fitted on every train row at `cut` tokens; the stage then cuts the test
    there too (v1_stage.py asserts --max-text-tokens == the head's max_text_tokens)."""
    import json
    train = load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(X.LABELS)}).to_numpy()
    sf, _ = X.load_scores("flash", ids)
    sg, _ = X.load_scores("glm53", ids)
    names = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    X160, _ = X.design(X.features(sf, texts, 160), X.features(sg, texts, 160), names, X.HIS["diff_features"])
    # the recipe on our train scores gives his head at 160 up to the GLM runs' batch noise (exp050: coef corr 0.994)
    sc, lr = X.fit(X160, y, X.HIS["C"])
    corr = np.corrcoef(lr.coef_.ravel(), np.ravel(X.HIS["coef"]))[0, 1]
    print(f"L160 refit vs ll_model.json: coef corr {corr:.4f}")
    assert corr > 0.99
    Xc, cols = X.design(X.features(sf, texts, cut), X.features(sg, texts, cut), names, X.HIS["diff_features"])
    assert cols == X.HIS["features"]
    sc, lr = X.fit(Xc, y, X.HIS["C"])
    assert list(lr.classes_) == list(range(18))
    model = dict(X.HIS, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_.tolist(),
                 intercept=lr.intercept_.tolist(), max_text_tokens=cut, train_rows=len(y),
                 source=f"Name-that-LLM experiments/exp060_glm_ll_cut.py: estopadilla's recipe at L{cut}")
    out = X.OUT / "heads" / f"his_L{cut}.json"
    out.write_text(json.dumps(model), encoding="utf-8")
    print(f"exported {out.name}: {len(cols)} features, max_text_tokens {cut}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["export"]:
        export(int(sys.argv[2]))
    else:
        main()
