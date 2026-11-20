"""exp059: more typographic rules for the 0.704's last step (apex_postprocess), out of fold.

The public 0.702 / 0.704 end with three soft rules: if a passage contains a marker and its current label is one the
marker rules out, the ruled-out labels' probabilities are multiplied by `penalty` (0.15) and the argmax retaken:
    ' — '  rules out every label but claude-opus-4-6, claude-opus-5, glm-5.3, glm-5.3-flash
    '’'    rules out claude-*, glm-*
    '—'    rules out gemini-*
Marker rates per label on Dhruv's 120k pool (nlm_pool_v5, no train prompts) suggest:
    '—' is used by gemini-3.7-flash in 18% of its pool passages (the other Gemini labels: 0), so ruling it out may
        cost gemini-3.7 recall                                   -> R3' = R3 without gemini-3.7-flash
    '“' / '”' never by claude-*, glm-*, gemini-3.1 / 3.6 / 3.7    -> R4
    '"' (straight) never by gpt-* (<= 0.1%) or qwen3.8-max (0.3%) -> R5
    "'" (straight) is ~1% in the pool's GPT text but 5-11% in train's gpt-5.5 / terra / astra: not used.
The pool is independent of train, so train out of fold is a fair test. The stage is the 0.704's (his head, Fable
routing, cover 0.15, W 1 and W 1.25, non-GLM stage results reverted); the rules use the stack's probabilities, and the
post-stage ones for GLM winners, as the 0.704 does. The sibling residual is left out (both arms alike).

    python -I experiments/exp059_typography_rules.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv.append("revert")   # exp058's blend reverts non-GLM stage results, as apex_postprocess does
import exp058_stage_w_fable as S  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import softmax  # noqa: E402

X = S.X
from ntl.cv import prompt_folds  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402

L = X.LABELS
ALLOWED = {"claude-opus-4-6", "claude-opus-5", "glm-5.3", "glm-5.3-flash"}
mask = lambda f: np.array([f(x) for x in L])
R1 = (" — ", mask(lambda x: x not in ALLOWED))
R2 = ("’", mask(lambda x: x.startswith(("claude-", "glm-"))))
R3 = ("—", mask(lambda x: x.startswith("gemini-")))
R3b = ("—", mask(lambda x: x.startswith("gemini-") and x != "gemini-3.7-flash"))
R4 = (("“", "”"), mask(lambda x: x.startswith(("claude-", "glm-")) or x in ("gemini-3.1-pro-preview", "gemini-3.6-flash",
                                                                           "gemini-3.7-flash")))
R5 = ('"', mask(lambda x: x.startswith("gpt-") or x == "qwen3.8-max"))
SETS = {"none": [], "public": [R1, R2, R3], "public, R3 w/o 3.7": [R1, R2, R3b], "public + R4": [R1, R2, R3, R4],
        "public + R5": [R1, R2, R3, R5], "R3 w/o 3.7 + R4 + R5": [R1, R2, R3b, R4, R5]}


def apply_rules(pred, probs, texts, rules, penalty):
    out, probs = pred.copy(), probs.copy()
    for i, t in enumerate(texts):
        for marker, forbidden in rules:
            ms = marker if isinstance(marker, tuple) else (marker,)
            if any(m in t for m in ms) and forbidden[out[i]]:
                probs[i, forbidden] *= penalty
                if probs[i].sum() > 0:
                    probs[i] /= probs[i].sum()
                    out[i] = probs[i].argmax()
    return out


def main():
    train = load_train()
    ids, texts = train.id.astype(str).tolist(), train.text.fillna("").astype(str).tolist()
    y = train.model.map({l: i for i, l in enumerate(L)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    for m in ("—", '"', "“"):
        has = np.array([m in t for t in texts])
        print(f"{m!r} in train: " + " ".join(f"{l}:{has[y == j].mean():.2f}" for j, l in enumerate(L)
                                             if l.startswith(("gemini", "gpt", "qwen"))))
    sf, _ = X.load_scores("flash", ids)
    sg, _ = X.load_scores("glm53", ids)
    names = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    X160, _ = X.design(X.features(sf, texts, 160), X.features(sg, texts, 160), names, X.HIS["diff_features"])
    del sf, sg
    folds = prompt_folds(prompts)
    nlm = np.load(X.E.NLM_OOF)
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    members["svc4"] = np.load(X.E.OUT / "svc4_oof.npy").astype(np.float64)
    zl = np.log(np.clip(X.E.sinkhorn(X.E.blend(members, X.E.CORE)), 1e-12, None))
    ll = X.oof_head(X160, y, folds, X.HIS["C"])
    h = X.HIS["entropy_ref"]
    score = lambda p: balanced_accuracy(train.model, np.asarray(L)[p])
    print(f"stand-in stack {score(zl.argmax(1)):.4f}")

    for W in (1.0, 1.25):
        pred = S.blend(zl, ll, 0.15, W, h, True)       # routing, cover 0.15, non-GLM stage results reverted
        post = zl.copy()                               # the post-stage logits for the GLM winners' probabilities
        mass = np.exp(zl[:, X.GLM]).sum(1)
        pri = mass + 0.35 * np.minimum(np.exp(zl[:, S.FABLE]), mass)
        sel = np.argsort(-pri, kind="stable")[:int(0.15 * len(zl))]
        post[sel] += W / X.match_temperature(zl, h) * ll[sel]
        probs = softmax(zl, 1)
        glm_win = np.isin(pred, X.GLM)
        probs[glm_win] = softmax(post[glm_win], 1)
        print(f"\nW {W}: after the stage {score(pred):.4f}")
        for pen in (0.15, 0.05):
            ref = apply_rules(pred, probs, texts, SETS["public"], pen)
            for name, rules in SETS.items():
                p = apply_rules(pred, probs, texts, rules, pen)
                g = paired_bootstrap(y, p, ref, prompts, n=1000)
                changed = int((p != ref).sum())
                print(f"  penalty {pen:<4g} {name:24s} {score(p):.4f}  vs public: {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}]"
                      f"  ({changed} rows differ)", flush=True)


if __name__ == "__main__":
    main()
