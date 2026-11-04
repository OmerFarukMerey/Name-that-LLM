"""exp037: the other public 0.666 is synthreaper's Sovereign Gold Apex (alexchilton's verbatim run): One Swap + Sinkhorn
0.30, typography x1.35, atlas_margin 0.32 and two steps after the typography step, Roman Tamrazov's sibling-style
residual (V40A: per family a stylometric logistic regression mixed into the sibling probabilities of low-margin rows,
family mass kept, max_alpha 0.10 x its grouped-OOF reliability) and mercurydust's V34.1 pairwise verifier (a same-model
pair classifier on official rows with the ChatGPT A/B annotations, revision 03; per test row and class the top-3 mean
of 6 verified references, fused at 0.10 on rows whose top-2 margin is below 0.35). K (areebbb's 0.666) has neither.
This scores both on exp036's K replica (core + deb, SVC 12, all three n-gram members on K's 2x rows, Sinkhorn 0.30) out
of fold on exp015's folds: V40A fitted on the other folds' official rows, V34.1 run on the other folds with only their
A/B annotations (in the real run the annotations cover the training rows only), both code verbatim from the pinned
notebook (kaggle/nlm/upstream_synthreaper_sovereign.ipynb).

    python -I experiments/exp037_sovereign_posthoc.py pairwise <A/B root: the NTLLM-ChatGPT-A_B folder>
    python -I experiments/exp037_sovereign_posthoc.py                      # compare

-I because the A/B files are downloads; the repo's src/ and experiments/ are put on the path below.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
if "--plain" not in sys.argv:
    sys.argv.append("--plain")
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import softmax  # noqa: E402

import exp015_nlm_core_oof as E  # noqa: E402  (reads --plain at import)
import exp034_matched_extra as X  # noqa: E402
import exp036_svc_matched as M  # noqa: E402
from exp029_atlas_ext import soft_sinkhorn  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402
from ntl.paths import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "outputs" / "exp037_sovereign_posthoc"
SOV = REPO_ROOT / "kaggle" / "nlm" / "upstream_synthreaper_sovereign.ipynb"
SOV_SHA256 = "e2bf27db239bf27fc286017389c373b35f66245b4061cfb0a66b01a365019a07"
# v34_pairwise.py run on a fold: its train.csv must equal the A/B copy of the full official train, so that check is made
# on the rows present, and the A evidence spans are cut to the fold's training rows
CV_PATCHES = [
    (" if len(canon)!=len(tr) or not canon['id'].astype(str).equals(tr['id'].astype(str)):\n",
     " canon=canon.set_index(canon['id'].astype(str)).loc[tr['id'].astype(str)].reset_index(drop=True)   # exp037: fold rows\n"
     " if len(canon)!=len(tr) or not canon['id'].astype(str).equals(tr['id'].astype(str)):\n"),
    (" spans=spans[spans['quote_validation'].astype(str).eq('exact_match')].copy()\n",
     " spans=spans[spans['quote_validation'].astype(str).eq('exact_match')].copy()\n"
     " spans=spans[spans['id'].astype(str).isin(set(tr['id'].astype(str)))].copy()   # exp037: training rows' annotations only\n")]


def sov_cell(head):
    raw = SOV.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOV_SHA256:
        raise SystemExit(f"{SOV} is not the verified Sovereign Gold Apex notebook")
    srcs = [c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in json.loads(raw)["cells"]]
    hits = [x for x in srcs if x.startswith(head)]
    assert len(hits) == 1, head
    return hits[0].split("\n", 1)[1]   # without the %%writefile line


def sibling_module():
    mod = types.ModuleType("sibling_style")
    exec(compile(sov_cell("%%writefile sibling_style.py"), "sibling_style.py", "exec"), mod.__dict__)
    return mod


def pairwise(ab_root):
    """V34.1 out of fold: per fold, train.csv = the other folds' official rows, test.csv = the fold's rows."""
    train = load_train()
    folds = np.load(E.NLM_OOF)["folds"]
    src = sov_cell("%%writefile v34_pairwise.py")
    for old, new in CV_PATCHES:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    (OUT / "pairwise").mkdir(parents=True, exist_ok=True)
    script = OUT / "pairwise" / "v34_pairwise_cv.py"
    script.write_text(src, encoding="utf-8")
    oof = np.zeros((len(train), 18), np.float32)
    for k in np.unique(folds):
        d = OUT / "pairwise" / f"fold{k}"
        d.mkdir(exist_ok=True)
        train[folds != k][["id", "prompt_id", "text", "model"]].to_csv(d / "train.csv", index=False)
        train[folds == k][["id", "text"]].to_csv(d / "test.csv", index=False)
        subprocess.run([sys.executable, "-I", str(script), "--data", str(d), "--ab-root", ab_root,
                        "--out", str(d / "probs.npy"), "--audit", str(d / "audit.json")], check=True, stdout=subprocess.DEVNULL)
        audit = json.loads((d / "audit.json").read_text())
        print(f"fold {k}: {audit['A_evidence_span_rows_used']} spans, {audit['A_feature_keys']} keys, {audit['seconds']:.0f}s", flush=True)
        oof[folds == k] = np.load(d / "probs.npy", allow_pickle=False)
    labels = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'gemini-3.1-pro-preview',
              'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna',
              'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']   # v34_pairwise.py's LABELS
    oof = oof[:, [labels.index(l) for l in E.NLM_LABELS]]
    np.save(OUT / "pairwise_oof.npy", oof)
    print(f"V34.1 pairwise alone: {balanced_accuracy(train.model, np.array(E.NLM_LABELS)[oof.argmax(1)]):.4f}", flush=True)


def v40a(logits, train, folds, alpha=0.10):
    """The residual cross-fitted: for each fold, fitted on the other folds' official rows (their own grouped OOF sets
    its reliability), applied to the fold's rows."""
    sib = sibling_module()
    labels = np.array(E.NLM_LABELS)
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    out = logits.copy()
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        out[va] = sib.apply_grouped_style_residual(logits[va], train.text.values[tr], train.text.values[va], labels, y[tr],
                                                   train.prompt_id.to_numpy()[tr], max_alpha=alpha)
    return out


def v34(logits, pair, w=0.10):
    """Sovereign's fusion: + w x gate x centred log pairwise probabilities, gate = clip((0.35 - top-2 margin) / 0.35)."""
    pair = np.clip(pair.astype(np.float64), 1e-8, None)
    pair /= pair.sum(1, keepdims=True)
    o = np.sort(softmax(logits, axis=1), axis=1)
    gate = np.clip((0.35 - (o[:, -1] - o[:, -2])) / 0.35, 0.0, 1.0)
    pl = np.log(pair)
    pl -= pl.mean(1, keepdims=True)
    return logits + w * gate[:, None] * pl


def compare():
    train = load_train()
    nlm = np.load(E.NLM_OOF)
    folds, labels = nlm["folds"], np.array(E.NLM_LABELS)
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
    members["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    E.TERMS.update({"tfidf": 1, "distort": 1})
    lp = {m: np.load(X.OUT / f"m2_{m}.npy") for m in ("tfidf", "distort")}
    svc = np.load(M.OUT / "m2_svc4.npy").astype(np.float64)
    score = lambda Z: balanced_accuracy(train.model, labels[Z.argmax(1)])
    per_fold = lambda Z: " ".join(f"{balanced_accuracy(train.model[folds == k], labels[Z[folds == k].argmax(1)]):.3f}"
                                  for k in np.unique(folds))
    pair = np.load(OUT / "pairwise_oof.npy")
    print(f"V34.1 pairwise alone: {score(pair):.4f}")
    rows = {}
    for w in (0.25, 0.5):
        L = soft_sinkhorn(E.blend({**members, "svc4": svc, **lp}, {**E.CORE, "svc4": 12.0, "deb": 1.0, "tfidf": w, "distort": 0.10}),
                          alpha=0.30)   # log-domain, as infer.py's _safe_logits
        S = v40a(L, train, folds)
        rows.update({f"K, TF-IDF {w}": L, f"  + V40A": S, f"  + V34.1": v34(L, pair), f"  + V40A + V34.1 (Sovereign)": v34(S, pair),
                     f"  + V40A + V34.1 at 0.20": v34(S, pair, 0.20)})
        ref = rows[f"K, TF-IDF 0.25"]
        for name in list(rows)[-5:]:
            g = paired_bootstrap(y, rows[name].argmax(1), ref.argmax(1), prompts)
            print(f"{name:32s} {score(rows[name]):.4f} (vs K {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}])  folds {per_fold(rows[name])}", flush=True)
        rows = {k: v for k, v in rows.items() if k == "K, TF-IDF 0.25"}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:2] == ["pairwise"]:
        pairwise(sys.argv[2])
    else:
        compare()
