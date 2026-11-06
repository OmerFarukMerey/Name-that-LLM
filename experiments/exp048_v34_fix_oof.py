"""exp048: does V34.1's test-loop index fix (sambitpadhan's public 0.667; build_nlm.py's `vfix`) change the pairwise
verifier? Its test loop reads Cte[q-start], Wte[q-start] and test_texts[z-start], where q already indexes the test rows,
so every batch after the first takes 4 of its 10 pair features from the first 128 test passages.

The verifier is public code, so it runs on Kaggle, not here: kernel ntl-v34-fix-oof (CPU) runs the scored 0.668's
v34_pairwise.py on each exp015 fold with that fold as the test set, once as published ("buggy") and once with the fix
("fixed"). Two simulation edits only: the A/B canonical train file is cut to the training folds, and so are the A evidence
spans (a held-out row's own annotation must not build the test profiles). Output: v34_oof_{buggy,fixed}.npy (train
order, the verifier's 18 labels = NLM_LABELS).

    python experiments/exp048_v34_fix_oof.py build          # writes kaggle/nlm/ntl-v34-fix-oof/
    python -I experiments/exp048_v34_fix_oof.py score <dir> # rates the downloaded outputs (standalone and in the replica)
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
KERNEL = "ntl-v34-fix-oof"
SCORED = "ntl-v31ac-a20-sk30-osvc12-xm1-sov-cx-p30"   # the 0.668: its v34_pairwise.py is the one that ran
AB_DATA = "mercurydust/ntllm-chatgpt-a-b"
SIM = [("canon=pd.read_csv(root/'inputs/train.csv').fillna({'text':''})\n",
        "canon=pd.read_csv(root/'inputs/train.csv').fillna({'text':''})\n"
        " canon=canon[canon['id'].astype(str).isin(set(tr['id'].astype(str)))].reset_index(drop=True)   # exp048: training folds only\n"),
       ("spans=pd.read_csv(root/'dataset_A_cue_spans.csv',low_memory=False)\n",
        "spans=pd.read_csv(root/'dataset_A_cue_spans.csv',low_memory=False)\n"
        " spans=spans[spans['id'].astype(str).isin(set(tr['id'].astype(str)))]   # exp048: no held-out row's own evidence\n")]
FIX = [("Cte[q-start]", "Cte[q]"), ("Wte[q-start]", "Wte[q]"), ("test_texts[z-start]", "test_texts[z]")]
DRIVER = r'''
import glob, json, os, subprocess, sys, time
import numpy as np, pandas as pd
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
AB = next(os.path.dirname(m) for m in glob.glob('/kaggle/input/**/data_manifest.json', recursive=True)
          if str(json.load(open(m)).get('revision')) == '03')
FOLDS = np.array([int(c) for c in FOLD_DIGITS])
SIM, FIX = SIM_EDITS, FIX_EDITS
def edit(src, edits):
    for old, new in edits:
        assert src.count(old) >= 1, old
        src = src.replace(old, new)
    return src
src = open('v34_pairwise.py').read()
open('v34_buggy.py', 'w').write(edit(src, SIM)); open('v34_fixed.py', 'w').write(edit(edit(src, SIM), FIX))
assert open('v34_fixed.py').read().count('Cte[q-start]') == 0 and src.count('test_texts[z-start]') == 3
tr = pd.read_csv(DATA + '/train.csv'); assert len(tr) == len(FOLDS)
out = {v: np.zeros((len(tr), 18), np.float32) for v in ('buggy', 'fixed')}
for f in range(5):
    d = f'/kaggle/working/fold{f}'; os.makedirs(d, exist_ok=True)
    tr[FOLDS != f].to_csv(d + '/train.csv', index=False); tr.loc[FOLDS == f, ['id', 'text']].to_csv(d + '/test.csv', index=False)
    for v in ('buggy', 'fixed'):
        t = time.time()
        subprocess.run([sys.executable, f'v34_{v}.py', '--data', d, '--ab-root', AB, '--out', f'{d}/{v}.npy', '--audit', f'{d}/{v}.json'],
                       check=True, stdout=subprocess.DEVNULL)
        out[v][FOLDS == f] = np.load(f'{d}/{v}.npy')
        print('fold', f, v, round(time.time() - t), 's', flush=True)
for v in out:
    np.save(f'/kaggle/working/v34_oof_{v}.npy', out[v])
tr[['id']].to_csv('/kaggle/working/v34_oof_ids.csv', index=False)
from sklearn.metrics import balanced_accuracy_score
LABELS = sorted(tr.model.unique())
y = tr.model.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
for v in out:
    print(v, 'standalone balanced accuracy', round(balanced_accuracy_score(y, out[v].argmax(1)), 4), flush=True)
'''


def build():
    import numpy as np
    sys.argv.append("--plain")
    import exp015_nlm_core_oof as E
    nb_in = json.loads((_ROOT / "kaggle/nlm" / SCORED / f"{SCORED}.ipynb").read_text(encoding="utf-8"))
    v34 = next(s for s in ("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb_in["cells"])
               if s.startswith("%%writefile v34_pairwise.py"))
    folds = np.load(E.NLM_OOF)["folds"]
    assert len(folds) == 2699 and set(folds.tolist()) == set(range(5))
    driver = (DRIVER.replace("FOLD_DIGITS", repr("".join(str(int(x)) for x in folds)))
              .replace("SIM_EDITS", repr(SIM)).replace("FIX_EDITS", repr(FIX)))
    intro = f"# {KERNEL}\n\nexp048: V34.1's pairwise verifier per exp015 fold, as published and with the test-loop index fix. Not a submission."
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": intro}]
          + [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s} for s in (v34, driver.lstrip("\n"))],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    out = _ROOT / "kaggle/nlm" / KERNEL
    out.mkdir(exist_ok=True)
    (out / f"{KERNEL}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{KERNEL}", "title": KERNEL, "code_file": f"{KERNEL}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": [AB_DATA], "kernel_sources": [], "competition_sources": ["name-that-llm"],
            "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("built", out, "v34 source", len(v34), "chars")


def score(d):
    import numpy as np
    sys.argv.append("--plain")
    import exp015_nlm_core_oof as E
    import exp034_matched_extra as X
    import exp035_comp_extra as CX
    import exp036_svc_matched as M
    from exp029_atlas_ext import soft_sinkhorn
    from ntl.data import load_train
    from ntl.metrics import balanced_accuracy
    from ntl.oofbank import paired_bootstrap
    from scipy.special import softmax
    d = Path(d)
    train = load_train()
    ids = (d / "v34_oof_ids.csv").read_text(encoding="utf-8").split()[1:]
    assert ids == train.id.astype(str).tolist(), "row order"
    labels = np.array(E.NLM_LABELS)
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    pair = {v: np.load(d / f"v34_oof_{v}.npy", allow_pickle=False).astype(np.float64) for v in ("buggy", "fixed")}
    for v, p in pair.items():
        assert p.shape == (len(train), 18) and np.isfinite(p).all()
        print(f"{v:6s} standalone balanced accuracy {balanced_accuracy(train.model, labels[p.argmax(1)]):.4f}")
    # K's replica (exp045's) as the base, then V34.1's fusion as infer.py applies it: on rows whose top-2 margin is
    # below 0.35, + w * gate * centred log pair (gate = (0.35 - margin) / 0.35), before the balancing
    nlm = np.load(E.NLM_OOF)
    mem = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
    mem["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    mem["svc4"] = np.load(M.OUT / "m2_svc4.npy").astype(np.float64)
    mem["distort"] = np.load(X.OUT / "m2_distort.npy")
    mem["tfidf"] = np.load(X.OUT / "m2_tfidf.npy")
    E.TERMS.update({"tfidf": 1, "distort": 1})
    sty = np.load(CX.OUT / "ntl-v31ac-a20-sk30-osvc12-xm2-cx-p30_stylegram.npy")
    C = np.exp(3.0 * E.lsm(sty)); C /= C.sum(1, keepdims=True)
    P = E.blend(mem, {**E.CORE, "svc4": 12.0, "deb": 1.0, "tfidf": 0.25, "distort": 0.10})
    P = 0.85 * P + 0.15 * C; P /= P.sum(1, keepdims=True)
    logp = np.log(P + 1e-12)
    srt = np.sort(softmax(logp, axis=1), axis=1)
    gate = np.clip((0.35 - (srt[:, -1] - srt[:, -2])) / 0.35, 0.0, 1.0)
    ref = soft_sinkhorn(P, alpha=0.30)
    score = lambda S: balanced_accuracy(train.model, labels[S.argmax(1)])
    print(f"replica without the verifier: {score(ref):.4f}; rows with gate > 0: {int((gate > 0).sum())}")
    for w in (0.10, 0.20, 0.40):
        cells = []
        for v, p in pair.items():
            pl = np.log(np.clip(p, 1e-8, None)); pl -= pl.mean(1, keepdims=True)
            S = soft_sinkhorn(softmax(logp + w * gate[:, None] * pl, axis=1), alpha=0.30)
            g = paired_bootstrap(y, S.argmax(1), ref.argmax(1), prompts)
            cells.append(f"{v} {score(S):.4f} ({g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}])")
        print(f"verifier at {w}: " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else score(sys.argv[2])
