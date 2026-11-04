"""exp045: our TPU-trained member (kaggle/member/tpu_member.py, kernel ntl-tpu-op8: Qwen3-8B-Base LoRA24, stage 1 on
crtproduction's open pool, stage 2 on the official rows per exp015 fold) in K's replica with exp041's stylegram stand-in
for the complementary stack (x3 at 15%) and Sinkhorn 0.30. Its fold adapters' held-out log-probs (oof.npz) are out of
fold on exp015's split; the pool-only stage-1 model (s1_on_train.npy) never saw an official row. On the hidden test the
member adds two LP entries (plain head and centred twin), which the blend's divisor counts.

    python -I experiments/exp045_tpu_member.py <kernel output dir>[:tag] ...

A tag reads <tag>_oof.npz / <tag>_s1_on_train.npy (the multi-member driver); without one, oof.npz and s1_on_train.npy
(ntl-tpu-op8's version 1), named op8. Several members are rated one by one, in place of their v5 twin, and together.

-I because the kernel output is a download; the repo's src/ and experiments/ are put on the path below.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
if "--plain" not in sys.argv:
    sys.argv.append("--plain")
import numpy as np  # noqa: E402

import exp015_nlm_core_oof as E  # noqa: E402  (reads --plain at import)
import exp034_matched_extra as X  # noqa: E402
import exp035_comp_extra as CX  # noqa: E402
import exp036_svc_matched as M  # noqa: E402
from exp029_atlas_ext import soft_sinkhorn  # noqa: E402
from ntl.data import load_train  # noqa: E402
from ntl.metrics import balanced_accuracy  # noqa: E402
from ntl.oofbank import paired_bootstrap  # noqa: E402


PLACE = {"op8": "q8v5", "ll8op1": "ll8v5", "q14op1": "q14v5"}   # the v5 member on the same base


def load_member(spec, train):
    d, _, tag = spec.partition(":")
    d, pre = Path(d), (f"{tag}_" if tag else "")
    o = np.load(d / f"{pre}oof.npz", allow_pickle=False)
    assert list(o["ids"]) == train.id.astype(str).tolist(), "row order"
    idx = [list(o["labels"]).index(l) for l in E.NLM_LABELS]
    s1 = np.load(d / f"{pre}s1_on_train.npy", allow_pickle=False)
    return tag or "op8", E.lsm(o["logits"][:, idx].astype(np.float64)), E.lsm(s1[:, idx].astype(np.float64))


def main(specs):
    train = load_train()
    labels = np.array(E.NLM_LABELS)
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    nlm = np.load(E.NLM_OOF)
    mem = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
    mem["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    mem["svc4"] = np.load(M.OUT / "m2_svc4.npy").astype(np.float64)
    mem["distort"] = np.load(X.OUT / "m2_distort.npy")
    mem["tfidf"] = np.load(X.OUT / "m2_tfidf.npy")
    new = []
    for spec in specs:
        name, oof, s1 = load_member(spec, train)
        mem[name], mem[name + "_s1"] = oof, s1
        new.append(name)
        E.TERMS.update({name: 2, name + "_s1": 2})
    E.TERMS.update({"tfidf": 1, "distort": 1})
    sty = np.load(CX.OUT / "ntl-v31ac-a20-sk30-osvc12-xm2-cx-p30_stylegram.npy")
    C = np.exp(3.0 * E.lsm(sty)); C /= C.sum(1, keepdims=True)
    score = lambda S: balanced_accuracy(train.model, labels[S.argmax(1)])
    W0 = {**E.CORE, "svc4": 12.0, "deb": 1.0, "tfidf": 0.25, "distort": 0.10}

    def run(extra):
        P = E.blend(mem, {**W0, **extra})
        P = 0.85 * P + 0.15 * C; P /= P.sum(1, keepdims=True)
        return soft_sinkhorn(P, alpha=0.30)

    for k in ("q14v5", "q8v5", "ll8v5", *[x for n in new for x in (n, n + "_s1")]):
        agree = {j: float((mem[k].argmax(1) == mem[j].argmax(1)).mean()) for j in ("q14v5", "q8v5", "ll8v5") if j != k}
        print(f"{k:6s} alone {score(mem[k]):.4f}  argmax agreement " + " ".join(f"{j} {v:.2f}" for j, v in agree.items()))
    ref = run({})
    print(f"replica (K + stylegram stand-in, Sinkhorn 0.30): {score(ref):.4f}")
    for name in [x for n in new for x in (n, n + "_s1")]:
        cells = []
        for w in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            S = run({name: w})
            g = paired_bootstrap(y, S.argmax(1), ref.argmax(1), prompts)
            folds = [balanced_accuracy(train.model[nlm["folds"] == f], labels[S.argmax(1)][nlm["folds"] == f])
                     - balanced_accuracy(train.model[nlm["folds"] == f], labels[ref.argmax(1)][nlm["folds"] == f]) for f in range(5)]
            cells.append(f"w{w}: {score(S):.4f} ({g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}], {sum(x > 0 for x in folds)}/5 folds up)")
        print(f"+ {name}: " + " | ".join(cells), flush=True)
    for n in new:
        if n in PLACE:
            twin = PLACE[n]
            for w in (E.CORE[twin], 1.5 * E.CORE[twin]):
                S = run({n: w, twin: 0.0})
                print(f"{n} at {w} in {twin}'s place: {score(S):.4f} ({paired_bootstrap(y, S.argmax(1), ref.argmax(1), prompts)[0]:+.4f})")
    if len(new) > 1:
        for w in (0.5, 1.0, 1.5, 2.0):
            S = run({n: w for n in new})
            g = paired_bootstrap(y, S.argmax(1), ref.argmax(1), prompts)
            print(f"all new members at {w}: {score(S):.4f} ({g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}])", flush=True)


if __name__ == "__main__":
    main([x for x in sys.argv[1:] if x != "--plain"])
