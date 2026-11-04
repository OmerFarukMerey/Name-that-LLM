"""exp041: the replica (exp015 core + DeBERTa, SVC 12, TF-IDF and distort on K's rows, Sinkhorn 0.30) rated TF-IDF 0.5 at
+0.0045 over K's 0.25 (exp036), yet it scored 0.663 against K's 0.666 on the LB (#48), and 0.75 too (#51). The replica
has no complementary stack, which infer.py mixes in at 15% (probabilities) and whose largest term is an n-gram model,
the stylegram SVC (weight 1.516 of ~3 in its log blend). This adds that SVC's out-of-fold scores (exp035, with the P30
or the cx extra passages) as a stand-in stack, softmax(c x log-softmax), c 1.516 (its own weight) or 3 (the stack's
sharpness), and scans the TF-IDF weight with and without it.

    python -I experiments/exp041_replica_stack.py

-I for consistency with the other replica scripts; the repo's src/ and experiments/ are put on the path below.
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


def main():
    train = load_train()
    nlm = np.load(E.NLM_OOF)
    labels = np.array(E.NLM_LABELS)
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts = train.prompt_id.to_numpy()
    mem = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
    mem["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    mem["svc4"] = np.load(M.OUT / "m2_svc4.npy").astype(np.float64)
    mem["distort"] = np.load(X.OUT / "m2_distort.npy")
    mem["tfidf"] = np.load(X.OUT / "m2_tfidf.npy")
    E.TERMS.update({"tfidf": 1, "distort": 1})
    sty = {n: np.load(CX.OUT / f"ntl-v31ac-a20-sk30-osvc12-xm2{n}-p30_stylegram.npy") for n in ("", "-cx")}
    score = lambda Z: balanced_accuracy(train.model, labels[Z.argmax(1)])

    def run(tf, comp=None, c=1.516, share=0.15):
        P = E.blend(mem, {**E.CORE, "svc4": 12.0, "deb": 1.0, "tfidf": tf, "distort": 0.10})
        if comp is not None:   # infer.py: P = (1 - comp_w) P + comp_w softmax(comp_logit)
            C = np.exp(c * E.lsm(comp)); C /= C.sum(1, keepdims=True)
            P = (1 - share) * P + share * C; P /= P.sum(1, keepdims=True)
        return soft_sinkhorn(P, alpha=0.30)

    for n, d in sty.items():
        print(f"stylegram SVC{n or ' (P30)'} alone {score(d):.4f}")
    for name, comp, c in (("no stack", None, 0), ("stylegram P30 x1.516", sty[""], 1.516), ("stylegram P30 x3", sty[""], 3.0),
                          ("stylegram cx x1.516", sty["-cx"], 1.516), ("stylegram cx x3", sty["-cx"], 3.0)):
        ref = run(0.25, comp, c)
        cells = []
        for tf in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            P = run(tf, comp, c)
            cells.append(f"{tf}: {score(P):.4f} ({paired_bootstrap(y, P.argmax(1), ref.argmax(1), prompts)[0]:+.4f})")
        print(f"{name:21s} TF-IDF " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
