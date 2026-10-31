"""exp024: soft Sinkhorn (PeterDreamHan's public "Soft Sinkhorn 0.5", 0.653) next to Roman Tamrazov's typography step,
out of fold.

infer.py's final log-probabilities are (1 - a) log P + a log Q, Q = Sinkhorn(P) (100 iterations, computed on the test
batch), so `a` scales the per-model log-offset of full balancing. Out-of-fold rows carry no train-test shift, so this
cannot show the balancing gain the public LB shows; it measures (1) how much Roman's step still adds at each `a` (do
the two levers overlap?) and (2) the cost of each `a` when the test mix is uneven (pools whose model mix is drawn from a
symmetric Dirichlet, the forum's check, Sinkhorn computed on each pool). Base: exp020's no-post replica (core + deb).

    PYTHONPATH=src python experiments/exp024_soft_sinkhorn_oof.py
"""

import numpy as np
from scipy.special import softmax

import exp020_typobayes_oof as T  # noqa: E402  (appends --plain and imports exp015 as T.E)
import exp021_typobayes_tuning as U  # noqa: E402
from ntl.data import load_train
from ntl.metrics import balanced_accuracy
from ntl.oofbank import paired_bootstrap

E = T.E
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def final(P, alpha):
    return (1 - alpha) * np.log(P + 1e-12) + alpha * np.log(E.sinkhorn(P) + 1e-12)


def typo_step(raw, ev):
    qs = np.sort(softmax(raw, 1), 1)
    return U.apply(raw, qs[:, -1] - qs[:, -2], ev, *U.DEFAULT)


def uneven_pools(y, conc, size=2000, draws=40, seed=0):
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(y == c) for c in range(18)]
    for _ in range(draws):
        counts = rng.multinomial(size, rng.dirichlet([conc] * 18))
        yield np.concatenate([rng.choice(ix, k) for ix, k in zip(by_class, counts) if k])


def main():
    train = load_train()
    y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
    prompts, texts = train.prompt_id.to_numpy(), train.text.to_numpy()
    nlm = np.load(E.NLM_OOF); folds = nlm["folds"]
    members = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    members["svc4"] = np.load(E.OUT / "svc4_oof.npy").astype(np.float64)
    z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
    assert np.array_equal(z["ids"], train.id.to_numpy(dtype=str))
    members["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
    P0 = E.blend(members, {**E.CORE, "deb": 1.0})
    ev = U.oof_evidence(U.markers(texts, T.MARKS), y, folds)

    print("all OOF rows (even mix):")
    ref = typo_step(final(P0, 1.0), ev)
    for a in ALPHAS:
        raw = final(P0, a); plain, step = raw.argmax(1), typo_step(raw, ev)
        c = paired_bootstrap(y, step, ref, prompts)
        print(f"  a={a:<5g} no step {balanced_accuracy(y, plain):.4f}  step {balanced_accuracy(y, step):.4f} "
              f"(step adds {balanced_accuracy(y, step) - balanced_accuracy(y, plain):+.4f}; vs a=1 + step {c[0]:+.4f} "
              f"[{c[1]:+.4f}, {c[2]:+.4f}]; argmax changes vs a=1 + step {(step != ref).mean():.3f})")
    for conc in (100, 20, 5, 1):
        res = np.array([[[balanced_accuracy(y[r], p) for p in (final(P0[r], a).argmax(1), typo_step(final(P0[r], a), ev[r]))]
                         for a in ALPHAS] for r in uneven_pools(y, conc)])
        m = res.mean(0)
        print(f"Dirichlet({conc:>3}) pools, mean of 40: " + "  ".join(
            f"a={a:g} {m[i, 0]:.4f}/{m[i, 1]:.4f}" for i, a in enumerate(ALPHAS)) + "   (no step / step)")


if __name__ == "__main__":
    main()
