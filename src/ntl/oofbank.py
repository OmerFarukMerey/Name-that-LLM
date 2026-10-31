"""lightningv08's published out-of-fold bank, used as a yardstick for a new ensemble member.

lightningv08's asset dataset (ntl-olmo2-native24-final-inference-assets, folder name-that-llm-artifacts/) ships
out-of-fold log-probabilities of 38 frozen-feature heads (DeBERTa, RoBERTa, Qwen3 8B/14B and Qwen3.5 9B features,
style, SVM) for all 2,699 train rows, on its own 5 prompt-grouped folds (GroupKFold, seed 20261015). A member
trained on those folds (`bank_folds`) is scored by what it adds to a blend of the heads. That is the closest
offline check we have, because the members of the public 0.64x notebooks have no out-of-fold predictions.

Copied from the dataset into data/external/oofbank/: oof_predictions.npz (37 heads), finetuned_oof.npz (adds
deberta_ft) and fold_plan.json. Each head comes in 3 scenarios, in which a validation row sees at most 1, 3 or 6
sibling replies to its prompt. A member without prompt context is the same in all three, so scenario "1" is the
headline. Bank columns are reordered to ntl.data.LABELS.
"""

import hashlib
import json

import numpy as np
import pandas as pd

from ntl.data import LABELS, load_train
from ntl.metrics import balanced_accuracy
from ntl.paths import DATA_DIR, REPO_ROOT

BANK_DIR = REPO_ROOT / "data" / "external" / "oofbank"
TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
SCENARIOS = ("1", "3", "6")
SHARES = np.round(np.arange(0.0, 0.51, 0.025), 3)


def _check_train(train):
    if hashlib.sha256((DATA_DIR / "train.csv").read_bytes()).hexdigest() != TRAIN_SHA256:
        raise ValueError("train.csv differs from the one the bank was built on")
    return train


def bank_folds(train=None):
    """Outer fold (0-4) of every train row, in train.csv order, as in the bank's fold_plan.json."""
    train = _check_train(load_train() if train is None else train)
    plan = json.loads((BANK_DIR / "fold_plan.json").read_text(encoding="utf-8"))
    folds = np.full(len(train), -1)
    for k in range(plan["outer_folds"]):
        valid = np.array(plan["jobs"][f"outer{k}"]["valid"])
        if (folds[valid] != -1).any():
            raise ValueError(f"row in two outer folds ({k})")
        folds[valid] = k
    if (folds < 0).any() or train.groupby("prompt_id").apply(lambda d: folds[d.index].min() != folds[d.index].max()).any():
        raise ValueError("fold plan does not cover every row once, grouped by prompt")
    return folds


def load_bank():
    """Return (names, components (K, 3, N, 18) log-probabilities, nested (3, N, 18), y), rows in train.csv order."""
    train = _check_train(load_train())
    order = None
    names, comps = [], []
    for file in ("oof_predictions.npz", "finetuned_oof.npz"):
        with np.load(BANK_DIR / file, allow_pickle=False) as z:
            labels = list(z["classes"] if "classes" in z.files else z["labels"])
            if not np.array_equal(z["ids"], train["id"].to_numpy(dtype=str)):
                raise ValueError(f"{file}: ids are not in train.csv order")
            columns = [labels.index(label) for label in LABELS]
            for name, values in zip(z["component_names"], z["components"]):
                if str(name) not in names:
                    names.append(str(name))
                    comps.append(values[..., columns])
            if file == "oof_predictions.npz":
                nested = z["nested"][..., columns]
    comps = np.stack(comps)
    y = train["model"].map({label: i for i, label in enumerate(LABELS)}).to_numpy()
    return names, comps, nested, y


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def fast_balanced_accuracy(y, pred, weights=None):
    """Balanced accuracy for many row weightings at once: weights (B, N) -> (B,). Equals ntl.metrics on 1s."""
    weights = np.ones((1, len(y))) if weights is None else np.atleast_2d(weights)
    onehot = np.eye(len(LABELS))[y]
    hits = (weights * (pred == y)) @ onehot
    totals = weights @ onehot
    present = totals > 0
    return np.where(present, hits / np.where(present, totals, 1), 0).sum(1) / present.sum(1)


def greedy_weights(stack, y, steps=60):
    """The bank's own selection: nonnegative greedy log pool, balanced-accuracy objective. stack (K, N, 18)."""
    sample = 1 / (len(LABELS) * np.bincount(y, minlength=len(LABELS))[y])
    total, counts = np.zeros_like(stack[0]), np.zeros(len(stack))
    for _ in range(steps):
        scores = (((total[None] + stack).argmax(-1) == y) * sample).sum(-1)
        chosen = int(np.argmax(scores))
        total += stack[chosen]
        counts[chosen] += 1
    return counts / counts.sum()


def crossfit(fit_and_apply, folds):
    """Out-of-fold blend: fold k is predicted by a blend chosen on the other folds' OOF rows only."""
    out, chosen = None, []
    for k in np.unique(folds):
        fit, valid = folds != k, folds == k
        params, blended = fit_and_apply(fit, valid)
        out = np.zeros((len(folds),) + blended.shape[1:]) if out is None else out
        out[valid] = blended
        chosen.append(params)
    return out, chosen


def paired_bootstrap(y, pred_new, pred_old, prompt_ids, n=4000, seed=0):
    """Gain in balanced accuracy and its 95% interval, resampling whole prompts."""
    prompts, row_prompt = np.unique(prompt_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, len(prompts), len(prompts)), minlength=len(prompts))
                       for _ in range(n)])
    weights = counts[:, row_prompt]
    gain = fast_balanced_accuracy(y, pred_new, weights) - fast_balanced_accuracy(y, pred_old, weights)
    return float(gain.mean()), float(np.quantile(gain, 0.025)), float(np.quantile(gain, 0.975))


def yardstick(member_proba, scenario=0, verbose=True):
    """What a member adds to the bank. member_proba: (N, 18) OOF probabilities on bank_folds, ntl.data.LABELS order.

    Two checks, each cross-fitted over the bank's outer folds and compared with a paired prompt bootstrap:
      share   the bank's nested blend mixed with the member at one share (chosen on the other folds), the way a
              member is added to a fixed public ensemble;
      greedy  the bank's greedy selection rerun over all 38 heads with and without the member.
    """
    train = load_train()
    folds = bank_folds(train)
    names, comps, nested, y = load_bank()
    labels = np.array(LABELS)
    member = np.log(np.clip(np.asarray(member_proba, float), 1e-9, None))
    member = log_softmax(member)
    base = log_softmax(nested[scenario])
    stack = comps[:, scenario]
    report = {"member": balanced_accuracy(train["model"], labels[member.argmax(1)]),
              "bank_nested": balanced_accuracy(train["model"], labels[base.argmax(1)])}

    def share_fit(fit, valid):
        scores = [fast_balanced_accuracy(y[fit], ((1 - w) * base[fit] + w * member[fit]).argmax(1))[0] for w in SHARES]
        w = float(SHARES[int(np.argmax(scores))])
        return w, (1 - w) * base[valid] + w * member[valid]

    def greedy_fit(with_member):
        full = np.concatenate([stack, member[None]]) if with_member else stack

        def fit_apply(fit, valid):
            weights = greedy_weights(full[:, fit], y[fit])
            return (float(weights[-1]) if with_member else 0.0), np.einsum("k,knc->nc", weights, full[:, valid])
        return fit_apply

    share_oof, shares = crossfit(share_fit, folds)
    old_greedy, _ = crossfit(greedy_fit(False), folds)
    new_greedy, member_weights = crossfit(greedy_fit(True), folds)
    prompts = train["prompt_id"].to_numpy()
    for key, new, old, picked in (("share", share_oof, base, shares), ("greedy", new_greedy, old_greedy, member_weights)):
        report[f"{key}_old"] = balanced_accuracy(train["model"], labels[old.argmax(1)])
        report[f"{key}_new"] = balanced_accuracy(train["model"], labels[new.argmax(1)])
        report[f"{key}_gain"] = paired_bootstrap(y, new.argmax(1), old.argmax(1), prompts)
        report[f"{key}_member_weight_per_fold"] = picked
    if verbose:
        print(f"member alone {report['member']:.4f} | bank nested blend (scenario {SCENARIOS[scenario]}) "
              f"{report['bank_nested']:.4f}")
        for key in ("share", "greedy"):
            gain, lo, hi = report[f"{key}_gain"]
            print(f"{key:6s}: {report[f'{key}_old']:.4f} -> {report[f'{key}_new']:.4f}  gain {gain:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  member weight per fold {np.round(report[f'{key}_member_weight_per_fold'], 3).tolist()}")
    return report


def member_from_oof(path):
    """Read an oof.parquet (id + one probability column per label) into train.csv row order."""
    oof = pd.read_parquet(path)
    train = load_train()
    oof = oof.set_index("id").loc[train["id"], list(LABELS)]
    return oof.to_numpy(float)
