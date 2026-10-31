"""exp012: more typography guards for nlm-v27.

nlm-v27 ends with five hard "typography guards": a mark (say a spaced em dash) that some models never use in train,
with zero violations on Dhruv Tibarewal's 120k pool. When a test passage contains the mark and the top label is a
model that never uses it, every such model is ruled out for that passage. This experiment mines more marks the
same way (all non-ASCII characters plus a few ASCII patterns) and measures what they add out of fold.

Candidate rule, as nlm-v27's: allowed = models that use the mark at least once in train.csv; keep the mark only if
no pool passage from another model contains it (zero violations on 119,899 passages) and at least one model is
ruled out.

Out-of-fold estimate, without leakage: lightningv08's OOF bank (data/external/oofbank, nested blend, per-passage
scenario) on its own 5 prompt-grouped folds. For fold k the allowed sets are rebuilt from the other folds' train rows
only, so a guard can also break a row whose model uses the mark only in fold k: the real risk on the hidden test.
The pool check stays as above. Gains are compared with a paired prompt bootstrap (ntl.oofbank.paired_bootstrap).

    PYTHONPATH=src /opt/anaconda3/bin/python experiments/exp012_typography_guards.py
"""

import re
import time
from collections import Counter

import numpy as np
import pandas as pd

from ntl.data import LABELS, load_train
from ntl.metrics import balanced_accuracy
from ntl.oofbank import bank_folds, load_bank, log_softmax, paired_bootstrap
from ntl.paths import REPO_ROOT

POOL = REPO_ROOT / "data" / "external" / "nlm" / "nlm_pool_v5.parquet"
MIN_PASSAGES = 20   # a non-ASCII character must occur in at least this many train + pool passages to be a candidate

# nlm-v27's guards (mark, allowed models), copied from its infer.py
V27 = [(" — ", {"claude-opus-4-6", "claude-opus-5", "glm-5.3", "glm-5.3-flash"}),
       ("’", {l for l in LABELS if not l.startswith(("claude", "glm"))}),
       ('"', {"claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
              "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "glm-5.3",
              "glm-5.3-flash", "gpt-5.5", "kimi-k3"}),
       ("“", {"gemini-3.8-flash", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
                   "kimi-k3", "qwen3.8-max"}),
       ("–", {"claude-opus-5", "gemini-3.1-pro-preview", "glm-5.3-flash", "gpt-5.6-luna", "gpt-5.6-sol",
                   "gpt-5.6-terra", "kimi-k3"})]

# ASCII and multi-character patterns worth a look besides single non-ASCII characters (substring marks only, so the
# notebook can apply them with `mark in text` exactly as nlm-v27 does)
PATTERNS = [" — ", "—", " – ", " - ", " -- ", "--", "...", "…", "  ", "\n", ";", ":", "!", "?",
            "(", "*", "#", "`", "&", "%", "/", "e.g.", "i.e.", "etc.", "'", '"', "‘", "”", " ",
            " ", "‑", " "]


def contains(texts, mark):
    return np.fromiter((mark in t for t in texts), bool, len(texts))


def candidates(train, pool):
    counts = Counter()
    for t in list(train.text) + list(pool.text):
        counts.update({c for c in t if ord(c) > 127})
    marks = sorted({c for c, n in counts.items() if n >= MIN_PASSAGES} | set(PATTERNS) | {m for m, _ in V27})
    return marks


def mine(train, pool, marks, rows=None):
    """Guards (mark, allowed set) from train rows `rows` (all if None) with zero pool violations."""
    rows = np.ones(len(train), bool) if rows is None else rows
    ty, py = train.model.to_numpy(), pool.model.to_numpy()
    out, report = [], []
    for mark in marks:
        tin, pin = TRAIN_HAS[mark], POOL_HAS[mark]
        allowed = set(ty[rows & tin])
        if not allowed or len(allowed) == len(LABELS):
            continue
        violations = int((pin & ~np.isin(py, list(allowed))).sum())
        report.append(dict(mark=repr(mark), allowed=len(allowed), train_rows=int(tin.sum()), pool_rows=int(pin.sum()),
                           pool_violations=violations))
        if violations == 0:
            out.append((mark, allowed))
    return out, pd.DataFrame(report)


def apply(logp, texts_has, guards, labels=np.array(LABELS)):
    """nlm-v27's rule, in order: for each guard, if the passage has the mark and the top label is not allowed,
    rule out every model the guard does not allow."""
    score = logp.copy()
    changed = 0
    for mark, allowed in guards:
        forbid = ~np.isin(labels, list(allowed))
        hit = texts_has[mark] & forbid[score.argmax(1)]
        score[np.ix_(hit, forbid)] = -1e9
        changed += int(hit.sum())
    return score, changed


def oof_guards(train, pool, marks, folds, logp, guard_fn):
    """Apply guards mined on the other folds to each fold's OOF log-probabilities."""
    out = logp.copy()
    changed = 0
    for k in np.unique(folds):
        valid = folds == k
        guards = guard_fn(folds != k)
        sub = {m: TRAIN_HAS[m][valid] for m in marks}
        out[valid], c = apply(logp[valid], sub, guards)
        changed += c
    return out, changed


def main():
    global TRAIN_HAS, POOL_HAS
    t0 = time.time()
    train = load_train()
    pool = pd.read_parquet(POOL, columns=["text", "model"])
    pool = pool[pool.model.isin(LABELS)].reset_index(drop=True)
    pool["text"] = pool.text.astype(str)
    marks = candidates(train, pool)
    TRAIN_HAS = {m: contains(train.text.to_numpy(), m) for m in marks}
    POOL_HAS = {m: contains(pool.text.to_numpy(), m) for m in marks}
    print(f"{len(marks)} candidate marks, {len(train)} train rows, {len(pool)} pool rows ({time.time() - t0:.0f}s)")

    full, report = mine(train, pool, marks)
    report = report.sort_values(["pool_violations", "train_rows"], ascending=[True, False])
    print("\nevery candidate that rules out at least one model (allowed = train users):")
    print(report.to_string(index=False))

    # the v27 guards as published versus their train-mined sets
    print("\nnlm-v27 guards: published allowed set vs models using the mark in train; pool violations of the published set")
    py = pool.model.to_numpy()
    for mark, allowed in V27:
        used = set(train.model[TRAIN_HAS[mark]])
        viol = int((POOL_HAS[mark] & ~np.isin(py, list(allowed))).sum())
        print(f"  {mark!r:10s} published {len(allowed):2d} | train users {len(used):2d} | same {used == allowed} | "
              f"pool violations {viol}")

    names, comps, nested, y = load_bank()
    folds = bank_folds(train)
    base = log_softmax(nested[0])
    labels = np.array(LABELS)
    prompts = train.prompt_id.to_numpy()
    v27_marks = [m for m, _ in V27]
    new_marks = [m for m, _ in full if m not in v27_marks]

    def score(name, logp, changed):
        pred = logp.argmax(1)
        bal = balanced_accuracy(train.model, labels[pred])
        return pred, bal, changed

    base_pred, base_bal, _ = score("bank", base, 0)
    print(f"\nOOF bank nested blend (per-passage scenario): balanced accuracy {base_bal:.4f}")
    # (a) v27 guards exactly as published (fixed sets: what the notebook does)
    v27_logp, v27_changed = apply(base, TRAIN_HAS, V27)
    v27_pred, v27_bal, _ = score("v27", v27_logp, v27_changed)
    print(f"(a) + nlm-v27 guards as published: {v27_bal:.4f} ({v27_changed} rows changed), "
          f"gain {paired_bootstrap(y, v27_pred, base_pred, prompts)}")
    # (b) v27 marks, allowed sets re-mined per fold (fold-safe estimate of the v27 guards' value)
    b_logp, b_changed = oof_guards(train, pool, v27_marks, folds, base,
                                   lambda rows: mine(train, pool, v27_marks, rows)[0])
    b_pred, b_bal, _ = score("v27 oof", b_logp, b_changed)
    print(f"(b) nlm-v27 marks, sets mined on the other folds: {b_bal:.4f} ({b_changed} changed), "
          f"gain {paired_bootstrap(y, b_pred, base_pred, prompts)}")
    # (c) the new marks on top of (a), sets mined on the other folds
    c_logp, c_changed = oof_guards(train, pool, new_marks, folds, v27_logp,
                                   lambda rows: mine(train, pool, new_marks, rows)[0])
    c_pred, c_bal, _ = score("new oof", c_logp, c_changed)
    print(f"(c) (a) + {len(new_marks)} new marks, sets mined on the other folds: {c_bal:.4f} ({c_changed} changed), "
          f"gain over (a) {paired_bootstrap(y, c_pred, v27_pred, prompts)}")
    fixed = int(((c_pred == y) & (v27_pred != y)).sum()); broken = int(((c_pred != y) & (v27_pred == y)).sum())
    print(f"    rows fixed {fixed}, rows broken {broken}")
    # per new mark, alone on top of (a)
    rows = []
    for mark in new_marks:
        m_logp, m_changed = oof_guards(train, pool, [mark], folds, v27_logp,
                                       lambda r, mark=mark: mine(train, pool, [mark], r)[0])
        pred = m_logp.argmax(1)
        rows.append(dict(mark=repr(mark), allowed=len(dict(full)[mark]), changed=m_changed,
                         fixed=int(((pred == y) & (v27_pred != y)).sum()),
                         broken=int(((pred != y) & (v27_pred == y)).sum()),
                         gain=round(balanced_accuracy(train.model, labels[pred]) - v27_bal, 4)))
    print("\nnew marks one at a time on top of (a):")
    print(pd.DataFrame(rows).sort_values("gain", ascending=False).to_string(index=False))
    print("\nfull-train guards for the notebook (new marks only):")
    for mark, allowed in full:
        if mark not in v27_marks:
            print(f"  ({mark!r}, {sorted(allowed)}),")


if __name__ == "__main__":
    main()
