"""Cross-validation grouped by prompt.

Test passages answer prompts that never appear in train, so a passage and its
17 sibling answers to the same prompt must always land in the same fold.
"""

import numpy as np

N_FOLDS = 5
SEED = 42


def prompt_folds(prompt_ids, n_folds=N_FOLDS, seed=SEED):
    """Return a fold index per row; every prompt maps to exactly one fold."""
    prompts = np.array(sorted(set(prompt_ids)))
    shuffled = prompts[np.random.default_rng(seed).permutation(len(prompts))]
    fold_of = {p: i % n_folds for i, p in enumerate(shuffled)}
    return np.array([fold_of[p] for p in prompt_ids])


def iter_folds(prompt_ids, n_folds=N_FOLDS, seed=SEED):
    """Yield (fold, train_idx, valid_idx) as positional index arrays."""
    folds = prompt_folds(prompt_ids, n_folds, seed)
    for k in range(n_folds):
        yield k, np.flatnonzero(folds != k), np.flatnonzero(folds == k)
