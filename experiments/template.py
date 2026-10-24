"""expNNN_<slug>: one-line description of the idea.

Copy to experiments/expNNN_<slug>.py, fill in fit / predict_proba / save, run

    python experiments/expNNN_<slug>.py

and add a row to experiments/README.md. Writes outputs/expNNN_<slug>/:
    oof.parquet   out-of-fold class probabilities, one column per label
    metrics.json  CV balanced accuracy overall and per fold
    model/        everything the Kaggle inference notebook needs (full-train fit)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ntl.cv import iter_folds
from ntl.data import LABELS, load_train
from ntl.metrics import balanced_accuracy, per_class_recall
from ntl.paths import OUTPUT_DIR

NAME = Path(__file__).stem
OUT = OUTPUT_DIR / NAME


def fit(train):
    """Train on rows with `text` and `model` columns; return the fitted object."""
    raise NotImplementedError


def predict_proba(fitted, texts):
    """Return an (n, 18) array whose columns follow ntl.data.LABELS."""
    raise NotImplementedError


def save(fitted, model_dir):
    """Write the fitted object to model_dir for offline inference on Kaggle."""
    raise NotImplementedError


def main():
    train = load_train()
    labels = np.array(LABELS)
    oof = np.zeros((len(train), len(LABELS)))
    fold_scores = []
    for fold, tr, va in iter_folds(train["prompt_id"]):
        fitted = fit(train.iloc[tr])
        oof[va] = predict_proba(fitted, train["text"].iloc[va].tolist())
        fold_scores.append(balanced_accuracy(train["model"].iloc[va], labels[oof[va].argmax(1)]))
        print(f"fold {fold}: {fold_scores[-1]:.4f}")

    pred = labels[oof.argmax(1)]
    cv = balanced_accuracy(train["model"], pred)
    print(f"CV balanced accuracy: {cv:.4f}")
    print(per_class_recall(train["model"], pred).round(3).to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [train[["id", "prompt_id", "model"]], pd.DataFrame(oof, columns=LABELS)], axis=1
    ).to_parquet(OUT / "oof.parquet", index=False)
    (OUT / "metrics.json").write_text(json.dumps({"cv": cv, "folds": fold_scores}, indent=2))
    save(fit(train), OUT / "model")


if __name__ == "__main__":
    main()
