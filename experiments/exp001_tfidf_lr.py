"""exp001_tfidf_lr: char 1-5 + word 1-2 TF-IDF into multinomial logistic regression.

The stylometric baseline: no topic memorisation is possible across prompts, so
the signal is punctuation, function words and phrasing, which n-grams capture.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ntl.cv import iter_folds
from ntl.data import LABELS, load_train
from ntl.metrics import balanced_accuracy, per_class_recall
from ntl.paths import OUTPUT_DIR

NAME = Path(__file__).stem
OUT = OUTPUT_DIR / NAME


class Model:
    def __init__(self):
        self.char = TfidfVectorizer(
            analyzer="char", ngram_range=(1, 5), sublinear_tf=True, min_df=2, lowercase=False
        )
        self.word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2,
            lowercase=False, token_pattern=r"(?u)\b\w+\b|[^\w\s]",
        )
        self.clf = LogisticRegression(C=20, max_iter=3000)

    def _features(self, texts, fit=False):
        if fit:
            return hstack([self.char.fit_transform(texts), self.word.fit_transform(texts)]).tocsr()
        return hstack([self.char.transform(texts), self.word.transform(texts)]).tocsr()


def fit(train):
    model = Model()
    X = model._features(train["text"].tolist(), fit=True)
    model.clf.fit(X, train["model"].map(LABELS.index))
    return model


def predict_proba(fitted, texts):
    return fitted.clf.predict_proba(fitted._features(texts))


def save(fitted, model_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(fitted, f)


def main():
    train = load_train()
    labels = np.array(LABELS)
    oof = np.zeros((len(train), len(LABELS)))
    fold_scores = []
    for fold, tr, va in iter_folds(train["prompt_id"]):
        fitted = fit(train.iloc[tr])
        oof[va] = predict_proba(fitted, train["text"].iloc[va].tolist())
        fold_scores.append(balanced_accuracy(train["model"].iloc[va], labels[oof[va].argmax(1)]))
        print(f"fold {fold}: {fold_scores[-1]:.4f}", flush=True)

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
