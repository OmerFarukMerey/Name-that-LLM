import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

from ntl.cv import iter_folds, prompt_folds
from ntl.data import LABELS, load_test, load_train
from ntl.metrics import balanced_accuracy
from ntl.paths import DATA_DIR
from ntl.submission import validate_submission, write_submission

needs_data = pytest.mark.skipif(
    not (DATA_DIR / "train.csv").exists(), reason="run `make data` first"
)


def test_labels_are_the_18_valid_names():
    assert len(LABELS) == len(set(LABELS)) == 18


def test_balanced_accuracy_matches_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.choice(LABELS, 500)
    y_pred = np.where(rng.random(500) < 0.6, y_true, rng.choice(LABELS, 500))
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(
        balanced_accuracy_score(y_true, y_pred)
    )


def test_folds_never_split_a_prompt_and_are_deterministic():
    prompt_ids = [f"p{i % 150:03d}" for i in range(2700)]
    folds = prompt_folds(prompt_ids)
    by_prompt = pd.Series(folds).groupby(pd.Series(prompt_ids)).nunique()
    assert (by_prompt == 1).all()
    assert np.array_equal(folds, prompt_folds(prompt_ids))
    sizes = [len(valid) for _, _, valid in iter_folds(prompt_ids)]
    assert sum(sizes) == 2700 and max(sizes) - min(sizes) <= 18


def test_validate_submission_rejects_what_kaggle_rejects():
    ids = ["a", "b", "c"]
    good = pd.DataFrame({"id": ids, "model": ["gpt-5.5", "kimi-k3", "glm-5.3"]})
    validate_submission(good, ids)
    bad_cases = [
        good.assign(model=["gpt-5.5", "kimi-k3", "gpt-4"]),
        good.assign(model=["gpt-5.5", None, "glm-5.3"]),
        good.assign(id=["a", "a", "c"]),
        good.iloc[:2],
        good.drop(columns="model"),
    ]
    for bad in bad_cases:
        with pytest.raises(ValueError):
            validate_submission(bad, ids)


def test_write_submission_round_trips(tmp_path):
    path = write_submission(["a", "b"], ["gpt-5.5", "kimi-k3"], tmp_path / "s.parquet")
    sub = pd.read_parquet(path)
    assert list(sub.columns) == ["id", "model"] and len(sub) == 2


@needs_data
def test_train_labels_and_sample_submission_match_label_set():
    train, test = load_train(), load_test()
    assert set(train["model"]) == set(LABELS)
    sample = pd.read_parquet(DATA_DIR / "sample_submission.parquet")
    validate_submission(sample, test["id"])
