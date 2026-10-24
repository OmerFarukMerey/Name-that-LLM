"""Build and validate submission.parquet against Kaggle's rejection rules."""

from pathlib import Path

import pandas as pd

from ntl.data import LABELS


def validate_submission(sub, test_ids):
    """Raise ValueError if Kaggle would reject `sub` for these test ids."""
    problems = []
    missing_cols = {"id", "model"} - set(sub.columns)
    if missing_cols:
        raise ValueError(f"missing columns: {sorted(missing_cols)}")
    if sub[["id", "model"]].isna().any().any():
        problems.append("null values")
    if sub["id"].duplicated().any():
        problems.append(f"{sub['id'].duplicated().sum()} repeated ids")
    unknown = set(sub["model"].dropna()) - set(LABELS)
    if unknown:
        problems.append(f"labels outside the 18 valid: {sorted(unknown)[:5]}")
    missing_ids = set(test_ids) - set(sub["id"])
    extra_ids = set(sub["id"]) - set(test_ids)
    if missing_ids:
        problems.append(f"{len(missing_ids)} test ids missing")
    if extra_ids:
        problems.append(f"{len(extra_ids)} ids not in test")
    if problems:
        raise ValueError("invalid submission: " + "; ".join(problems))


def write_submission(ids, preds, path="submission.parquet"):
    """Validate and write predictions; returns the written path."""
    sub = pd.DataFrame({"id": list(ids), "model": list(preds)})
    validate_submission(sub, sub["id"])
    path = Path(path)
    sub.to_parquet(path, index=False)
    return path
