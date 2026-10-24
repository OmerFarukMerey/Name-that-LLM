"""The competition metric: balanced accuracy (mean per-class recall)."""

import pandas as pd


def per_class_recall(y_true, y_pred):
    """Recall for each label present in y_true, sorted worst first."""
    frame = pd.DataFrame({"true": list(y_true), "pred": list(y_pred)})
    hit = frame["true"] == frame["pred"]
    return hit.groupby(frame["true"]).mean().sort_values()


def balanced_accuracy(y_true, y_pred):
    return float(per_class_recall(y_true, y_pred).mean())
