"""Foreground IoU and pixel accuracy; counts are pooled over all pixels."""
import numpy as np


def binary_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match")
    tp = int(np.count_nonzero(prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    total = tp + tn + fp + fn
    union = tp + fp + fn
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp+tn)/total if total else 1.0,
            "iou": tp/union if union else 1.0}
