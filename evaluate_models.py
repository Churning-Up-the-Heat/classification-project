import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, accuracy_score

def evaluate_threshold(t, y, probs):
    yhat = (probs > t).astype(int)
    return {
        "threshold": t,
        "precision": precision_score(y, yhat),
        "recall": recall_score(y, yhat),
        "accuracy": accuracy_score(y, yhat),
    }

def evaluate_thresholds(y, probs):
    return pd.DataFrame(
        [evaluate_threshold(t, y, probs) for t in np.arange(0, 1.01, 0.01)]
    )

def plot_metrics_by_thresholds(y, probs, subplots=False):
    evaluation = evaluate_thresholds(y, probs)
    axs = (
        evaluation.query("precision > 0")
        .set_index("threshold")
        .plot(subplots=subplots, sharex=True, sharey=True, figsize=(12, 8.5))
    )
    (axs[-1] if subplots else axs).set_xticks(np.arange(0, 1.05, 0.05))
    plt.tight_layout()