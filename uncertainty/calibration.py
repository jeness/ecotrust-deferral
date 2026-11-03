"""
Calibration metrics for EcoTrust.

Expected Calibration Error (ECE):
    Measures whether a model's predicted probabilities match
    empirical accuracy. A well-calibrated model with confidence 0.8
    should be correct ~80% of the time.

ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

Also computes:
    MCE  - Maximum Calibration Error (worst bin)
    Brier Score - proper scoring rule for probabilistic predictions
    Reliability diagram data - for plotting
"""

import numpy as np
from typing import Tuple


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute ECE, MCE, and reliability diagram data.

    Args:
        y_true: binary ground truth labels (0/1)
        y_prob: predicted probabilities for class 1
        n_bins: number of confidence bins

    Returns:
        dict with ece, mce, brier_score, bin_data
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins, right=True)
    bin_indices = np.clip(bin_indices, 1, n_bins)

    bin_accs  = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_sizes = np.zeros(n_bins)

    for b in range(1, n_bins + 1):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        bin_accs[b-1]  = y_true[mask].mean()
        bin_confs[b-1] = y_prob[mask].mean()
        bin_sizes[b-1] = mask.sum()

    n = len(y_true)
    weights = bin_sizes / n
    ece = float(np.sum(weights * np.abs(bin_accs - bin_confs)))
    mce = float(np.max(np.abs(bin_accs - bin_confs)[bin_sizes > 0])) if (bin_sizes > 0).any() else 0.0

    # Brier score
    brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "ece":         round(ece, 4),
        "mce":         round(mce, 4),
        "brier_score": round(brier, 4),
        "bin_accs":    bin_accs.tolist(),
        "bin_confs":   bin_confs.tolist(),
        "bin_sizes":   bin_sizes.tolist(),
        "bin_edges":   bins.tolist(),
    }


def reliability_diagram_text(cal: dict) -> str:
    """ASCII reliability diagram for quick inspection."""
    lines = ["Reliability Diagram (confidence → accuracy)"]
    lines.append(f"ECE={cal['ece']:.4f}  MCE={cal['mce']:.4f}  Brier={cal['brier_score']:.4f}")
    lines.append("-" * 52)
    for b, (conf, acc, size) in enumerate(
        zip(cal["bin_confs"], cal["bin_accs"], cal["bin_sizes"])
    ):
        if size == 0:
            continue
        bar_len = int(acc * 30)
        ideal_len = int(conf * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        gap = abs(acc - conf)
        lines.append(f"[{conf:.2f}] {bar} acc={acc:.2f} gap={gap:.2f} n={int(size)}")
    return "\n".join(lines)
