"""Metrics for tempo estimation evaluation."""


def accuracy_1(truths: list, preds: list, tol: float = 0.04):
    """Calculate accuracy within a certain tolerance (percentage)."""
    results = [0] * len(truths)
    for i, (truth, pred) in enumerate(zip(truths, preds)):
        if (abs(truth - pred) / truth) <= tol:
            results[i] = 1
    accuracy = sum(results) / len(results)

    return accuracy, results


def accuracy_2(
    truths: list,
    preds: list,
    tol: float = 0.04,
    octave_error_ratios: list = [2, 3, 1 / 2, 1 / 3],
):
    """Calculate accuracy within a certain tolerance (percentage), allowing for
    "octave errors" at the specified ratios."""
    results = [0] * len(truths)
    for i, (truth, pred) in enumerate(zip(truths, preds)):
        if (abs(truth - pred) / truth) <= tol:
            results[i] = 1
        else:
            for ratio in octave_error_ratios:
                if (abs(truth - (pred * ratio)) / truth) <= tol:
                    results[i] = 1
                    break
    accuracy = sum(results) / len(results)

    return accuracy, results
