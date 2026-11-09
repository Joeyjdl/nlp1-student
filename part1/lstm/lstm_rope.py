import json
from pathlib import Path
from statistics import mean

from part1.analysis import BenavoliCorrelatedT


def main():
    # Read the validation scores saved by the LSTM experiment.
    scores_file = Path(__file__).with_name("lstm_cv_5fold_seed42.json")
    results = json.loads(scores_file.read_text(encoding="utf-8"))

    with_features = results["accuracies"]["with_features"]
    without_features = results["accuracies"]["without_features"]

    # Compare both models on the same validation fold.
    print("Fold | With extras | Without extras | Difference (pp)")
    paired_scores = zip(with_features, without_features, strict=True)

    for fold, (score_with, score_without) in enumerate(paired_scores, start=1):
        difference_pp = (score_without - score_with) * 100
        print(f"{fold} | {score_with:.2%} | {score_without:.2%} | {difference_pp:+.2f}")

    print(f"Mean accuracy with extras: {mean(with_features):.2%}")
    print(f"Mean accuracy without extras: {mean(without_features):.2%}")

    # The course helper measures B minus A.
    # A = with extra features. B = without extra features.
    # runs=1 represents one run of five-fold cross-validation.
    comparison = BenavoliCorrelatedT(with_features, without_features, runs=1)
    print(f"Mean advantage without extras: {comparison.mean_diff * 100:+.2f} pp")

    # Check whether the conclusion changes at different ROPE widths.
    print("ROPE (+/- pp) | P(with better) | P(equivalent) | P(without better)")
    for width in (0.0025, 0.005, 0.01):
        with_better, equivalent, without_better = comparison.probs(width)
        print(
            f"{width * 100:.2f} | {with_better:.2%} | "
            f"{equivalent:.2%} | {without_better:.2%}"
        )


if __name__ == "__main__":
    main()

    