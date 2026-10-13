"""Reproduce the model comparison reported in the README.

Usage:  python evaluate.py
"""
import os

import ml

QUESTIONS = ml.QUESTION_COLS


def main():
    raw = ml.load_raw(os.path.join(os.path.dirname(__file__), "data", "autism_child_data.csv"))
    print(f"Raw records: {len(raw)}  |  exact duplicate rows: {raw.duplicated().sum()}")

    # The ASD label in this dataset is defined by the AQ-10 score threshold.
    score = raw[QUESTIONS].sum(axis=1)
    rule_acc = ((score >= 7) == (raw["Class/ASD"] == "YES")).mean()
    print(f"Accuracy of the rule 'AQ-10 score >= 7': {rule_acc:.1%}\n")

    with_dupes = ml.prepare(raw, drop_duplicates=False)
    deduped = ml.prepare(raw, drop_duplicates=True)
    print(f"Unique records used for evaluation: {len(deduped)}\n")

    print(f"{'Model':26s} {'70/30 split (with dupes)':>25s} {'5-fold CV (deduplicated)':>26s}")
    for name in ml.ALGORITHMS:
        holdout = ml.holdout_scores(name, *ml.split(with_dupes, 0.3))
        cv = ml.cv_scores(name, deduped)
        print(f"{name:26s} {holdout['accuracy']:>24.1%} {cv['accuracy']:>17.1%} ± {cv['accuracy_std']:.1%}")


if __name__ == "__main__":
    main()
