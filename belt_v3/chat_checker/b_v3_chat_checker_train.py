import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def test_thresholds(
    model,
    X_test,
    y_test,
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
):
    # Your labels:
    # 0 = action
    # 1 = chat / knowledge lookup

    action_index = list(model.classes_).index(0)
    action_probabilities = model.predict_proba(X_test)[:, action_index]

    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Action Precision':<18}"
        f"{'Action Recall':<16}"
        f"{'Action F1':<12}"
        f"{'Extractor Rate':<16}"
    )

    for threshold in thresholds:
        predictions = np.where(
            action_probabilities >= threshold,
            0,
            1
        )

        accuracy = accuracy_score(y_test, predictions)

        action_precision = precision_score(
            y_test,
            predictions,
            pos_label=0,
            zero_division=0
        )

        action_recall = recall_score(
            y_test,
            predictions,
            pos_label=0,
            zero_division=0
        )

        action_f1 = f1_score(
            y_test,
            predictions,
            pos_label=0,
            zero_division=0
        )

        extractor_rate = np.mean(predictions == 0)

        print(
            f"{threshold:<12.2f}"
            f"{accuracy:<12.3f}"
            f"{action_precision:<18.3f}"
            f"{action_recall:<16.3f}"
            f"{action_f1:<12.3f}"
            f"{extractor_rate:<16.3f}"
        )


def detailed_analysis(model, X_test, y_test, threshold):
    action_index = list(model.classes_).index(0)
    action_probabilities = model.predict_proba(X_test)[:, action_index]

    predictions = np.where(
        action_probabilities >= threshold,
        0,
        1
    )

    print(f"\nThreshold: {threshold}")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["action", "chat/knowledge"],
            zero_division=0
        )
    )