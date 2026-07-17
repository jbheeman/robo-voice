import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


MODEL_FILE = "chat_checker_model.joblib"
TEST_FILE = "handwritten_data.csv"

# Your labels:
# 0 = action
# 1 = chat / knowledge lookup


def load_model_and_data():
    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(TEST_FILE)

    X_test = df["text"]
    y_test = df["label"]

    return model, X_test, y_test


def get_action_probabilities(model, X_test):
    # Find the probability column corresponding to class 0: action
    action_index = list(model.classes_).index(0)

    return model.predict_proba(X_test)[:, action_index]


def test_thresholds(
    action_probabilities,
    y_test,
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
):
    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Action Prec.':<15}"
        f"{'Action Recall':<16}"
        f"{'Action F1':<12}"
        f"{'Extractor Rate':<16}"
    )

    for threshold in thresholds:
        predictions = np.where(
            action_probabilities >= threshold,
            0,  # action
            1   # chat / knowledge
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
            f"{action_precision:<15.3f}"
            f"{action_recall:<16.3f}"
            f"{action_f1:<12.3f}"
            f"{extractor_rate:<16.3f}"
        )


def detailed_analysis(
    X_test,
    y_test,
    action_probabilities,
    threshold=0.4
):
    predictions = np.where(
        action_probabilities >= threshold,
        0,
        1
    )

    print(f"\nDetailed analysis at threshold {threshold:.2f}")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["action", "chat/knowledge"],
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    results = pd.DataFrame({
        "text": X_test,
        "actual": y_test,
        "predicted": predictions,
        "action_probability": action_probabilities
    })

    mistakes = results[
        results["actual"] != results["predicted"]
    ].copy()

    mistakes = mistakes.sort_values(
        by="action_probability",
        ascending=False
    )

    print(f"\nIncorrect predictions: {len(mistakes)}")

    for _, row in mistakes.iterrows():
        actual_name = (
            "action" if row["actual"] == 0 else "chat/knowledge"
        )

        predicted_name = (
            "action" if row["predicted"] == 0 else "chat/knowledge"
        )

        print("\n" + "-" * 70)
        print("Text:", row["text"])
        print("Actual:", actual_name)
        print("Predicted:", predicted_name)
        print(
            "Action probability:",
            f"{row['action_probability']:.3f}"
        )

    mistakes.to_csv(
        "handwritten_mistakes.csv",
        index=False
    )


def main():
    model, X_test, y_test = load_model_and_data()

    print(f"Handwritten test samples: {len(X_test)}")
    print(f"Action samples: {(y_test == 0).sum()}")
    print(f"Chat/knowledge samples: {(y_test == 1).sum()}")
    print()

    action_probabilities = get_action_probabilities(
        model,
        X_test
    )

    test_thresholds(
        action_probabilities,
        y_test
    )

    chosen_threshold = 0.4

    detailed_analysis(
        X_test,
        y_test,
        action_probabilities,
        threshold=chosen_threshold
    )


if __name__ == "__main__":
    main()