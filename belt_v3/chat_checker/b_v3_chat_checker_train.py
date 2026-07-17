import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def input_load(filename="chat_checker_data.csv"):
    df = pd.read_csv(filename)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    return X_train, X_test, y_train, y_test


def train(X_train, y_train):
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    model.fit(X_train, y_train)
    return model


def test_analysis(model, X_test, y_test):
    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["action", "chat"]
        )
    )


def main():
    X_train, X_test, y_train, y_test = input_load()

    model = train(X_train, y_train)

    test_analysis(model, X_test, y_test)

    joblib.dump(model, "chat_checker_model.joblib")


if __name__ == "__main__":
    main()