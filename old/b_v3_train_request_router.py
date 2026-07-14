import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score


# Load the training data
df = pd.read_csv("request_router_data.csv")

X = df["text"]
y = df[["navigation", "simple_action"]]


# Create a combined category only for splitting purposes.
# This helps ensure that none/navigation/action/both are represented
# in both the training and testing sets.
split_categories = (
    df["navigation"].astype(str)
    + df["simple_action"].astype(str)
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=split_categories
)


model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1
        )
    ),
    (
        "classifier",
        OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    )
])


model.fit(X_train, y_train)

predictions = model.predict(X_test)


print(
    classification_report(
        y_test,
        predictions,
        target_names=["navigation", "simple_action"],
        zero_division=0
    )
)

print(
    "Exact-match accuracy:",
    accuracy_score(y_test, predictions)
)


joblib.dump(model, "request_router_model.joblib")

print("Model saved.")