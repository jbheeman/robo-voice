import pandas as pd
import joblib
import matplotlib.pyplot as plt
import warnings

from tqdm import tqdm

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

# Optional: hide FutureWarnings from sklearn
warnings.filterwarnings("ignore", category=FutureWarning)

# Load data
df = pd.read_csv("intent_data.csv")

X = df["text"]
y = df["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Base pipeline
base_model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True)),
    ("clf", LogisticRegression(max_iter=3000))
])

# Hyperparameter grid
# I removed penalty tuning to avoid the warning spam.
# This still tunes realistic and useful parameters.
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf__min_df": [1, 2],
    "tfidf__sublinear_tf": [False, True],
    "tfidf__max_features": [None, 500, 1000, 2000],

    "clf__C": [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
    "clf__solver": ["lbfgs", "saga"],
    "clf__class_weight": [None, "balanced"],
}

params_list = list(ParameterGrid(param_grid))

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print(f"Testing {len(params_list)} hyperparameter combinations.")
print(f"Total model fits: {len(params_list) * cv.get_n_splits()}\n")

best_score = -1
best_params = None
results = []

# tqdm progress bar over hyperparameter combinations
for params in tqdm(params_list, desc="Tuning Logistic Regression"):
    fold_scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]

        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        model = clone(base_model)
        model.set_params(**params)

        model.fit(X_fold_train, y_fold_train)

        val_preds = model.predict(X_val)

        score = f1_score(
            y_val,
            val_preds,
            average="macro"
        )

        fold_scores.append(score)

    mean_score = sum(fold_scores) / len(fold_scores)

    results.append({
        "params": params,
        "mean_f1_macro": mean_score
    })

    if mean_score > best_score:
        best_score = mean_score
        best_params = params

print("\nBest CV score:")
print(best_score)

print("\nBest hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

# Train best model on full training set
best_model = clone(base_model)
best_model.set_params(**best_params)
best_model.fit(X_train, y_train)

# Evaluate on test set
test_preds = best_model.predict(X_test)

print("\nTest Set Classification Report:")
print(classification_report(y_test, test_preds))

# Confusion matrix
labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    test_preds,
    labels=labels
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(xticks_rotation=45)

plt.title("BELT Intent Router Confusion Matrix - Tuned Logistic Regression")
plt.tight_layout()
plt.show()

# Wrong predictions
test_results = pd.DataFrame({
    "text": X_test,
    "true_intent": y_test,
    "predicted_intent": test_preds
})

wrong = test_results[test_results["true_intent"] != test_results["predicted_intent"]]

print("\nWrong Predictions:")
if wrong.empty:
    print("No wrong predictions 🎉")
else:
    print(wrong.to_string(index=False))

# Save best model
joblib.dump(best_model, "belt_intent_router_best_logreg.joblib")

print("\nSaved best model as belt_intent_router_best_logreg.joblib")