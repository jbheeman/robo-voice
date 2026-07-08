import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    ))
])

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, preds))

# Confusion matrix
labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    preds,
    labels=labels
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(
    xticks_rotation=45
)

plt.title("BELT Intent Router Confusion Matrix")
plt.tight_layout()
plt.show()

# Show wrong predictions clearly
results = pd.DataFrame({
    "text": X_test,
    "true_intent": y_test,
    "predicted_intent": preds
})

wrong = results[results["true_intent"] != results["predicted_intent"]]

print("\nWrong Predictions:")
if wrong.empty:
    print("No wrong predictions 🎉")
else:
    print(wrong.to_string(index=False))

joblib.dump(model, "belt_intent_router.joblib")