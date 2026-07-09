import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

#Load csv
df = pd.read_csv("new_intent_data.csv")

X = df["text"]
y = df["intent"]

print("Dataset size:", len(df))
print("\nLabels:")
print(y.value_counts())


#Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


#Model Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),   # uses words and 2-word phrases
        min_df=1
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=1.0
    ))
])


#Train/Eval
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

display.plot(xticks_rotation=45)

plt.title("BELT Intent Router Confusion Matrix")
plt.tight_layout()
plt.show()


#show wrong predictions
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


# ------------------------------------------------------------
# 7. Save trained model
# ------------------------------------------------------------

joblib.dump(model, "belt_intent_router_5_labels.joblib")

print("\nSaved model as intent_detector_logreg.joblib")