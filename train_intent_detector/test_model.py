import joblib

model = joblib.load("belt_intent_router_best_logreg.joblib")

def predict_intent(text):
    predicted_intent = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    confidence_scores = dict(zip(classes, probabilities))

    return predicted_intent, confidence_scores


text = "bro"

intent, scores = predict_intent(text)

print("Input:", text)
print("Predicted intent:", intent)

print("\nConfidence scores:")
for label, score in scores.items():
    print(f"{label}: {score:.3f}")