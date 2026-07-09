import joblib
import pandas as pd

model = joblib.load("intent_detector_logreg.joblib")

def predict_intent():
    print("Input:", end=" ")
    text = input()
    
    intent = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_
    scores = dict(zip(classes, probabilities))

    print("Predicted intent:", intent)

    print("\nConfidence scores:")
    for label, score in scores.items():
        print(f"{label}: {score:.3f}")
        
        
def find_low_conf():
    probs = model.predict_proba(X_test)
    classes = model.classes_

    low_confidence_rows = []

    for text, true_label, prob_row in zip(X_test, y_test, probs):
        best_index = prob_row.argmax()
        best_label = classes[best_index]
        best_score = prob_row[best_index]

        if best_score < 0.60:
            low_confidence_rows.append({
                "text": text,
                "true_intent": true_label,
                "predicted_intent": best_label,
                "confidence": best_score
            })

    low_conf_df = pd.DataFrame(low_confidence_rows)

    print("\nLow confidence predictions:")
    print(low_conf_df.sort_values("confidence").to_string(index=False))

predict_intent()


