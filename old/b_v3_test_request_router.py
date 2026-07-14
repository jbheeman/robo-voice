import joblib
from pathlib import Path


MODEL_PATH = Path("request_router_model.joblib")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_PATH}. "
            "Run your training script first."
        )

    return joblib.load(MODEL_PATH)


def test(model, text_input: str) -> str:
    # Example prediction: [1, 0]
    prediction = model.predict([text_input])[0]

    # Confidence for each label
    probabilities = model.predict_proba([text_input])[0]

    navigation_requested = bool(prediction[0])
    simple_action_requested = bool(prediction[1])

    navigation_confidence = float(probabilities[0])
    simple_action_confidence = float(probabilities[1])

    navigation_result = "Yes" if navigation_requested else "No"
    simple_action_result = "Yes" if simple_action_requested else "No"

    analysis = f"""
Input:
{text_input}


Navigation requested: {navigation_result}
Navigation confidence: {navigation_confidence:.2%}

Simple action requested: {simple_action_result}
Simple action confidence: {simple_action_confidence:.2%}
""".strip()

    return analysis


def main():
    model = load_model()

    print("Request router loaded!")
    print("Type /quit to stop.\n")

    while True:
        text_input = input("You: ").strip()

        if text_input == "/quit":
            print("Stopping request router.")
            break

        if not text_input:
            print("Please enter some text.\n")
            continue

        result = test(model, text_input)

        print("\n" + result)
        print("-" * 50)


if __name__ == "__main__":
    main()