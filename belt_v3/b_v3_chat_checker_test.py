import joblib


def test(model, text):
    chat_probability = model.predict_proba([text])[0][1]
    prediction = int(chat_probability >= 0.5)

    return chat_probability


def main():
    model = joblib.load("chat_checker_model.joblib")
    text = input("Enter text: ")

    probability = test(model, text)
    print("Chat probability:", probability)


if __name__ == "__main__":
    main()