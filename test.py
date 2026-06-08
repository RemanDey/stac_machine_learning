import lunadem

from model_code.terrain_classifier import predict_dataframe


def main():
    print("Loading current live telemetry data...")
    df = lunadem.get_current_data()

    predictions = predict_dataframe(df)

    print(f"Predicted {len(predictions)} live terrain probabilities.")
    print("Sample class-1 probabilities on live data:")
    for index, probability in enumerate(predictions[:5]):
        print(f"Row {index} -> Model Probability: {probability:.6f}")


if __name__ == "__main__":
    main()
