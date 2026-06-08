import sys

import pandas as pd

from model_code.terrain_classifier import predict_dataframe


DEFAULT_CSV_PATH = "data.csv"
DEFAULT_ROLL_NO = "B25331"


def evaluate(csv_path, roll_no):
    df = pd.read_csv(csv_path)
    predictions = predict_dataframe(df)
    out_df = pd.DataFrame({"prediction": predictions})
    out_df.to_csv(f"{roll_no}.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        evaluate(sys.argv[1], sys.argv[2])
    else:
        evaluate(DEFAULT_CSV_PATH, DEFAULT_ROLL_NO)
