from src.data_preprocessing import load_and_merge_data
from src.feature_engineering import clean_and_engineer
from src.model import LinearRegressionScratch
from src.train import train_models
from src.evaluate import evaluate_model, plot_results
import numpy as np

def main():
    df = load_and_merge_data()
    df = clean_and_engineer(df)

    X = df.drop("delivery_time_days", axis=1).values
    y = df["delivery_time_days"].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test, scaler, sklearn_model = train_models(X, y)

    # Scratch Model
    scratch_model = LinearRegressionScratch()
    scratch_model.fit(X_train, y_train)

    y_pred = scratch_model.predict(X_test)

    evaluate_model(y_test, y_pred)
    plot_results(scratch_model.cost_history, y_test, y_pred)

if __name__ == "__main__":
    main()