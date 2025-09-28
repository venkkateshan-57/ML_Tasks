import pandas as pd
from sklearn.model_selection import train_test_split
# We no longer need 'os' or 'fetch_california_housing'
# from sklearn.datasets import fetch_california_housing
# import os

# Import our custom model class from the src folder
from src.linear_regression_model import HousePriceModel


def prepare_data(data_path="data/train.csv"):
    """
    Loads, cleans, and prepares the dataset for modeling, including one-hot encoding.

    Args:
        data_path (str): The path to the CSV data file.

    Returns:
        A tuple of (features, target) pandas DataFrames/Series.
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    target_column = 'SalePrice'
    df.dropna(subset=[target_column], inplace=True)

    # Separate features and target BEFORE any processing
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # --- NEW: ONE-HOT ENCODING ---
    # Convert categorical columns into numeric format.
    # This will automatically find text-based columns and encode them.
    X = pd.get_dummies(X,drop_first=True)

    # --- UPDATED: FILL MISSING VALUES ---
    # Now that all data is numeric, we can fill any missing values.
    # This handles NaNs from the original numeric columns AND any that might appear.
    X.fillna(X.median(), inplace=True)

    return X, y


def main():
    """
    Main function to run the final house price prediction model.
    """
    print("Starting the House Price Prediction pipeline...")

    # 1. Prepare data
    X, y = prepare_data("data/train.csv")

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Create and train the final model using the best alpha found
    best_alpha = 50.0
    print(f"\n--- Training final model with best alpha: {best_alpha} ---")

    model = HousePriceModel(alpha=best_alpha)
    model.train(X_train, y_train)

    # 4. Make predictions
    predictions = model.predict(X_test)

    # 5. Evaluate the final model
    mse, r2 = model.evaluate(y_test, predictions)
    print("\n--- Final Model Evaluation Results ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    print("------------------------------------")


# ... (prepare_data function and the if __name__ == "__main__": part are the same)

if __name__ == "__main__":
    main()