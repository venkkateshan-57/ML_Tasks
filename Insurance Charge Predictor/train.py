import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_and_evaluate():
    """
    This function trains, evaluates, and saves the best regression model.
    """
    print("--- Starting Model Training ---")

    # 1. Load and Preprocess the Dataset
    df = pd.read_csv('data/insurance.csv')

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

    # Define features (X) and target (y)
    X = df.drop('charges', axis=1)
    y = df['charges']

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Define Models to Train
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42)
    }

    best_model = None
    best_r2_score = -1

    # 4. Train and Evaluate Each Model
    print("\n--- Model Performance ---")
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Model: {name}")
        print(f"  Mean Squared Error (MSE): {mse:.2f}")
        print(f"  R-squared (R²) Score: {r2:.4f}\n")

        # Check if this is the best model so far
        if r2 > best_r2_score:
            best_r2_score = r2
            best_model = model

    print(f"Best performing model is: {type(best_model).__name__} with R² score of {best_r2_score:.4f}")

    # 5. Save the Best Model
    # We also need to save the columns to ensure the app uses the same feature order
    model_data = {
        'model': best_model,
        'columns': X.columns.tolist()
    }

    with open('best_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("\nBest model and column list have been saved to 'best_model.pkl' ✨")


if __name__ == '__main__':
    train_and_evaluate()