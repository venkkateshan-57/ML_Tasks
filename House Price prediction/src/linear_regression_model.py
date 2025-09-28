from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
# Import the StandardScaler
from sklearn.preprocessing import StandardScaler
import numpy as np




class HousePriceModel:
    """
    A class to represent the Ridge Regression model with feature scaling.
    """
    def __init__(self, alpha=1.0): # <-- ADD ALPHA HERE
        """
        Initializes the model and the scaler.
        """
        # Use the provided alpha value
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()


    def train(self, X_train, y_train):
        """
        Scales the training data and then trains the Ridge regression model.
        """
        # Fit the scaler to the training data and transform it
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train the model on the scaled data
        self.model.fit(X_train_scaled, y_train)
        print("Model training complete. ✨")

    def predict(self, X_test):
        """
        Scales the test data and then makes predictions.
        """
        # Use the already-fitted scaler to transform the test data
        X_test_scaled = self.scaler.transform(X_test)

        # Make predictions on the scaled test data
        return self.model.predict(X_test_scaled)

    def evaluate(self, y_test, y_pred):
        """
        Evaluates the model's performance.
        """
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return mse, r2