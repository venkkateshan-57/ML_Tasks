import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re

#  Loading the Dataset
df = pd.read_csv("data/emails.csv")
print(f"Dataset shape: {df.shape}")

# Features and label
feature_cols = [col for col in df.columns if col not in ['Email No.', 'Prediction']]
X = df[feature_cols]
y = df['Prediction']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "spam_classifier_model.pkl")
print("Model saved!")

# Function to convert raw email into numeric features
def preprocess_email(email_text, feature_columns):
    """
    Converts raw email text into numeric feature vector based on dataset columns.
    Only counts words that exist in the feature_columns.
    """
    email_text = email_text.lower()
    # Remove non-alphabet characters
    words = re.findall(r'\b[a-z]+\b', email_text)
    feature_dict = dict.fromkeys(feature_columns, 0)
    for word in words:
        if word in feature_dict:
            feature_dict[word] += 1
    return pd.DataFrame([feature_dict])

# Function to predict raw email
def predict_email(email_text):
    vect_email = preprocess_email(email_text, feature_cols)
    prediction = model.predict(vect_email)[0]
    return "Spam" if prediction == 1 else "Not Spam"

#Interactive input
while True:
    print("\nType/paste an email to check for Spam (or type 'exit' to quit):")
    new_email = input()
    if new_email.lower() == 'exit':
        break
    result = predict_email(new_email)
    print("Prediction:", result)
