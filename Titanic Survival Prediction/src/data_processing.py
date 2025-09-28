import pandas as pd
import os


def clean_and_prepare_data(input_path, output_path):
    """
    Loads the raw Titanic data, cleans it, and saves the processed file.
    """
    df = pd.read_csv(input_path)
    print("Raw data loaded successfully.")

    # --- 1. Handle Missing Values ---
    age_median = df['Age'].median()
    # This is the updated line - it's more explicit and future-proof
    df['Age'] = df['Age'].fillna(age_median)

    df.drop('Cabin', axis=1, inplace=True)
    df.dropna(subset=['Embarked'], inplace=True)
    print("Missing values handled.")

    # --- 2. Handle Duplicates ---
    df.drop_duplicates(inplace=True)
    print("Duplicates handled.")

    # --- 3. Save the Cleaned Data ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

    return df