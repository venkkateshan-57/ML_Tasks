from src.data_processing import clean_and_prepare_data


def main():
    """
    Main function to execute the data cleaning and analysis pipeline.
    """
    print("--- Starting Titanic Data Pipeline ---")

    # Define file paths with the updated location
    raw_data_path = 'data/Titanic-Dataset.csv'  # <-- THIS LINE IS CHANGED
    processed_data_path = 'data/processed/cleaned_titanic.csv'

    # --- 1. Clean the Data ---
    cleaned_df = clean_and_prepare_data(raw_data_path, processed_data_path)

    # --- 2. Analyze the Cleaned Data ---
    print("\n--- Analysis of Cleaned Data ---")
    print("\nSummary statistics:")
    print(cleaned_df.describe())
    print("\nSurvival rate by passenger class:")
    print(cleaned_df.groupby('Pclass')['Survived'].value_counts(normalize=True))
    print("\n--- Pipeline Finished Successfully ---")


if __name__ == '__main__':
    main()