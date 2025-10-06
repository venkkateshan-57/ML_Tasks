import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor

def train_models():
    print("--- Starting Data Processing and Model Training ---")

    df = pd.read_csv('data/crime_dataset_india.csv')

    df['Date of Occurrence'] = pd.to_datetime(df['Date of Occurrence'])
    df['YEAR'] = df['Date of Occurrence'].dt.year
    df = df[df['YEAR'] != 2024]
    print("Filtered out incomplete data for the year 2024.")

    crime_pivot = df.pivot_table(index=['City', 'YEAR'], columns='Crime Description', values='Police Deployed', aggfunc='count').fillna(0)
    crime_pivot['TOTAL_CRIMES'] = crime_pivot.sum(axis=1)

    # --- SVM Model ---
    print("Training SVM model...")
    city_summary = crime_pivot.groupby('City').sum()
    crime_threshold = city_summary['TOTAL_CRIMES'].median()
    city_summary['IS_CRIME_PRONE'] = (city_summary['TOTAL_CRIMES'] > crime_threshold).astype(int)

    X_svm = city_summary.drop(['TOTAL_CRIMES', 'IS_CRIME_PRONE'], axis=1)
    y_svm = city_summary['IS_CRIME_PRONE']

    scaler = StandardScaler().fit(X_svm)
    X_svm_scaled = scaler.transform(X_svm)
    svm_model = SVC(kernel='linear', random_state=42).fit(X_svm_scaled, y_svm)

    # --- Random Forest Model ---
    print("Training Random Forest model...")
    regression_data = crime_pivot.reset_index()
    X_reg = regression_data[['YEAR', 'City']]
    X_reg = pd.get_dummies(X_reg, columns=['City'], drop_first=True)
    y_reg = regression_data['TOTAL_CRIMES']
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_reg, y_reg)

    # --- Save Files ---
    print("Saving models and necessary files...")
    with open('svm_model.pkl', 'wb') as f: pickle.dump(svm_model, f)
    with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
    with open('rf_model.pkl', 'wb') as f: pickle.dump(rf_model, f)
    city_summary.to_csv('data/city_summary_for_prediction.csv')

    print("--- Training complete. Models are saved. ---")

if __name__ == '__main__':
    train_models()