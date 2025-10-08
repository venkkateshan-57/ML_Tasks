import pandas as pd
import numpy as np
import datetime as dt
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def train_rfm_segmentation_model():
    print("--- Starting RFM Customer Segmentation Training ---")

    # 1. Load and Clean Data
    df = pd.read_excel('data/Online Retail.xlsx')
    df.dropna(subset=['CustomerID'], inplace=True)  # Critical for grouping
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]  # Remove returns and invalid prices
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print("Data loaded and cleaned.")

    # 2. Calculate RFM Features
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda date: (snapshot_date - date.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'}, inplace=True)
    print("RFM features calculated.")

    # 3. Handle Skewness and Scale Data
    # Log transformation to handle skewed data, then scale
    rfm_log = np.log1p(rfm)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    # 4. Determine Optimal Clusters (Elbow Method)
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('The Elbow Method for RFM')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('WCSS')
    plt.savefig('rfm_elbow_plot.png')
    print("Elbow plot saved. Check this plot to determine the best 'k'. Let's assume k=4.")

    # 5. Apply K-Means with Optimal k
    optimal_k = 4
    kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    rfm['Cluster'] = kmeans_final.fit_predict(rfm_scaled)
    print(f"K-Means clustering applied with k={optimal_k}.")

    # 6. Use PCA for Visualization
    pca = PCA(n_components=2)
    rfm_pca = pca.fit_transform(rfm_scaled)
    rfm['PCA1'] = rfm_pca[:, 0]
    rfm['PCA2'] = rfm_pca[:, 1]

    # 7. Save Artifacts
    with open('kmeans_rfm_model.pkl', 'wb') as f: pickle.dump(kmeans_final, f)
    with open('scaler_rfm.pkl', 'wb') as f: pickle.dump(scaler, f)
    rfm.to_csv('data/customers_rfm_segmented.csv')

    print("\n--- Training complete. RFM model, scaler, and data have been saved. ---")


if __name__ == '__main__':
    train_rfm_segmentation_model()