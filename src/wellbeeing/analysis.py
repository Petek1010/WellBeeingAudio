import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN


import plots

def load_features(csv_path):
    """
    Load audio features from CSV file.
    """
    
    df = pd.read_csv(csv_path)

    print("Dataset shape:", df.shape)
    print("Unique recordings:", df["recording_id"].nunique())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:", df.duplicated().sum())
    print(df["recording_id"].head(10))

    # sample rate is constant
    X = df.drop(
        columns=["sample rate"]
    )

    return X

def add_datetime(df):
    """
    Add datetime, hour, day_of_week and date columns to the dataframe based on recording_id
    """
    df = df.copy()

    df["datetime"] = pd.to_datetime(
        df["recording_id"],
        format="ONLINE_%a_%b_%d_%H-%M-%S_%Y"
    )

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["date"] = df["datetime"].dt.date

    return df

def feature_statistics(df):
    """
    Basic statistics of features
    """

    print("\nFeature statistics:")
    print(df.describe())
    print("\nMissing values:")
    print(df.isnull().sum())

def correlation_matrix(df):
    """
    Show feature correlations
    """

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(12,10))
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90
    )
    plt.yticks(
        range(len(corr.columns)),
        corr.columns
    )
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.show()

def run_pca(X, n_components=2):
    """
    PCA projection of feature space
    """

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)
    print("Explained variance:", pca.explained_variance_ratio_)

    return components

def kmeans_clustering(X, n_clusters=3):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    )

    labels = model.fit_predict(X)

    print("Cluster counts:")
    print(np.unique(labels, return_counts=True))
    

    return labels

def dbscan_clustering(X, eps=1.0, min_samples=5):

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = model.fit_predict(X)
    n_noise = np.sum(labels == -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print("DBSCAN labels:")
    print(np.unique(labels, return_counts=True))
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")

    return labels


if __name__ == "__main__":

    csv_file = ("data/processed/audio_features.csv")
    df = load_features(csv_file)
    df = add_datetime(df)
    
    features = df.drop(columns=[
        "recording_id",
        "datetime",
        "hour",
        "day_of_week",
        "date"
    ])

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # PCA calculation
    components = run_pca(features_scaled)
    clusters = kmeans_clustering(features_scaled, n_clusters=3)  
    db_clusters = dbscan_clustering(features_scaled, eps=5.2, min_samples=5) 

    df["cluster"] = clusters
    df["db_cluster"] = db_clusters

 
    # Plots
    #plots.plot_day_activity(df, date="2025-06-12", feature="rms_mean")
    #plots.plot_average_day(df, feature="rms_mean")
    #plots.plot_average_day(df, feature="zcr_mean")
    #plots.plot_average_day(df, feature="flux_mean")
    #plots.plot_overall_rms_by_time(df)
    #plots.plot_median_rms_by_hour(df) 
    plots.plot_k_distance(features_scaled, min_samples=5) 
  



    # Short explanation of features:
    # RMS = measure of the average power of the signal, related to loudness

    # zcr = measure of the number of times the signal crosses zero, related
    # to noisiness. High zcr indicates a noisy signal, while low zcr indicates
    # a more tonal signal.

    # flux = measure of the change in the spectral content of the signal, 
    # related to the amount of movement or activity in the signal. High flux
    # indicates. High flux indicates a signal with rapid changes in frequency content, while low flux indicates a more stable signal.


    
   

    
