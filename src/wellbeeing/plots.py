import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import seaborn as sns

def plot_pca(components, clusters):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        components[:, 0],
        components[:, 1],
        c=clusters,
        s=5
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("KMeans clusters in PCA space")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.show()

def plot_dbscan(components, db_clusters):
    plt.figure(figsize=(8, 6))
    plt.scatter(
        components[:, 0],
        components[:, 1],
        c=db_clusters,
        s=5,
        cmap="viridis"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("DBSCAN clusters in PCA space")
    plt.colorbar(label="DBSCAN Cluster")
    plt.tight_layout()
    plt.show()

def plot_k_distance(X, min_samples=5):
    """
    Plot the k-distance graph to help determine the optimal epsilon for DBSCAN.
    """

    # Fit Nearest Neighbors model
    nbrs = NearestNeighbors(n_neighbors=min_samples)
    nbrs.fit(X)

    # Compute the distances to the nearest neighbors
    distances, indices = nbrs.kneighbors(X)

    # Sort the distances to the k-th nearest neighbor
    k_distances = np.sort(distances[:, -1])

    # Plot the k-distance graph
    plt.figure(figsize=(8, 6))
    plt.plot(k_distances)
    plt.title(f'k-Distance Graph (k={min_samples})')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_day_activity(df, date, feature="rms_mean"):
    day = df[
        df["date"] == pd.to_datetime(date).date()
    ].copy()

    if day.empty:
        print(f"No data found for {date}")
        return

    day = day.sort_values("datetime")

    plt.figure(figsize=(12, 5))

    plt.plot(
        day["datetime"],
        day[feature],
        marker="o",
        markersize=3,
        linewidth=1.5
    )

    plt.xlabel("Time of day")
    plt.ylabel(feature)
    plt.title(f"Bee acoustic activity — {date}")

    # Show a time label every 2 hours
    plt.gca().xaxis.set_major_locator(
        plt.matplotlib.dates.HourLocator(interval=2)
    )

    # Format labels as HH:MM
    plt.gca().xaxis.set_major_formatter(
        plt.matplotlib.dates.DateFormatter("%H:%M")
    )

    plt.xticks(rotation=45)

    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

def plot_average_day(df, feature="rms_mean"):
    """
    Plot the average daily pattern of a given feature across all days in
    the dataset. 
    """
    df = df.copy()

    # Assign each measurement to the nearest 10-minute mark
    df["time_bin"] = df["datetime"].dt.round("10min")

    # Extract time of day
    df["time_of_day"] = df["time_bin"].dt.time

    # Average measurements from different days
    avg = (
        df.groupby("time_of_day")[feature]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Convert time to decimal hours for plotting
    avg["hour"] = (
        avg["time_of_day"].apply(lambda x: x.hour + x.minute / 60)
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        avg["hour"],
        avg["mean"],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Mean activity"
    )

    plt.xlabel("Time of day")
    plt.ylabel(feature)
    plt.title(f"Average daily pattern — {feature}")

    plt.xticks(
        range(0, 24, 2),
        [f"{h:02d}:00" for h in range(0, 24, 2)]
    )

    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_overall_rms_by_time(df):
    """
    Plot overall RMS over time to visualize trends in acoustic activity.
    """
    df = df.copy()

    plt.figure(figsize=(12, 5))
    plt.scatter(
        df["datetime"],
        df["overall_rms"],
        s=5,
        alpha=0.4
    )

    plt.xlabel("Time")
    plt.ylabel("Overall RMS")
    plt.title("Overall RMS over time")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

def plot_median_rms_by_hour(df):
    """
    Plot the median overall RMS for each hour of the day to visualize
    daily patterns in acoustic activity.
    """
    hourly = (
        df.groupby("hour")["overall_rms"]
        .median()
        .reset_index()
    )

    plt.figure(figsize=(12, 5))
    plt.plot(
        hourly["hour"],
        hourly["overall_rms"],
        marker="o"
    )

    plt.xlabel("Hour of day")
    plt.ylabel("Median overall RMS")
    plt.title("Median overall RMS by hour of day")
    plt.xticks(range(24))
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

