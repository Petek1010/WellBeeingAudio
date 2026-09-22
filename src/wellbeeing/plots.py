import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_over_all_time(df, feature):

    data = df.sort_values("datetime")

    plt.figure(figsize=(14, 5))

    plt.plot(
        data["datetime"],
        data[feature],
        linewidth=0.5
    )

    plt.xlabel("Time")
    plt.ylabel(feature)
    plt.title(f"{feature} over time")

    plt.tight_layout()
    plt.show()

def plot_day(df, date, feature="rms_mean"):

    day = df[
        df["date"] == pd.to_datetime(date).date()
    ].copy()

    if day.empty:
        print(f"No data found for {date}")
        return

    day = day.sort_values("datetime")

    normal = day[day["anomaly"] == 1]
    anomalies = day[day["anomaly"] == -1]

    plt.figure(figsize=(14, 5))

    plt.plot(
        normal["datetime"],
        normal[feature],
        ".",
        markersize=4,
        label="Normal"
    )

    plt.scatter(
        anomalies["datetime"],
        anomalies[feature],
        s=40,
        label="Anomaly"
    )

    plt.xlabel("Time")
    plt.ylabel(feature)
    plt.title(f"{feature} — {date}")

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_average_day(df, feature="rms_mean"):

    hourly = (
        df.groupby("hour")[feature]
        .agg(["mean", "std"])
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        hourly.index,
        hourly["mean"],
        "o-",
        label="Mean"
    )

    plt.fill_between(
        hourly.index,
        hourly["mean"] - hourly["std"],
        hourly["mean"] + hourly["std"],
        alpha=0.2,
        label="±1 SD"
    )

    plt.xlabel("Hour of day")
    plt.ylabel(feature)
    plt.title(f"Average daily pattern — {feature}")

    plt.xticks(range(24))
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_anomaly_rate_by_hour(df):

    hourly = (
        df.groupby("hour")["anomaly"]
        .apply(lambda x: (x == -1).mean() * 100)
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        hourly.index,
        hourly.values
    )

    plt.xlabel("Hour of day")
    plt.ylabel("Anomalies (%)")
    plt.title("Anomaly rate by hour")

    plt.xticks(range(24))

    plt.tight_layout()
    plt.show()

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

def plot_feature_by_hour(df, feature):
    plt.figure(figsize=(12, 5))

    df.boxplot(
        column=feature,
        by="hour",
        grid=False
    )

    plt.title(f"{feature} by hour")
    plt.suptitle("")
    plt.xlabel("Hour")
    plt.ylabel(feature)

    plt.tight_layout()
    plt.show()

def plot_typical_day(df, feature):

    hourly = (
        df.groupby("hour")[feature]
        .agg(["median", "mean", "std"])
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        hourly.index,
        hourly["median"],
        marker="o",
        label="Median"
    )

    plt.fill_between(
        hourly.index,
        hourly["mean"] - hourly["std"],
        hourly["mean"] + hourly["std"],
        alpha=0.2,
        label="±1 std"
    )

    plt.xlabel("Hour")
    plt.ylabel(feature)
    plt.title(f"Typical daily pattern — {feature}")

    plt.xticks(range(24))
    plt.legend()

    plt.tight_layout()
    plt.show()