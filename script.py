import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================
# CONFIGURATION
# ==============================
CSV_FILE = "data.csv"

ACC_COLS = ["ax", "ay", "az"]
GYRO_COLS = ["gx", "gy", "gz"]
MAG_COLS = ["mx", "my", "mz"]

ROLLING_ACTIVITY_DAYS = 3    # dynamic activity classification
ESTRUS_BASELINE_DAYS = 7     # estrus baseline window

RELATIVE_STD_MULTIPLIER = 1.5
ABSOLUTE_ACTIVITY_MULTIPLIER = 1.25
WALKING_THRESHOLD = 0.30
DOMINANCE_RATIO_THRESHOLD = 1.15

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(CSV_FILE)
df["time"] = pd.to_datetime(df["time"])
df["date"] = df["time"].dt.date

# ==============================
# SENSOR MAGNITUDES
# ==============================
df["acc_mag"] = np.sqrt((df[ACC_COLS] ** 2).sum(axis=1))
df["gyro_mag"] = np.sqrt((df[GYRO_COLS] ** 2).sum(axis=1))
df["mag_mag"] = np.sqrt((df[MAG_COLS] ** 2).sum(axis=1))

df["movement_score"] = df["acc_mag"] + df["gyro_mag"] + df["mag_mag"]

# ==============================
# DYNAMIC ACTIVITY THRESHOLDS
# ==============================
df["low_thresh"] = (
    df["movement_score"]
    .rolling(ROLLING_ACTIVITY_DAYS, min_periods=1)
    .quantile(0.30)
)

df["high_thresh"] = (
    df["movement_score"]
    .rolling(ROLLING_ACTIVITY_DAYS, min_periods=1)
    .quantile(0.70)
)

df["mid_thresh"] = (df["low_thresh"] + df["high_thresh"]) / 2

# ==============================
# ACTIVITY CLASSIFICATION
# ==============================
def classify_activity(row):
    if row["movement_score"] < row["low_thresh"]:
        return "resting"
    elif row["movement_score"] < row["mid_thresh"]:
        return "eating"
    elif row["movement_score"] < row["high_thresh"]:
        return "ruminating"
    else:
        return "walking"

df["activity_state"] = df.apply(classify_activity, axis=1)

# ==============================
# DAILY ACTIVITY FRACTIONS
# ==============================
daily_activity = (
    df.groupby(["date", "activity_state"])
    .size()
    .unstack(fill_value=0)
)

daily_activity["total"] = daily_activity.sum(axis=1)

for state in ["resting", "eating", "ruminating", "walking"]:
    daily_activity[f"{state}_fraction"] = (
        daily_activity[state] / daily_activity["total"]
    )

# ==============================
# DAILY SENSOR FEATURES
# ==============================
daily_features = df.groupby("date").agg(
    acc_mean=("acc_mag", "mean"),
    acc_std=("acc_mag", "std"),
    gyro_mean=("gyro_mag", "mean"),
    gyro_std=("gyro_mag", "std"),
    mag_mean=("mag_mag", "mean"),
    mag_std=("mag_mag", "std"),
    amb_std=("amb", "std"),
    obj_std=("obj", "std")
)

daily_features = daily_features.join(
    daily_activity[
        ["resting_fraction", "eating_fraction",
         "ruminating_fraction", "walking_fraction"]
    ]
)

# ==============================
# PCA ACTIVITY SCORE
# ==============================
scaler = StandardScaler()
scaled_features = scaler.fit_transform(
    daily_features[
        ["acc_mean", "acc_std",
         "gyro_mean", "gyro_std",
         "mag_mean", "mag_std"]
    ]
)

pca = PCA(n_components=1)
daily_features["activity_score"] = pca.fit_transform(scaled_features).flatten()

# ==============================
# ROBUST ESTRUS DETECTION
# ==============================
daily_features["baseline_mean"] = (
    daily_features["activity_score"]
    .rolling(ESTRUS_BASELINE_DAYS, min_periods=1)
    .mean()
    .shift(1)
)

daily_features["baseline_std"] = (
    daily_features["activity_score"]
    .rolling(ESTRUS_BASELINE_DAYS, min_periods=1)
    .std()
    .shift(1)
    .fillna(0)
)

# ---- Multi-condition candidate ----
cond_relative = (
    daily_features["activity_score"]
    > daily_features["baseline_mean"]
    + RELATIVE_STD_MULTIPLIER * daily_features["baseline_std"]
)

cond_absolute = (
    daily_features["activity_score"]
    > daily_features["baseline_mean"] * ABSOLUTE_ACTIVITY_MULTIPLIER
)

cond_walking = (
    daily_features["walking_fraction"] > WALKING_THRESHOLD
)

daily_features["estrus_candidate"] = (
    cond_relative & cond_absolute & cond_walking
)

# ---- Final dominance check (NO forced estrus) ----
daily_features["estrus_confirmed"] = False

candidates = daily_features[daily_features["estrus_candidate"]]

if len(candidates) > 0:
    top_two = daily_features["activity_score"].nlargest(2)

    if len(top_two) == 2:
        dominance_ratio = top_two.iloc[0] / top_two.iloc[1]
    else:
        dominance_ratio = 0

    if dominance_ratio > DOMINANCE_RATIO_THRESHOLD:
        estrus_day = top_two.index[0]
        daily_features.loc[estrus_day, "estrus_confirmed"] = True

# ==============================
# OUTPUT
# ==============================
print("\n======================================")
print(" 🐄 DAILY CATTLE ACTIVITY & ESTRUS REPORT")
print("======================================\n")

for date, row in daily_features.iterrows():
    print(
        f"{date} | Activity: {row['activity_score']:.2f} | "
        f"Rest: {row['resting_fraction']:.2f} | "
        f"Eat: {row['eating_fraction']:.2f} | "
        f"Rum: {row['ruminating_fraction']:.2f} | "
        f"Walk: {row['walking_fraction']:.2f} | "
        f"Estrus: {row['estrus_confirmed']}"
    )

print("\n======================================")
print(" 📊 SUMMARY")
print("======================================")

if daily_features["estrus_confirmed"].any():
    print("Estrus detected on:", daily_features[daily_features["estrus_confirmed"]].index[0])
else:
    print("No estrus detected in this dataset")

print("\n======================================")
print(" 🐄 FINAL DAILY FEATURES")
print("======================================\n")

print(
    daily_features[
        [
            "activity_score",
            "resting_fraction",
            "eating_fraction",
            "ruminating_fraction",
            "walking_fraction",
            "estrus_confirmed"
        ]
    ]
)
