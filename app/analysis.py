import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from app.config import get_settings, format_bst, now_bst
from datetime import datetime

settings = get_settings()

ACC_COLS = ["ax", "ay", "az"]
GYRO_COLS = ["gx", "gy", "gz"]
MAG_COLS = ["mx", "my", "mz"]


def preprocess_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw sensor data:
    - Handle invalid readings (amb/obj = -0.001)
    - Clean NaN values
    """
    df = df.copy()
    
    # Replace garbage temperature readings
    # (device sometimes emits negative values; also guard against extreme outliers)
    if "amb" in df.columns:
        df.loc[(df["amb"] < 0) | (df["amb"] > 80), "amb"] = np.nan
    if "obj" in df.columns:
        df.loc[(df["obj"] < 0) | (df["obj"] > 80), "obj"] = np.nan
    
    # Forward/back fill temperature values if present
    if "amb" in df.columns:
        df["amb"] = df["amb"].ffill().bfill()
    if "obj" in df.columns:
        df["obj"] = df["obj"].ffill().bfill()
    
    return df


def analyze_estrus(df: pd.DataFrame, config: dict = None) -> dict:
    """
    Main analysis function - processes sensor data and detects estrus.
    Returns a dictionary with all results.
    
    Args:
        df: DataFrame with sensor data
        config: Optional dict to override default settings
    """
    if df.empty:
        return {
            "error": "No data available",
            "daily_data": [],
            "estrus_detected": False,
            "estrus_date": None,
            "total_days": 0,
            "total_readings": 0,
            "summary": {
                "avg_activity_score": 0,
                "avg_walking_fraction": 0,
                "avg_resting_fraction": 0
            },
            "last_updated": format_bst(now_bst())
        }
    
    # Use config or default settings
    cfg = config or {}
    rolling_days = cfg.get("rolling_activity_days", settings.rolling_activity_days)
    baseline_days = cfg.get("estrus_baseline_days", settings.estrus_baseline_days)
    relative_mult = cfg.get("relative_std_multiplier", settings.relative_std_multiplier)
    absolute_mult = cfg.get("absolute_activity_multiplier", settings.absolute_activity_multiplier)
    walk_thresh = cfg.get("walking_threshold", settings.walking_threshold)
    dominance_thresh = cfg.get("dominance_ratio_threshold", settings.dominance_ratio_threshold)
    
    # Preprocess
    df = preprocess_sensor_data(df)
    total_readings = len(df)
    
    # ==============================
    # SENSOR MAGNITUDES
    # ==============================
    df["acc_mag"] = np.sqrt((df[ACC_COLS].astype(float) ** 2).sum(axis=1))
    df["gyro_mag"] = np.sqrt((df[GYRO_COLS].astype(float) ** 2).sum(axis=1))
    df["mag_mag"] = np.sqrt((df[MAG_COLS].astype(float) ** 2).sum(axis=1))
    df["movement_score"] = df["acc_mag"] + df["gyro_mag"] + df["mag_mag"]

    # ==============================
    # DYNAMIC ACTIVITY THRESHOLDS
    # ==============================
    df["low_thresh"] = (
        df["movement_score"]
        .rolling(rolling_days, min_periods=1)
        .quantile(0.30)
    )
    df["high_thresh"] = (
        df["movement_score"]
        .rolling(rolling_days, min_periods=1)
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
        if state in daily_activity.columns:
            daily_activity[f"{state}_fraction"] = (
                daily_activity[state] / daily_activity["total"]
            )
        else:
            daily_activity[f"{state}_fraction"] = 0.0

    # ==============================
    # DAILY SENSOR FEATURES
    # ==============================
    agg_dict = {
        "acc_mean": ("acc_mag", "mean"),
        "acc_std": ("acc_mag", "std"),
        "gyro_mean": ("gyro_mag", "mean"),
        "gyro_std": ("gyro_mag", "std"),
        "mag_mean": ("mag_mag", "mean"),
        "mag_std": ("mag_mag", "std"),
    }
    
    # Add temperature aggregations if available
    if "amb" in df.columns and df["amb"].notna().any():
        agg_dict["amb_mean"] = ("amb", "mean")
        agg_dict["amb_std"] = ("amb", "std")
    if "obj" in df.columns and df["obj"].notna().any():
        agg_dict["obj_mean"] = ("obj", "mean")
        agg_dict["obj_std"] = ("obj", "std")
    
    daily_features = df.groupby("date").agg(**agg_dict)

    # Join activity fractions
    daily_features = daily_features.join(
        daily_activity[
            ["resting_fraction", "eating_fraction",
             "ruminating_fraction", "walking_fraction"]
        ]
    )

    # ==============================
    # PCA ACTIVITY SCORE
    # ==============================
    # IMPORTANT:
    # - `activity_score_raw` is a centered PCA component score (often averages ~0).
    # - `activity_score` is a normalized 0..100 score for UI display.
    if len(daily_features) < 2:
        daily_features["activity_score_raw"] = 0.0
        daily_features["activity_score"] = 0.0
    else:
        feature_cols = ["acc_mean", "acc_std", "gyro_mean", "gyro_std", "mag_mean", "mag_std"]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(
            daily_features[feature_cols].fillna(0)
        )
        pca = PCA(n_components=1)
        raw = pca.fit_transform(scaled_features).flatten()
        daily_features["activity_score_raw"] = raw

        raw_min = float(np.nanmin(raw)) if len(raw) else 0.0
        raw_max = float(np.nanmax(raw)) if len(raw) else 0.0
        if raw_max == raw_min:
            daily_features["activity_score"] = 0.0
        else:
            daily_features["activity_score"] = ((raw - raw_min) / (raw_max - raw_min)) * 100.0

    # ==============================
    # ESTRUS DETECTION
    # ==============================
    daily_features["baseline_mean"] = (
        daily_features["activity_score_raw"]
        .rolling(baseline_days, min_periods=1)
        .mean()
        .shift(1)
    )
    daily_features["baseline_std"] = (
        daily_features["activity_score_raw"]
        .rolling(baseline_days, min_periods=1)
        .std()
        .shift(1)
        .fillna(0)
    )

    # Multi-condition detection
    cond_relative = (
        daily_features["activity_score_raw"]
        > daily_features["baseline_mean"]
        + relative_mult * daily_features["baseline_std"]
    )
    cond_absolute = (
        daily_features["activity_score_raw"]
        > daily_features["baseline_mean"] * absolute_mult
    )
    cond_walking = (
        daily_features["walking_fraction"] > walk_thresh
    )

    daily_features["estrus_candidate"] = cond_relative & cond_absolute & cond_walking
    daily_features["estrus_confirmed"] = False

    estrus_day = None
    candidates = daily_features[daily_features["estrus_candidate"]]

    if len(candidates) > 0:
        top_two = daily_features["activity_score_raw"].nlargest(2)
        if len(top_two) == 2:
            if top_two.iloc[1] != 0:
                dominance_ratio = top_two.iloc[0] / top_two.iloc[1]
            else:
                dominance_ratio = float('inf')
        else:
            dominance_ratio = 0

        if dominance_ratio > dominance_thresh:
            estrus_day = top_two.index[0]
            daily_features.loc[estrus_day, "estrus_confirmed"] = True

    # ==============================
    # PREPARE OUTPUT
    # ==============================
    daily_features = daily_features.reset_index()
    daily_features["date"] = daily_features["date"].astype(str)
    
    # Fill NaN values for JSON serialization
    daily_features = daily_features.fillna(0)

    # Build daily data list
    daily_data = []
    for _, row in daily_features.iterrows():
        daily_data.append({
            "date": row["date"],
            "activity_score": float(row["activity_score"]),
            "activity_score_raw": float(row.get("activity_score_raw", 0)),
            "resting_fraction": float(row["resting_fraction"]),
            "eating_fraction": float(row["eating_fraction"]),
            "ruminating_fraction": float(row["ruminating_fraction"]),
            "walking_fraction": float(row["walking_fraction"]),
            "estrus_confirmed": bool(row["estrus_confirmed"]),
            "amb_mean": float(row.get("amb_mean", 0)),
            "obj_mean": float(row.get("obj_mean", 0))
        })

    result = {
        "daily_data": daily_data,
        "estrus_detected": bool(daily_features["estrus_confirmed"].any()),
        "estrus_date": str(estrus_day) if estrus_day else None,
        "total_days": len(daily_features),
        "total_readings": total_readings,
        "summary": {
            "avg_activity_score": float(daily_features["activity_score"].mean()),
            "avg_activity_score_raw": float(daily_features["activity_score_raw"].mean()) if "activity_score_raw" in daily_features.columns else 0.0,
            "avg_walking_fraction": float(daily_features["walking_fraction"].mean()),
            "avg_resting_fraction": float(daily_features["resting_fraction"].mean()),
            "max_activity_score": float(daily_features["activity_score"].max()),
            "min_activity_score": float(daily_features["activity_score"].min()),
            "max_activity_score_raw": float(daily_features["activity_score_raw"].max()) if "activity_score_raw" in daily_features.columns else 0.0,
            "min_activity_score_raw": float(daily_features["activity_score_raw"].min()) if "activity_score_raw" in daily_features.columns else 0.0,
        },
        "last_updated": format_bst(now_bst())
    }

    return result


def compute_daily_metrics(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Compute per-day aggregate metrics from raw flattened sensor data.

    Output columns are designed to be stored in Mongo (daily_metrics) and later
    used to run estrus detection quickly without re-reading all raw rows.
    """
    if df.empty:
        return pd.DataFrame()

    cfg = config or {}
    rolling_days = cfg.get("rolling_activity_days", settings.rolling_activity_days)

    df = preprocess_sensor_data(df)

    df["acc_mag"] = np.sqrt((df[ACC_COLS].astype(float) ** 2).sum(axis=1))
    df["gyro_mag"] = np.sqrt((df[GYRO_COLS].astype(float) ** 2).sum(axis=1))
    df["mag_mag"] = np.sqrt((df[MAG_COLS].astype(float) ** 2).sum(axis=1))
    df["movement_score"] = df["acc_mag"] + df["gyro_mag"] + df["mag_mag"]

    df = df.sort_values("time").reset_index(drop=True)

    df["low_thresh"] = df["movement_score"].rolling(rolling_days, min_periods=1).quantile(0.30)
    df["high_thresh"] = df["movement_score"].rolling(rolling_days, min_periods=1).quantile(0.70)
    df["mid_thresh"] = (df["low_thresh"] + df["high_thresh"]) / 2

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

    daily_activity = df.groupby(["date", "activity_state"]).size().unstack(fill_value=0)
    daily_activity["total"] = daily_activity.sum(axis=1)
    for state in ["resting", "eating", "ruminating", "walking"]:
        if state in daily_activity.columns:
            daily_activity[f"{state}_fraction"] = daily_activity[state] / daily_activity["total"]
        else:
            daily_activity[f"{state}_fraction"] = 0.0

    daily = df.groupby("date").agg(
        acc_mean=("acc_mag", "mean"),
        acc_std=("acc_mag", "std"),
        gyro_mean=("gyro_mag", "mean"),
        gyro_std=("gyro_mag", "std"),
        mag_mean=("mag_mag", "mean"),
        mag_std=("mag_mag", "std"),
        amb_mean=("amb", "mean") if "amb" in df.columns else ("movement_score", "mean"),
        obj_mean=("obj", "mean") if "obj" in df.columns else ("movement_score", "mean"),
        total_readings=("movement_score", "size"),
    )

    # If amb/obj didn't exist, drop placeholder columns
    if "amb" not in df.columns:
        daily = daily.drop(columns=["amb_mean"], errors="ignore")
    if "obj" not in df.columns:
        daily = daily.drop(columns=["obj_mean"], errors="ignore")

    daily = daily.join(
        daily_activity[[
            "resting_fraction",
            "eating_fraction",
            "ruminating_fraction",
            "walking_fraction",
        ]]
    )

    daily = daily.reset_index()
    daily["date"] = daily["date"].astype(str)
    return daily.fillna(0)


def analyze_estrus_from_daily(daily_df: pd.DataFrame, config: dict = None) -> dict:
    """Run estrus detection from per-day aggregate rows (daily_metrics)."""
    if daily_df is None or daily_df.empty:
        return {
            "error": "No data available",
            "daily_data": [],
            "estrus_detected": False,
            "estrus_date": None,
            "total_days": 0,
            "total_readings": 0,
            "summary": {
                "avg_activity_score": 0,
                "avg_activity_score_raw": 0,
                "avg_walking_fraction": 0,
                "avg_resting_fraction": 0,
            },
            "last_updated": format_bst(now_bst()),
        }

    cfg = config or {}
    baseline_days = cfg.get("estrus_baseline_days", settings.estrus_baseline_days)
    relative_mult = cfg.get("relative_std_multiplier", settings.relative_std_multiplier)
    absolute_mult = cfg.get("absolute_activity_multiplier", settings.absolute_activity_multiplier)
    walk_thresh = cfg.get("walking_threshold", settings.walking_threshold)
    dominance_thresh = cfg.get("dominance_ratio_threshold", settings.dominance_ratio_threshold)

    df = daily_df.copy()
    # Ensure required cols exist
    for c in [
        "acc_mean", "acc_std", "gyro_mean", "gyro_std", "mag_mean", "mag_std",
        "resting_fraction", "eating_fraction", "ruminating_fraction", "walking_fraction",
    ]:
        if c not in df.columns:
            df[c] = 0.0

    df = df.sort_values("date").reset_index(drop=True)

    # PCA score from daily features
    feature_cols = ["acc_mean", "acc_std", "gyro_mean", "gyro_std", "mag_mean", "mag_std"]
    if len(df) < 2:
        df["activity_score_raw"] = 0.0
        df["activity_score"] = 0.0
    else:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[feature_cols].fillna(0))
        pca = PCA(n_components=1)
        raw = pca.fit_transform(scaled).flatten()
        df["activity_score_raw"] = raw

        raw_min = float(np.nanmin(raw)) if len(raw) else 0.0
        raw_max = float(np.nanmax(raw)) if len(raw) else 0.0
        if raw_max == raw_min:
            df["activity_score"] = 0.0
        else:
            df["activity_score"] = ((raw - raw_min) / (raw_max - raw_min)) * 100.0

    df["baseline_mean"] = df["activity_score_raw"].rolling(baseline_days, min_periods=1).mean().shift(1)
    df["baseline_std"] = df["activity_score_raw"].rolling(baseline_days, min_periods=1).std().shift(1).fillna(0)

    cond_relative = df["activity_score_raw"] > df["baseline_mean"] + relative_mult * df["baseline_std"]
    cond_absolute = df["activity_score_raw"] > df["baseline_mean"] * absolute_mult
    cond_walking = df["walking_fraction"] > walk_thresh

    df["estrus_candidate"] = cond_relative & cond_absolute & cond_walking
    df["estrus_confirmed"] = False

    estrus_day = None
    candidates = df[df["estrus_candidate"]]
    if len(candidates) > 0:
        top_two = df["activity_score_raw"].nlargest(2)
        dominance_ratio = 0
        if len(top_two) == 2:
            dominance_ratio = float('inf') if top_two.iloc[1] == 0 else (top_two.iloc[0] / top_two.iloc[1])
        if dominance_ratio > dominance_thresh:
            estrus_day = df.loc[top_two.index[0], "date"]
            df.loc[top_two.index[0], "estrus_confirmed"] = True

    total_readings = int(df["total_readings"].sum()) if "total_readings" in df.columns else 0

    daily_data = []
    for _, row in df.fillna(0).iterrows():
        daily_data.append({
            "date": str(row.get("date")),
            "activity_score": float(row.get("activity_score", 0)),
            "activity_score_raw": float(row.get("activity_score_raw", 0)),
            "resting_fraction": float(row.get("resting_fraction", 0)),
            "eating_fraction": float(row.get("eating_fraction", 0)),
            "ruminating_fraction": float(row.get("ruminating_fraction", 0)),
            "walking_fraction": float(row.get("walking_fraction", 0)),
            "estrus_confirmed": bool(row.get("estrus_confirmed", False)),
            "amb_mean": float(row.get("amb_mean", 0)) if "amb_mean" in row else 0.0,
            "obj_mean": float(row.get("obj_mean", 0)) if "obj_mean" in row else 0.0,
        })

    return {
        "daily_data": daily_data,
        "estrus_detected": bool(df["estrus_confirmed"].any()),
        "estrus_date": str(estrus_day) if estrus_day else None,
        "total_days": int(len(df)),
        "total_readings": int(total_readings),
        "summary": {
            "avg_activity_score": float(df["activity_score"].mean()) if "activity_score" in df.columns else 0.0,
            "avg_activity_score_raw": float(df["activity_score_raw"].mean()) if "activity_score_raw" in df.columns else 0.0,
            "avg_walking_fraction": float(df["walking_fraction"].mean()),
            "avg_resting_fraction": float(df["resting_fraction"].mean()),
            "max_activity_score": float(df["activity_score"].max()) if "activity_score" in df.columns else 0.0,
            "min_activity_score": float(df["activity_score"].min()) if "activity_score" in df.columns else 0.0,
            "max_activity_score_raw": float(df["activity_score_raw"].max()) if "activity_score_raw" in df.columns else 0.0,
            "min_activity_score_raw": float(df["activity_score_raw"].min()) if "activity_score_raw" in df.columns else 0.0,
        },
        "last_updated": format_bst(now_bst()),
    }


def get_current_activity(df: pd.DataFrame) -> str:
    """Get the most recent activity state"""
    if df.empty:
        return "unknown"
    
    # Get last hour of data
    recent = df.tail(100)
    if recent.empty:
        return "unknown"
    
    # Classify based on recent movement
    df_temp = preprocess_sensor_data(recent.copy())
    df_temp["acc_mag"] = np.sqrt((df_temp[ACC_COLS].astype(float) ** 2).sum(axis=1))
    df_temp["gyro_mag"] = np.sqrt((df_temp[GYRO_COLS].astype(float) ** 2).sum(axis=1))
    df_temp["mag_mag"] = np.sqrt((df_temp[MAG_COLS].astype(float) ** 2).sum(axis=1))
    df_temp["movement_score"] = df_temp["acc_mag"] + df_temp["gyro_mag"] + df_temp["mag_mag"]
    
    avg_movement = df_temp["movement_score"].mean()
    q30 = df_temp["movement_score"].quantile(0.30)
    q70 = df_temp["movement_score"].quantile(0.70)
    mid = (q30 + q70) / 2
    
    if avg_movement < q30:
        return "resting"
    elif avg_movement < mid:
        return "eating"
    elif avg_movement < q70:
        return "ruminating"
    else:
        return "walking"
