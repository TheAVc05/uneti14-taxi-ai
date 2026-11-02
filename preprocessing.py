import numpy as np
import pandas as pd

TARGET = "trip_duration"
FEATURE_COLUMNS = (
    "vendor_id",
    "pickup_month",
    "pickup_dayofweek",
    "pickup_hour",
    "pickup_minute_bucket",
    "passenger_count",
    "pickup_lon_approx",
    "pickup_lat_approx",
    "dropoff_lon_approx",
    "dropoff_lat_approx",
    "straight_line_km_approx",
    "store_and_fwd_flag",
)

CATEGORICAL_COLUMNS = (
    "vendor_id",
    "store_and_fwd_flag",
    "pickup_month",
    "pickup_dayofweek",
    "pickup_hour",
    "pickup_minute_bucket",
    "passenger_count",
)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create model features using only information available at pickup."""
    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {', '.join(missing_columns)}")

    X = frame.loc[:, FEATURE_COLUMNS].copy()
    if X.isna().any().any():
        missing_values = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Input features contain missing values: {', '.join(missing_values)}")

    X["delta_lon"] = X["dropoff_lon_approx"] - X["pickup_lon_approx"]
    X["delta_lat"] = X["dropoff_lat_approx"] - X["pickup_lat_approx"]
    mean_latitude = np.deg2rad(
        (X["pickup_lat_approx"] + X["dropoff_lat_approx"]) / 2
    )
    east_west_km = X["delta_lon"] * 111.32 * np.cos(mean_latitude)
    north_south_km = X["delta_lat"] * 111.32
    X["manhattan_km_from_coords"] = east_west_km.abs() + north_south_km.abs()

    pickup_latitude = np.deg2rad(X["pickup_lat_approx"])
    dropoff_latitude = np.deg2rad(X["dropoff_lat_approx"])
    delta_latitude = dropoff_latitude - pickup_latitude
    delta_longitude = np.deg2rad(X["delta_lon"])
    haversine = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(pickup_latitude)
        * np.cos(dropoff_latitude)
        * np.sin(delta_longitude / 2) ** 2
    )
    X["haversine_km_from_coords"] = (
        2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
    )

    h = X["pickup_hour"].astype(float)
    pickup_time = h + X["pickup_minute_bucket"].astype(float) / 60
    X["pickup_time_sin"] = np.sin(2 * np.pi * pickup_time / 24)
    X["pickup_time_cos"] = np.cos(2 * np.pi * pickup_time / 24)
    d = X["pickup_dayofweek"].astype(float)
    X["pickup_day_sin"] = np.sin(2 * np.pi * d / 7)
    X["pickup_day_cos"] = np.cos(2 * np.pi * d / 7)
    month = X["pickup_month"].astype(float) - 1
    X["pickup_month_sin"] = np.sin(2 * np.pi * month / 12)
    X["pickup_month_cos"] = np.cos(2 * np.pi * month / 12)
    for column in CATEGORICAL_COLUMNS:
        X[column] = X[column].astype("category")
    return X
