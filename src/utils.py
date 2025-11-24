import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as rasterio_transform
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone, timedelta


DISPERSION_START_EPOCH = 1762544520  # 2025-11-07 11:42:00 PST
DISPERSION_END_EPOCH = 1762548780    # 2025-11-07 12:50:00 PST


def load_sensor_data():
    """Load sensor coordinates and pyranometer data."""
    with open('sensor_coordinates/sensor_coordinates.json', 'r') as f:
        sensor_coords = json.load(f)

    pyranometer_df = pd.read_csv('data/2025-11-07/pyranometers/pyranometer_sensors.csv',
                                  low_memory=False)

    return sensor_coords, pyranometer_df


def group_sensors_by_location(sensor_coords, precision_decimals=7):
    """
    Groups sensors by exact (or near-exact) location.

    This method assumes sensors at the same location have coordinates
    that are identical within a very small floating-point tolerance.
    It works by rounding coordinates to a specified number of
    decimal places and grouping all sensors that map to the
    same rounded coordinate.

    Args:
        sensor_coords (dict):
            Maps unit_name (str) to (lat, lon) tuple.
        precision_decimals (int):
            The number of decimal places to round to for grouping.
            The default of 7 corresponds to ~1e-7 degrees (approx 1.11 cm).

    Returns:
        dict: Maps a unique, 0-indexed location_id (int) to a
              list of unit names (str) at that location.
    """

    location_map = defaultdict(list)

    for unit, (lat, lon) in sensor_coords.items():
        rounded_key = (round(lat, precision_decimals),
                       round(lon, precision_decimals))

        location_map[rounded_key].append(unit)


    final_groups = {}
    for i, units_list in enumerate(location_map.values()):
        final_groups[i] = units_list

    return final_groups


def parse_tif_timestamp(tif_path):
    """
    Parse timestamp from TIF filename.

    Format: 2025-11-07T1101PST_ORTHO_TIR_Celsius.tif

    Returns:
        epoch timestamp (int) and datetime object in PST
    """
    filename = Path(tif_path).stem
    parts = filename.split('T')
    date_str = parts[0]  # 2025-11-07
    time_str = parts[1].split('PST')[0]  # 1101

    year, month, day = date_str.split('-')
    hour = int(time_str[:2])
    minute = int(time_str[2:4])

    pst = timezone(timedelta(hours=-8))
    dt = datetime(int(year), int(month), int(day), hour, minute, tzinfo=pst)
    epoch = int(dt.timestamp())

    return epoch, dt


def get_pyranometer_data_at_time(pyranometer_df, epoch_time, time_window_s=60):
    """
    Get pyranometer measurements within a time window around a specific epoch time.

    Args:
        pyranometer_df: DataFrame with pyranometer data
        epoch_time: Target epoch time
        time_window_s: Time window in seconds (default: 60, i.e., +/- 1 min)

    Returns:
        DataFrame of pyranometer measurements within the time window
    """
    time_mask = (
        (pyranometer_df['epoch_utc'] >= epoch_time - time_window_s) &
        (pyranometer_df['epoch_utc'] <= epoch_time + time_window_s)
    )

    return pyranometer_df[time_mask]


def compute_regression_pvalue(r_squared, n):
    """
    Compute p-value for linear regression given R² and sample size.

    Uses F-statistic: F = (R² / k) / ((1 - R²) / (n - k - 1))
    where k=1 for simple linear regression.

    Args:
        r_squared: R-squared value from linear regression
        n: Number of data points

    Returns:
        p-value (float)
    """
    from scipy.stats import f

    if n <= 2:
        return 1.0

    if r_squared >= 1.0:
        return 0.0

    if r_squared <= 0.0:
        return 1.0

    k = 1
    f_stat = (r_squared / k) / ((1 - r_squared) / (n - k - 1))
    p_value = 1 - f.cdf(f_stat, k, n - k - 1)

    return p_value


def compute_correlations(data_df, x_column, variables, min_points=10):
    """
    Compute correlations between a predictor variable and multiple outcome variables.

    Args:
        data_df: DataFrame containing the data
        x_column: Name of the predictor column
        variables: List of outcome variable names to correlate with x_column
        min_points: Minimum number of valid data points required (default: 10)

    Returns:
        dict mapping variable names to (x_values, y_values, r_squared, n_points, slope, intercept) tuples
    """
    from scipy.stats import linregress

    correlations = {}

    for var in variables:
        if var not in data_df.columns:
            continue

        valid_mask = data_df[var].notna() & data_df[x_column].notna()
        valid_data = data_df[valid_mask]

        if len(valid_data) < min_points:
            print(f"  {var}: Insufficient data (n={len(valid_data)})")
            continue

        x = valid_data[x_column].values
        y = valid_data[var].values

        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            print(f"  {var}: No variance in data")
            continue

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value ** 2

        correlations[var] = (x, y, r_squared, len(valid_data), slope, intercept)

        print(f"  {var}: R² = {r_squared:.4f}, n = {len(valid_data)}, p = {p_value:.2e}")

    return correlations


def plot_correlation(var_name, x, y, r_squared, n_points, slope, intercept, output_dir, x_label):
    """
    Create a scatterplot showing correlation between two variables.

    Args:
        var_name: Name of the y-variable
        x: x-values
        y: y-values
        r_squared: R-squared value
        n_points: Number of data points
        slope: Slope of regression line
        intercept: Intercept of regression line
        output_dir: Directory to save the plot
        x_label: Label for x-axis
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(x, y, alpha=0.3, s=10, edgecolors='none')

    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {slope:.3f}x + {intercept:.3f}')

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(var_name, fontsize=12)
    ax.set_title(f'Correlation: {var_name} vs {x_label}\nR² = {r_squared:.4f}, n = {n_points}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir) / f'{var_name}_r_squared_{r_squared:.4f}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot to {output_path}")