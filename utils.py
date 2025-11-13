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


def latlon_to_utm(lat, lon, crs='EPSG:32610'):
    """
    Transform lat/lon coordinates to UTM.

    Args:
        lat: Latitude
        lon: Longitude
        crs: Target coordinate reference system (default: EPSG:32610 - UTM Zone 10N)

    Returns:
        tuple: (x, y) in UTM coordinates
    """
    x, y = rasterio_transform('EPSG:4326', crs, [lon], [lat])
    return x[0], y[0]


def extract_temps_near_sensor(tif_path, sensor_lat, sensor_lon, radius_m=10):
    """
    Extract temperature values within a radius of a sensor location from a TIF file.

    Args:
        tif_path: Path to the TIF file
        sensor_lat: Sensor latitude
        sensor_lon: Sensor longitude
        radius_m: Radius in meters (default: 10)

    Returns:
        numpy array of temperature values within radius, or empty array if none found
    """
    with rasterio.open(tif_path) as src:
        sensor_x, sensor_y = latlon_to_utm(sensor_lat, sensor_lon, src.crs)

        transform = src.transform

        col, row = ~transform * (sensor_x, sensor_y)
        col, row = int(col), int(row)

        if col < 0 or col >= src.width or row < 0 or row >= src.height:
            return np.array([])

        pixel_width = abs(transform.a)
        pixel_radius = int(np.ceil(radius_m / pixel_width))

        row_start = max(0, row - pixel_radius)
        row_end = min(src.height, row + pixel_radius + 1)
        col_start = max(0, col - pixel_radius)
        col_end = min(src.width, col + pixel_radius + 1)

        if row_end <= row_start or col_end <= col_start:
            return np.array([])

        window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
        data = src.read(1, window=window)

        temps = []
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                actual_row = row_start + r
                actual_col = col_start + c
                pixel_x, pixel_y = transform * (actual_col, actual_row)

                distance = np.sqrt((pixel_x - sensor_x)**2 + (pixel_y - sensor_y)**2)

                if distance <= radius_m:
                    temp = data[r, c]
                    if temp > -1000:
                        temps.append(temp)

        return np.array(temps)


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


def mean_center(values):
    """
    Mean-center a set of values.

    Args:
        values: Array-like of numeric values

    Returns:
        numpy array of mean-centered values
    """
    values = np.array(values)
    return values - np.nanmean(values)