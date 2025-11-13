import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress
from utils import (
    load_sensor_data,
    parse_tif_timestamp,
    extract_temps_near_sensor,
    get_pyranometer_data_at_time,
    mean_center
)


def collect_drone_pyranometer_pairs(tif_files, sensor_coords, pyranometer_df, radius_m=10, time_window_s=60):
    """
    Collect paired measurements of drone temperatures and pyranometer variables.

    For each drone pass:
      - Extract drone temps within radius_m of each sensor
      - Get pyranometer measurements at +/- time_window_s around drone pass time
      - Compute means and pair them

    Returns:
        DataFrame with columns: drone_temp, and pyranometer variable columns
    """
    all_pairs = []

    print(f"\nProcessing {len(tif_files)} drone passes...")

    for tif_file in tif_files:
        epoch, dt = parse_tif_timestamp(tif_file)
        print(f"  {Path(tif_file).name} (epoch: {epoch}, time: {dt})")

        for unit, (lat, lon) in sensor_coords.items():
            drone_temps = extract_temps_near_sensor(tif_file, lat, lon, radius_m)

            if len(drone_temps) == 0:
                continue

            mean_drone_temp = np.mean(drone_temps)

            pyran_data = get_pyranometer_data_at_time(pyranometer_df, epoch, time_window_s)

            unit_data = pyran_data[pyran_data['Device_ID'] == unit]

            if len(unit_data) == 0:
                continue

            pair = {'drone_temp': mean_drone_temp}

            for col in unit_data.columns:
                if col in ['Device_ID', 'epoch_utc', 'local_iso8601', 't_plus_s', 'source_file', 'wind_dir_cardinal']:
                    continue

                if unit_data[col].dtype == 'object':
                    continue

                values = unit_data[col].dropna()
                if len(values) > 0:
                    pair[col] = values.mean()

            if len(pair) > 1:
                all_pairs.append(pair)

    pairs_df = pd.DataFrame(all_pairs)

    print(f"\nCollected {len(pairs_df)} paired measurements")
    if len(pairs_df) > 0:
        print(f"  Drone temp range: {pairs_df['drone_temp'].min():.2f} to {pairs_df['drone_temp'].max():.2f} C")

    return pairs_df


def compute_correlations_with_drone_temp(pairs_df):
    """
    Compute correlations between pyranometer variables and drone temperatures.

    Returns:
        dict mapping variable names to (x_values, y_values, r_squared, n_points) tuples
    """
    if 'drone_temp' not in pairs_df.columns or len(pairs_df) < 10:
        print("Insufficient paired data for correlation analysis")
        return {}

    pairs_df['mean_centered_drone_temp'] = mean_center(pairs_df['drone_temp'])

    print(f"  Original drone temp mean: {pairs_df['drone_temp'].mean():.4f} C")
    print(f"  Mean-centered mean: {pairs_df['mean_centered_drone_temp'].mean():.4e} C")

    variables = [
        'temp_C', 'rh_pct', 'pyr_V', 'ghi_calibrated',
        'sps_pm1_0', 'sps_pm2_5', 'sps_pm4_0', 'sps_pm10',
        'sps_nc0_5', 'sps_nc1_0', 'sps_nc2_5', 'sps_nc4_0', 'sps_nc10',
        'sps_typ_particle_um', 'wind_dir_deg', 'wind_speed_kph'
    ]

    correlations = {}

    for var in variables:
        if var not in pairs_df.columns:
            continue

        valid_mask = pairs_df[var].notna() & pairs_df['mean_centered_drone_temp'].notna()
        valid_data = pairs_df[valid_mask]

        if len(valid_data) < 10:
            print(f"  {var}: Insufficient data (n={len(valid_data)})")
            continue

        x = valid_data['mean_centered_drone_temp'].values
        y = valid_data[var].values

        if np.std(y) < 1e-10:
            print(f"  {var}: No variance in data")
            continue

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = r_value ** 2

        correlations[var] = (x, y, r_squared, len(valid_data), slope, intercept)

        print(f"  {var}: R² = {r_squared:.4f}, n = {len(valid_data)}, p = {p_value:.2e}")

    return correlations


def plot_correlation(var_name, x, y, r_squared, n_points, slope, intercept, output_dir):
    """
    Create a scatterplot showing correlation between a variable and mean-centered drone temperature.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(x, y, alpha=0.3, s=10, edgecolors='none')

    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {slope:.3f}x + {intercept:.3f}')

    ax.set_xlabel('Mean-Centered Drone Temperature (C)', fontsize=12)
    ax.set_ylabel(var_name, fontsize=12)
    ax.set_title(f'Correlation: {var_name} vs Drone Temperature\nR² = {r_squared:.4f}, n = {n_points}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    output_path = output_dir / f'{var_name}_r_squared_{r_squared:.4f}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot to {output_path}")


def main():
    print("Loading sensor data...")
    sensor_coords, pyranometer_df = load_sensor_data()

    print(f"\nLoaded pyranometer data: {len(pyranometer_df)} rows, {pyranometer_df['Device_ID'].nunique()} devices")

    data_dir = Path('data/2025-11-07/drone_imaging')
    tif_files = sorted(data_dir.glob('*.tif'))

    if not tif_files:
        print("No TIF files found")
        return

    print(f"\nFound {len(tif_files)} drone TIF files")

    print("\nCollecting drone-pyranometer paired measurements...")
    pairs_df = collect_drone_pyranometer_pairs(tif_files, sensor_coords, pyranometer_df)

    if len(pairs_df) == 0:
        print("No paired data collected. Exiting.")
        return

    print("\nComputing correlations...")
    correlations = compute_correlations_with_drone_temp(pairs_df)

    if len(correlations) == 0:
        print("No correlations computed. Exiting.")
        return

    output_dir = Path('plots/drone_correlations')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating plots...")
    for var_name, (x, y, r_squared, n_points, slope, intercept) in correlations.items():
        print(f"  Plotting {var_name}...")
        plot_correlation(var_name, x, y, r_squared, n_points, slope, intercept, output_dir)

    print(f"\nDone! Created {len(correlations)} plots in {output_dir}")


if __name__ == '__main__':
    main()
