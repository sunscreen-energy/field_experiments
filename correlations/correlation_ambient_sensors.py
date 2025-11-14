import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress
from utils import load_sensor_data, mean_center


def compute_ambient_temperature(pyranometer_df):
    """
    Compute ambient air temperature as the mean temperature across all sensors at each time point.

    Returns:
        DataFrame with epoch_utc and mean_temp_C columns
    """
    ambient_temp = pyranometer_df.groupby('epoch_utc')['temp_C'].agg(['mean', 'std', 'count']).reset_index()
    ambient_temp.columns = ['epoch_utc', 'mean_temp_C', 'std_temp_C', 'n_sensors']

    ambient_temp = ambient_temp[ambient_temp['n_sensors'] >= 5]

    print(f"Computed ambient temperature for {len(ambient_temp)} time points")
    print(f"  Temperature range: {ambient_temp['mean_temp_C'].min():.2f} to {ambient_temp['mean_temp_C'].max():.2f} C")

    return ambient_temp


def compute_correlations_with_temp(pyranometer_df, ambient_temp):
    """
    Compute correlations between pyranometer variables and ambient temperature.

    Returns:
        dict mapping variable names to (x_values, y_values, r_squared, n_points) tuples
    """
    variables = [
        'rh_pct', 'pyr_V', 'ghi_calibrated',
        'sps_pm1_0', 'sps_pm2_5', 'sps_pm4_0', 'sps_pm10',
        'sps_nc0_5', 'sps_nc1_0', 'sps_nc2_5', 'sps_nc4_0', 'sps_nc10',
        'sps_typ_particle_um', 'wind_dir_deg', 'wind_speed_kph'
    ]

    merged_df = pyranometer_df.merge(ambient_temp[['epoch_utc', 'mean_temp_C']], on='epoch_utc', how='inner')

    merged_df['mean_centered_temp'] = mean_center(merged_df['mean_temp_C'])

    correlations = {}

    for var in variables:
        if var not in merged_df.columns:
            continue

        valid_mask = merged_df[var].notna() & merged_df['mean_centered_temp'].notna()
        valid_data = merged_df[valid_mask]

        if len(valid_data) < 10:
            print(f"  {var}: Insufficient data (n={len(valid_data)})")
            continue

        x = valid_data['mean_centered_temp'].values
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
    Create a scatterplot showing correlation between a variable and mean-centered temperature.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(x, y, alpha=0.3, s=10, edgecolors='none')

    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {slope:.3f}x + {intercept:.3f}')

    ax.set_xlabel('Mean-Centered Ambient Temperature (C)', fontsize=12)
    ax.set_ylabel(var_name, fontsize=12)
    ax.set_title(f'Correlation: {var_name} vs Ambient Temperature\nR² = {r_squared:.4f}, n = {n_points}', fontsize=14)
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

    print("\nComputing ambient temperature...")
    ambient_temp = compute_ambient_temperature(pyranometer_df)

    print("\nMean-centering temperatures...")
    ambient_temp['mean_centered_temp'] = mean_center(ambient_temp['mean_temp_C'])
    print(f"  Original mean: {ambient_temp['mean_temp_C'].mean():.4f} C")
    print(f"  Mean-centered mean: {ambient_temp['mean_centered_temp'].mean():.4e} C")

    print("\nComputing correlations...")
    correlations = compute_correlations_with_temp(pyranometer_df, ambient_temp)

    output_dir = Path('plots/ambient_air_correlations')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating plots...")
    for var_name, (x, y, r_squared, n_points, slope, intercept) in correlations.items():
        print(f"  Plotting {var_name}...")
        plot_correlation(var_name, x, y, r_squared, n_points, slope, intercept, output_dir)

    print(f"\nDone! Created {len(correlations)} plots in {output_dir}")


if __name__ == '__main__':
    main()
