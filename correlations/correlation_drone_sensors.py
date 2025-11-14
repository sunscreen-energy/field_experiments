import numpy as np
import pandas as pd
from pathlib import Path
from utils import (
    load_sensor_data,
    group_sensors_by_location,
    parse_tif_timestamp,
    get_pyranometer_data_at_time,
    mean_center,
    compute_correlations,
    plot_correlation,
    DISPERSION_START_EPOCH,
    DISPERSION_END_EPOCH
)
from drone_footage import DroneFootageAnalysis, CORNERS, BUFFER_METERS


def collect_drone_pyranometer_pairs(tif_files, sensor_coords, pyranometer_df, drone_analysis, radius_m=2, time_window_s=30, verbose=False):
    """
    Collect paired measurements of drone temperatures and pyranometer variables.

    For each drone pass:
      - Group sensors by location
      - For each location, check if there's drone coverage (temps within radius_m)
      - If no drone coverage at a location, skip all sensors at that location
      - If there is coverage, get pyranometer measurements for all sensors at that location
      - Compute means and pair them

    Args:
        tif_files: List of TIF file paths
        sensor_coords: Dict mapping sensor IDs to (lat, lon) tuples
        pyranometer_df: DataFrame with pyranometer data
        drone_analysis: DroneFootageAnalysis instance (with decorrelation already computed)
        radius_m: Half-width of square in meters (default: 2)
        time_window_s: Time window in seconds for matching pyranometer data (default: 30)
        verbose: Print detailed debugging info (default: False)

    Returns:
        DataFrame with columns: drone_temp, and pyranometer variable columns
    """
    all_pairs = []
    location_groups = group_sensors_by_location(sensor_coords)

    print(f"\nProcessing {len(tif_files)} drone passes for {len(location_groups)} sensor locations...")

    if verbose and hasattr(drone_analysis, 'decorrelation_params'):
        print("\nDecorelation parameters:")
        for direction in ['latitude', 'longitude']:
            if drone_analysis.decorrelation_params[direction] is not None:
                params = drone_analysis.decorrelation_params[direction]
                print(f"  {direction}: slope={params['slope']:.6f}, R²={params['r_squared']:.6f}")

    for tif_file in tif_files:
        epoch, dt = parse_tif_timestamp(tif_file)
        print(f"  {Path(tif_file).name} (epoch: {epoch}, time: {dt})")

        for units in location_groups.values():
            lat, lon = sensor_coords[units[0]]

            drone_temps_decorr = drone_analysis.extract_temps_near_sensor(
                tif_file, lat, lon, radius_m, apply_decorrelation=True
            )

            if verbose and len(drone_temps_decorr) > 0:
                drone_temps_raw = drone_analysis.extract_temps_near_sensor(
                    tif_file, lat, lon, radius_m, apply_decorrelation=False
                )
                print(f"    Sensor ({lat:.6f}, {lon:.6f}): "
                      f"raw={np.mean(drone_temps_raw):.3f}C, "
                      f"decorr={np.mean(drone_temps_decorr):.3f}C, "
                      f"diff={np.mean(drone_temps_raw) - np.mean(drone_temps_decorr):.3f}C")

            if len(drone_temps_decorr) == 0:
                continue

            mean_drone_temp = np.mean(drone_temps_decorr)

            pyran_data = get_pyranometer_data_at_time(pyranometer_df, epoch, time_window_s)

            location_data = pyran_data[pyran_data['Device_ID'].isin(units)]

            if len(location_data) == 0:
                continue

            pair = {'drone_temp': mean_drone_temp}

            for col in location_data.columns:
                if col in ['Device_ID', 'epoch_utc', 'local_iso8601', 't_plus_s', 'source_file', 'wind_dir_cardinal']:
                    continue

                if location_data[col].dtype == 'object':
                    continue

                values = location_data[col].dropna()
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
        dict mapping variable names to (x_values, y_values, r_squared, n_points, slope, intercept) tuples
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

    return compute_correlations(pairs_df, 'mean_centered_drone_temp', variables)


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

    print("\nInitializing drone footage analysis for spatial decorrelation...")
    drone_analysis = DroneFootageAnalysis(
        corners=CORNERS,
        buffer_m=BUFFER_METERS,
        tif_pattern='data/2025-11-07/drone_imaging/*.tif',
        dispersion_start_epoch=DISPERSION_START_EPOCH,
        dispersion_end_epoch=DISPERSION_END_EPOCH
    )

    print("\nComputing spatial decorrelation parameters...")
    drone_analysis.decorrelate_spatial(timeframes=['before'], direction='latitude')
    drone_analysis.decorrelate_spatial(timeframes=['before'], direction='longitude')

    print("\nCollecting drone-pyranometer paired measurements (with spatial decorrelation)...")
    pairs_df = collect_drone_pyranometer_pairs(tif_files, sensor_coords, pyranometer_df, drone_analysis, verbose=False)

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
        plot_correlation(var_name, x, y, r_squared, n_points, slope, intercept, output_dir, 'Mean-Centered Drone Temperature (C)')

    print(f"\nDone! Created {len(correlations)} plots in {output_dir}")


if __name__ == '__main__':
    main()
