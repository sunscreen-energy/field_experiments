from pathlib import Path
from ..utils import load_sensor_data, compute_correlations, plot_correlation


def main():
    print("Loading sensor data...")
    sensor_coords, pyranometer_df = load_sensor_data()

    print(f"\nLoaded pyranometer data: {len(pyranometer_df)} rows, {pyranometer_df['Device_ID'].nunique()} devices")

    valid_data = pyranometer_df[pyranometer_df['pyr_V'].notna()].copy()
    print(f"Total rows with valid pyr_V: {len(valid_data)}")

    variables = [
        'temp_C', 'rh_pct', 'ghi_calibrated',
        'sps_pm1_0', 'sps_pm2_5', 'sps_pm4_0', 'sps_pm10',
        'sps_nc0_5', 'sps_nc1_0', 'sps_nc2_5', 'sps_nc4_0', 'sps_nc10',
        'sps_typ_particle_um', 'wind_dir_deg', 'wind_speed_kph'
    ]

    print("\nComputing correlations with pyranometer voltage...")
    correlations = compute_correlations(valid_data, 'pyr_V', variables)

    if len(correlations) == 0:
        print("No correlations computed. Exiting.")
        return

    output_dir = Path('../plots/pyr_voltage_correlations')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating plots...")
    for var_name, (x, y, r_squared, n_points, slope, intercept) in correlations.items():
        print(f"  Plotting {var_name}...")
        plot_correlation(var_name, x, y, r_squared, n_points, slope, intercept, output_dir, 'Pyranometer Voltage (V)')

    print(f"\nDone! Created {len(correlations)} plots in {output_dir}")


if __name__ == '__main__':
    main()
