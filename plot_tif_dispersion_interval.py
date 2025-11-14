import rasterio
from rasterio.warp import transform
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from utils import (
    load_sensor_data,
    group_sensors_by_location,
    parse_tif_timestamp,
    DISPERSION_START_EPOCH,
    DISPERSION_END_EPOCH
)
from drone_footage import EMISSION_SITE_X, EMISSION_SITE_Y

# Configuration flag
USE_OUTSIDE_BASELINE = False  # If False, use only BEFORE; if True, use BEFORE + AFTER


def classify_tif_files(tif_files):
    """
    Classify TIF files as BEFORE, DURING, or AFTER dispersion.

    Returns:
        dict with keys 'before', 'during', 'after' mapping to lists of file paths
    """
    classification = {'before': [], 'during': [], 'after': []}

    for tif_file in tif_files:
        epoch, _ = parse_tif_timestamp(tif_file)

        if epoch < DISPERSION_START_EPOCH:
            classification['before'].append(tif_file)
        elif epoch <= DISPERSION_END_EPOCH:
            classification['during'].append(tif_file)
        else:
            classification['after'].append(tif_file)

    return classification


def load_tif_data(tif_files, common_bounds=None, common_shape=None):
    """
    Load TIF data from multiple files with geographic alignment.

    Uses geographic bounds to ensure pixel (0,0) represents the same location in all images.

    Returns:
        tuple: (data_stack, bounds, crs, shape) where data_stack is shape (n_files, height, width)
    """
    data_list = []
    crs = None

    # First pass: determine common bounds (intersection of all bounds)
    if common_bounds is None or common_shape is None:
        all_bounds = []

        for tif_file in tif_files:
            with rasterio.open(tif_file) as src:
                all_bounds.append(src.bounds)
                if crs is None:
                    crs = src.crs

        # Find intersection of all bounds (common geographic area)
        common_bounds = rasterio.coords.BoundingBox(
            left=max(b.left for b in all_bounds),
            bottom=max(b.bottom for b in all_bounds),
            right=min(b.right for b in all_bounds),
            top=min(b.top for b in all_bounds)
        )

        # Use first file's transform to determine pixel size
        with rasterio.open(tif_files[0]) as src:
            transform = src.transform
            pixel_width = abs(transform.a)
            pixel_height = abs(transform.e)

        # Calculate shape from geographic bounds
        width = int((common_bounds.right - common_bounds.left) / pixel_width)
        height = int((common_bounds.top - common_bounds.bottom) / pixel_height)
        common_shape = (height, width)

        print(f"  Common geographic bounds:")
        print(f"    Left: {common_bounds.left:.2f}, Right: {common_bounds.right:.2f}")
        print(f"    Bottom: {common_bounds.bottom:.2f}, Top: {common_bounds.top:.2f}")
        print(f"  Common shape: {common_shape}")
        print(f"  Pixel resolution: {pixel_width:.2f}m x {pixel_height:.2f}m")

    # Second pass: load data using geographic windows
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            # Convert geographic bounds to pixel coordinates for this file
            window = src.window(
                common_bounds.left,
                common_bounds.bottom,
                common_bounds.right,
                common_bounds.top
            )

            # Round window to integer pixels
            window = window.round_lengths().round_offsets()

            # Read the windowed data
            data = src.read(1, window=window)

            # Ensure consistent shape (may need minor adjustment due to rounding)
            if data.shape != common_shape:
                # Crop or pad to match common_shape
                data_adjusted = np.full(common_shape, -9999, dtype=data.dtype)
                h_min = min(data.shape[0], common_shape[0])
                w_min = min(data.shape[1], common_shape[1])
                data_adjusted[:h_min, :w_min] = data[:h_min, :w_min]
                data = data_adjusted

            data_list.append(data)

    data_stack = np.stack(data_list, axis=0)
    return data_stack, common_bounds, crs, common_shape


def compute_valid_mask(data_stack, nodata_threshold=-1000):
    """
    Compute mask of pixels that are valid in ALL files.

    Returns:
        boolean mask of shape (height, width)
    """
    # Pixel is valid if it's valid in all time points
    valid_mask = np.all(data_stack > nodata_threshold, axis=0)
    return valid_mask


def get_pm_for_location_during_dispersion(location_units, pyranometer_df):
    """
    Get average PM2.5 concentration for a location during the dispersion interval.

    Returns:
        float or None: Average PM2.5 concentration during dispersion
    """
    pm_values = []

    for device_id in location_units:
        device_data = pyranometer_df[
            (pyranometer_df['Device_ID'] == device_id) &
            (pyranometer_df['epoch_utc'] >= DISPERSION_START_EPOCH) &
            (pyranometer_df['epoch_utc'] <= DISPERSION_END_EPOCH) &
            (pyranometer_df['sps_pm2_5'].notna())
        ]

        if not device_data.empty:
            pm_values.append(device_data['sps_pm2_5'].mean())

    return np.mean(pm_values) if pm_values else None


def plot_dispersion_effect(temporal_difference, bounds, crs, sensor_coords,
                           location_groups, pyranometer_df, output_path):
    """
    Plot the temporal difference with sensor locations labeled by PM concentration.
    """
    # Mask fill values
    data_masked = np.ma.masked_invalid(temporal_difference)

    # Calculate percentiles for colorbar
    valid_temps = temporal_difference[~np.isnan(temporal_difference)]
    p10 = np.percentile(valid_temps, 10)
    p90 = np.percentile(valid_temps, 90)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1],
                         hspace=0.3, wspace=0.3)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_hist = fig.add_subplot(gs[0, 1])

    # Plot thermal difference
    im = ax_map.imshow(
        data_masked, cmap='RdBu_r',
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        vmin=-3.5, vmax=3.5
    )

    # Plot sensor locations grouped by location
    for location_id, units in location_groups.items():
        # Get average lat/lon for this location
        lats = [sensor_coords[unit][0] for unit in units]
        lons = [sensor_coords[unit][1] for unit in units]
        avg_lat = np.mean(lats)
        avg_lon = np.mean(lons)

        # Transform to map coordinates
        x, y = transform('EPSG:4326', crs, [avg_lon], [avg_lat])

        # Get PM concentration during dispersion for this location
        pm_conc = get_pm_for_location_during_dispersion(units, pyranometer_df)

        # Plot marker
        ax_map.scatter(x[0], y[0], c='cyan', s=80, marker='o',
                      edgecolors='black', linewidths=1.5, alpha=0.8, zorder=5)

        # Add label
        if pm_conc is not None:
            label = f'{pm_conc:.1f}'
        else:
            label = f'L{location_id}'

        ax_map.annotate(label, (x[0], y[0]), xytext=(3, 3),
                       textcoords='offset points', fontsize=7,
                       color='white', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='black', alpha=0.6))

    # Plot emission site
    ax_map.scatter(EMISSION_SITE_X, EMISSION_SITE_Y, c='red', s=150, marker='*',
                  edgecolors='white', linewidths=2, alpha=1.0, zorder=10,
                  label='Emission Site')
    ax_map.legend(loc='upper right', fontsize=8)

    plt.colorbar(im, ax=ax_map, label='Normalized Temperature Difference (C)')
    ax_map.set_xlabel('Easting (m)')
    ax_map.set_ylabel('Northing (m)')

    baseline_type = 'BEFORE+AFTER' if USE_OUTSIDE_BASELINE else 'BEFORE'
    ax_map.set_title(f'Dispersion Effect: DURING - {baseline_type}\n(Mean-centered)')

    # Plot histogram
    ax_hist.hist(valid_temps, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax_hist.axvline(p10, color='blue', linestyle='--', linewidth=2,
                   label=f'10th %ile: {p10:.2f}C')
    ax_hist.axvline(p90, color='red', linestyle='--', linewidth=2,
                   label=f'90th %ile: {p90:.2f}C')
    ax_hist.axvline(0, color='black', linestyle='-', linewidth=1.5,
                   label='Zero (mean)')
    ax_hist.set_xlabel('Temperature Difference (C)')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_title('Difference Distribution')
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to {output_path}")
    print(f"  Value range: {valid_temps.min():.2f} to {valid_temps.max():.2f} C")
    print(f"  Mean: {valid_temps.mean():.4f} C (should be ~0)")
    print(f"  Std dev: {valid_temps.std():.2f} C")


def main():
    data_dir = Path('data/2025-11-07/drone_imaging')
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    # Load sensor data
    sensor_coords, pyranometer_df = load_sensor_data()

    # Group sensors by location
    location_groups = group_sensors_by_location(sensor_coords)
    print(f"Grouped {len(sensor_coords)} sensors into {len(location_groups)} locations")

    # Get sorted TIF files
    tif_files = sorted(data_dir.glob('*.tif'))

    if not tif_files:
        print("No .tif files found")
        return

    print(f"\nFound {len(tif_files)} .tif files")

    # Classify TIF files by time period
    classification = classify_tif_files(tif_files)
    print(f"\nTIF file classification:")
    print(f"  BEFORE dispersion: {len(classification['before'])} files")
    for f in classification['before']:
        print(f"    {Path(f).name}")
    print(f"  DURING dispersion: {len(classification['during'])} files")
    for f in classification['during']:
        print(f"    {Path(f).name}")
    print(f"  AFTER dispersion: {len(classification['after'])} files")
    for f in classification['after']:
        print(f"    {Path(f).name}")

    # Load all TIF data to determine common bounds and shape
    print("\nLoading TIF data...")
    all_data, common_bounds, crs, common_shape = load_tif_data(tif_files)
    print(f"  Data shape: {all_data.shape}")

    # Compute valid pixel mask (pixels valid in ALL files)
    print("\nComputing valid pixel mask...")
    mask_of_all_pix = compute_valid_mask(all_data)
    n_valid = mask_of_all_pix.sum()
    total_pix = mask_of_all_pix.size
    print(f"  Valid pixels: {n_valid}/{total_pix} ({100*n_valid/total_pix:.1f}%)")

    # Load data by period using same bounds and shape
    print("\nLoading data by period (using common bounds/shape)...")
    before_data, _, _, _ = load_tif_data(classification['before'],
                                          common_bounds=common_bounds,
                                          common_shape=common_shape)
    during_data, _, _, _ = load_tif_data(classification['during'],
                                          common_bounds=common_bounds,
                                          common_shape=common_shape)
    after_data, _, _, _ = load_tif_data(classification['after'],
                                         common_bounds=common_bounds,
                                         common_shape=common_shape)

    # Filter for only valid pixels: shape (n_times, n_valid_pixels)
    before_pix = before_data[:, mask_of_all_pix]
    during_pix = during_data[:, mask_of_all_pix]
    after_pix = after_data[:, mask_of_all_pix]

    print(f"\nFiltered data shapes:")
    print(f"  BEFORE: {before_pix.shape}")
    print(f"  DURING: {during_pix.shape}")
    print(f"  AFTER: {after_pix.shape}")

    # Compute temporal means
    during_mean = during_pix.mean(axis=0)  # Mean over time

    if USE_OUTSIDE_BASELINE:
        # Combine BEFORE and AFTER
        outside_pix = np.concatenate([before_pix, after_pix], axis=0)
        baseline_mean = outside_pix.mean(axis=0)
        print(f"\nUsing OUTSIDE baseline (BEFORE + AFTER): {outside_pix.shape}")
    else:
        baseline_mean = before_pix.mean(axis=0)
        print(f"\nUsing BEFORE baseline only")

    # Compute temporal difference
    temporal_difference_1d = during_mean - baseline_mean

    # Subtract mean to center at zero
    temporal_difference_1d -= temporal_difference_1d.mean()

    # Reconstruct 2D array
    temporal_difference = np.full(mask_of_all_pix.shape, np.nan)
    temporal_difference[mask_of_all_pix] = temporal_difference_1d

    print(f"\nTemporal difference statistics (before mean-centering):")
    print(f"  Original mean: {(during_mean - baseline_mean).mean():.4f} C")
    print(f"  After mean-centering: {temporal_difference_1d.mean():.4f} C")
    print(f"  Std dev: {temporal_difference_1d.std():.2f} C")

    # Plot result
    output_path = plots_dir / 'tif_dispersion_effect.png'
    print(f"\nPlotting dispersion effect...")
    plot_dispersion_effect(temporal_difference, common_bounds, crs, sensor_coords,
                          location_groups, pyranometer_df, output_path)


if __name__ == '__main__':
    main()
