import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime, timezone, timedelta
import yaml
import argparse
from scipy import stats


def parse_config_time(time_str, tz_offset_hours):
    """
    Convert time string in format YYYY-MM-DD-HH-MM-SS to epoch UTC.

    Args:
        time_str: Time string in format YYYY-MM-DD-HH-MM-SS
        tz_offset_hours: Timezone offset from UTC in hours (e.g., -8 for PST)

    Returns:
        int: Epoch UTC timestamp
    """
    parts = time_str.split('-')
    year, month, day, hour, minute, second = map(int, parts)
    tz = timezone(timedelta(hours=tz_offset_hours))
    dt = datetime(year, month, day, hour, minute, second, tzinfo=tz)
    return int(dt.timestamp())


def load_config(date):
    """
    Load configuration from YAML file for the specified date.

    Args:
        date: Date string in format YYYY-MM-DD

    Returns:
        dict: Configuration dictionary with epoch UTC timestamps
    """
    config_path = Path(f'data/{date}/thermal_cameras/config.yaml')

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert all time strings to epoch UTC
    tz_offset = config['timezone_offset']

    config['camera_active_start_epoch'] = parse_config_time(config['camera_active_start'], tz_offset)
    config['camera_active_end_epoch'] = parse_config_time(config['camera_active_end'], tz_offset)
    config['dispersion_start_epoch'] = parse_config_time(config['dispersion_start'], tz_offset)
    config['dispersion_end_epoch'] = parse_config_time(config['dispersion_end'], tz_offset)

    if config['drop_cloud']:
        config['cloud_start_epoch'] = parse_config_time(config['cloud_start'], tz_offset)
        config['cloud_end_epoch'] = parse_config_time(config['cloud_end'], tz_offset)

    return config


def load_thermal_camera_data(control_path, experimental_path, config):
    """
    Load and preprocess thermal camera data from CSV files.
    Filters to time range when both cameras are active.

    Returns:
        tuple: (control_df, experimental_df, pixel_cols)
    """
    print("Loading thermal camera data...")
    control_df = pd.read_csv(control_path)
    experimental_df = pd.read_csv(experimental_path)

    # Filter to active time range
    control_df = control_df[
        (control_df['epoch_utc'] >= config['camera_active_start_epoch']) &
        (control_df['epoch_utc'] <= config['camera_active_end_epoch'])
    ].reset_index(drop=True)

    experimental_df = experimental_df[
        (experimental_df['epoch_utc'] >= config['camera_active_start_epoch']) &
        (experimental_df['epoch_utc'] <= config['camera_active_end_epoch'])
    ].reset_index(drop=True)

    # Filter out cloud period if configured
    if config['drop_cloud']:
        control_df = control_df[
            ~((control_df['epoch_utc'] >= config['cloud_start_epoch']) &
              (control_df['epoch_utc'] <= config['cloud_end_epoch']))
        ].reset_index(drop=True)

        experimental_df = experimental_df[
            ~((experimental_df['epoch_utc'] >= config['cloud_start_epoch']) &
              (experimental_df['epoch_utc'] <= config['cloud_end_epoch']))
        ].reset_index(drop=True)

        print(f"\nAfter removing cloud period (epoch {config['cloud_start_epoch']} to {config['cloud_end_epoch']}):")
        print(f"  Control: {control_df.shape}")
        print(f"  Experimental: {experimental_df.shape}")

    print(f"\nTime range:")
    print(f"  Start: {control_df['local_iso8601'].min()}")
    print(f"  End: {control_df['local_iso8601'].max()}")

    pixel_cols = [col for col in control_df.columns if col.startswith('pix_')]
    print(f"\nNumber of pixel columns: {len(pixel_cols)}")

    # Convert pixel columns to numeric
    for col in pixel_cols:
        control_df[col] = pd.to_numeric(control_df[col], errors='coerce')
        experimental_df[col] = pd.to_numeric(experimental_df[col], errors='coerce')

    # Print overall statistics
    control_pixels = control_df[pixel_cols].values
    experimental_pixels = experimental_df[pixel_cols].values

    print(f"\nControl temperature statistics:")
    print(f"  Min: {control_pixels.min():.2f} C")
    print(f"  Max: {control_pixels.max():.2f} C")
    print(f"  Mean: {control_pixels.mean():.2f} C")

    print(f"\nExperimental temperature statistics:")
    print(f"  Min: {experimental_pixels.min():.2f} C")
    print(f"  Max: {experimental_pixels.max():.2f} C")
    print(f"  Mean: {experimental_pixels.mean():.2f} C")

    return control_df, experimental_df, pixel_cols


def plot_sample_frames_during_dispersion(control_df, experimental_df, pixel_cols, plots_dir, config):
    """
    Plot sample thermal camera frames from the dispersion window.

    Args:
        control_df: Control camera dataframe
        experimental_df: Experimental camera dataframe
        pixel_cols: List of pixel column names
        plots_dir: Directory to save plots
        config: Configuration dictionary
    """
    im_height = 24
    control_dispersion_df = control_df[
        (control_df['epoch_utc'] >= config['dispersion_start_epoch']) &
        (control_df['epoch_utc'] <= config['dispersion_end_epoch'])
    ].reset_index(drop=True)

    experimental_dispersion_df = experimental_df[
        (experimental_df['epoch_utc'] >= config['dispersion_start_epoch']) &
        (experimental_df['epoch_utc'] <= config['dispersion_end_epoch'])
    ].reset_index(drop=True)

    print(f"\nFrames within dispersion window:")
    print(f"  Control: {len(control_dispersion_df)} frames")
    print(f"  Experimental: {len(experimental_dispersion_df)} frames")

    if len(control_dispersion_df) == 0 or len(experimental_dispersion_df) == 0:
        print("WARNING: No frames found in dispersion window. Skipping sample frames plot.")
        return

    # Sample 6 frames evenly across dispersion window
    sample_indices_control = np.linspace(0, len(control_dispersion_df)-1, 6, dtype=int)
    sample_indices_exp = np.linspace(0, len(experimental_dispersion_df)-1, 6, dtype=int)

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    for idx, ax_idx in enumerate(sample_indices_control):
        frame_pixels = control_dispersion_df.iloc[ax_idx][pixel_cols].values.astype(float)

        frame_image = frame_pixels.reshape(im_height, 32)

        im = axes[0, idx].imshow(frame_image, cmap='hot', aspect='auto')
        axes[0, idx].set_title(f'Control\nFrame {ax_idx}/{len(control_dispersion_df)}', fontsize=8)
        axes[0, idx].axis('off')
        plt.colorbar(im, ax=axes[0, idx], fraction=0.046, pad=0.04)

    for idx, ax_idx in enumerate(sample_indices_exp):
        frame_pixels = experimental_dispersion_df.iloc[ax_idx][pixel_cols].values.astype(float)
        frame_image = frame_pixels.reshape(im_height, 32)

        im = axes[1, idx].imshow(frame_image, cmap='hot', aspect='auto')
        axes[1, idx].set_title(f'Experimental\nFrame {ax_idx}/{len(experimental_dispersion_df)}', fontsize=8)
        axes[1, idx].axis('off')
        plt.colorbar(im, ax=axes[1, idx], fraction=0.046, pad=0.04)

    dispersion_time_label = f"{config['dispersion_start']} - {config['dispersion_end']}"
    plt.suptitle(f'Thermal Camera Sample Frames During CaCO3 Dispersion ({dispersion_time_label})', fontsize=14)
    plt.tight_layout()
    output_path = plots_dir / 'thermal_camera_sample_frames.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sample frames to {output_path}")


def plot_temporal_analysis(control_df, experimental_df, pixel_cols, plots_dir, config):
    """
    Plot temporal analysis of thermal camera data using temperature anomalies.
    Anomalies are calculated as difference from each pixel's temporal mean.

    Args:
        control_df: Control camera dataframe
        experimental_df: Experimental camera dataframe
        pixel_cols: List of pixel column names
        plots_dir: Directory to save plots
        config: Configuration dictionary
    """
    print(f"\nCaCO3 Dispersion Window:")
    print(f"  Start: {config['dispersion_start']} -> Epoch: {config['dispersion_start_epoch']}")
    print(f"  End:   {config['dispersion_end']} -> Epoch: {config['dispersion_end_epoch']}")

    # Calculate pixel-wise temporal means
    if config['mean_based_on_after']:
        control_pixel_means = control_df[
            (control_df['epoch_utc'] > config['dispersion_end_epoch']) |
            (control_df['epoch_utc'] < config['dispersion_start_epoch'])
        ][pixel_cols].mean(axis=0)
        experimental_pixel_means = experimental_df[
            (experimental_df['epoch_utc'] > config['dispersion_end_epoch']) |
            (experimental_df['epoch_utc'] < config['dispersion_start_epoch'])
        ][pixel_cols].mean(axis=0)
    else:
        control_pixel_means = control_df[
            control_df['epoch_utc'] < config['dispersion_start_epoch']
        ][pixel_cols].mean(axis=0)
        experimental_pixel_means = experimental_df[
            experimental_df['epoch_utc'] < config['dispersion_start_epoch']
        ][pixel_cols].mean(axis=0)

    # Calculate temperature anomalies (difference from pixel mean)
    control_anomalies = control_df[pixel_cols].sub(control_pixel_means, axis=1)
    experimental_anomalies = experimental_df[pixel_cols].sub(experimental_pixel_means, axis=1)

    # Calculate mean anomaly across all pixels for each time point
    control_mean_anomaly = control_anomalies.mean(axis=1)
    experimental_mean_anomaly = experimental_anomalies.mean(axis=1)

    # Filter to dispersion window
    control_dispersion = control_df[
        (control_df['epoch_utc'] >= config['dispersion_start_epoch']) &
        (control_df['epoch_utc'] <= config['dispersion_end_epoch'])
    ]
    experimental_dispersion = experimental_df[
        (experimental_df['epoch_utc'] >= config['dispersion_start_epoch']) &
        (experimental_df['epoch_utc'] <= config['dispersion_end_epoch'])
    ]

    control_dispersion_anomalies = control_anomalies.loc[control_dispersion.index]
    experimental_dispersion_anomalies = experimental_anomalies.loc[experimental_dispersion.index]

    # Print statistics for dispersion window
    print(f"\nStatistics within CaCO3 Dispersion Window:")
    print(f"\nControl camera:")
    print(f"  Frames during dispersion: {len(control_dispersion)}")
    if len(control_dispersion) > 0:
        control_disp_anomaly_values = control_dispersion_anomalies.values.flatten()
        print(f"  Mean temperature anomaly: {control_disp_anomaly_values.mean():.3f} C")
        print(f"  Std dev: {control_disp_anomaly_values.std():.3f} C")

    print(f"\nExperimental camera:")
    print(f"  Frames during dispersion: {len(experimental_dispersion)}")
    if len(experimental_dispersion) > 0:
        experimental_disp_anomaly_values = experimental_dispersion_anomalies.values.flatten()
        print(f"  Mean temperature anomaly: {experimental_disp_anomaly_values.mean():.3f} C")
        print(f"  Std dev: {experimental_disp_anomaly_values.std():.3f} C")

    if len(control_dispersion) > 0 and len(experimental_dispersion) > 0:
        diff = experimental_disp_anomaly_values.mean() - control_disp_anomaly_values.mean()
        print(f"Temperature difference between control and experimental: {diff:.3f} C")

        # Statistical significance test using frame-level means to avoid temporal autocorrelation
        # Each frame is treated as one observation (n = number of frames, not pixels × frames)
        control_frame_means = control_mean_anomaly.loc[control_dispersion.index]
        experimental_frame_means = experimental_mean_anomaly.loc[experimental_dispersion.index]

        t_stat, p_value = stats.ttest_ind(experimental_frame_means, control_frame_means)
        print(f"\nStatistical significance test (two-sample t-test on frame-level means):")
        print(f"  Sample sizes: n_control={len(control_frame_means)}, n_experimental={len(experimental_frame_means)}")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.6f}")

    # Create temporal analysis plots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Mean temperature anomaly plot
    dispersion_label = f"CaCO3 Dispersion\n({config['dispersion_start']} - {config['dispersion_end']})"
    ax1.axvspan(config['dispersion_start_epoch'], config['dispersion_end_epoch'], alpha=0.2, color='green',
                label=dispersion_label)
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.plot(control_df['epoch_utc'], control_mean_anomaly,
             label='Control', color='blue', alpha=0.7, linewidth=1)
    ax1.plot(experimental_df['epoch_utc'], experimental_mean_anomaly,
             label='Experimental', color='red', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Time (epoch UTC)')
    ax1.set_ylabel('Mean Temperature Anomaly (C)')
    ax1.set_title('Temperature Anomaly Over Time: Control vs Experimental\n(Anomaly = Temperature - Pixel Mean)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax1.grid(which='minor', alpha=0.15)

    # Histogram of temperature anomalies during dispersion window
    if len(control_dispersion) > 0 and len(experimental_dispersion) > 0:
        ax2.hist(control_disp_anomaly_values, bins=50, alpha=0.6, color='blue',
                label=f'Control (mean={control_disp_anomaly_values.mean():.3f} C)', edgecolor='black')
        ax2.hist(experimental_disp_anomaly_values, bins=50, alpha=0.6, color='red',
                label=f'Experimental (mean={experimental_disp_anomaly_values.mean():.3f} C)', edgecolor='black')
        ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Temperature Anomaly (C)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Temperature Anomalies During Dispersion Window')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.savefig(plots_dir / 'thermal_camera_temporal_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved temporal analysis to {plots_dir / 'thermal_camera_temporal_analysis.png'}")


def explore_thermal_camera_data(date):
    """
    Main function to explore and visualize thermal camera data for a specific date.

    Args:
        date: Date string in format YYYY-MM-DD
    """
    # Load configuration
    config = load_config(date)

    # Setup paths
    control_path = Path(f'data/{date}/thermal_cameras/control.csv')
    experimental_path = Path(f'data/{date}/thermal_cameras/experimental.csv')
    plots_dir = Path(f'plots/thermal_camera/{date}')
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    control_df, experimental_df, pixel_cols = load_thermal_camera_data(
        control_path, experimental_path, config
    )

    # Plot sample frames during dispersion
    plot_sample_frames_during_dispersion(
        control_df, experimental_df, pixel_cols, plots_dir, config
    )

    # Plot temporal analysis
    plot_temporal_analysis(
        control_df, experimental_df, pixel_cols, plots_dir, config
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explore thermal camera data for a specific date')
    parser.add_argument('date', type=str, help='Date in format YYYY-MM-DD')
    args = parser.parse_args()

    explore_thermal_camera_data(args.date)
