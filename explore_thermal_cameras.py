import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime, timezone, timedelta


DROP_CLOUD = True
KEEP_ONLY_MIDDLE_BAND = False

# Time Range Constants
CAMERA_ACTIVE_START = 1762541000  # Epoch UTC when both cameras are active
CAMERA_ACTIVE_END = 1762552000    # Epoch UTC when both cameras are active

CAMERA_CLOUD_START = 1762547500
CAMERA_CLOUD_END = 1762547900 if DROP_CLOUD else CAMERA_CLOUD_START 

# CaCO3 Dispersion Window Constants
PST = timezone(timedelta(hours=-8))
DISPERSION_START_DT = datetime(2025, 11, 7, 11, 42, 0, tzinfo=PST)
DISPERSION_END_DT = datetime(2025, 11, 7, 12, 50, 0, tzinfo=PST)
DISPERSION_START_EPOCH = int(DISPERSION_START_DT.timestamp())
DISPERSION_END_EPOCH = int(DISPERSION_END_DT.timestamp())

def load_thermal_camera_data(control_path, experimental_path):
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
        (control_df['epoch_utc'] >= CAMERA_ACTIVE_START) &
        (control_df['epoch_utc'] <= CAMERA_ACTIVE_END)
    ].reset_index(drop=True)

    experimental_df = experimental_df[
        (experimental_df['epoch_utc'] >= CAMERA_ACTIVE_START) &
        (experimental_df['epoch_utc'] <= CAMERA_ACTIVE_END)
    ].reset_index(drop=True)

    # Filter out cloud period
    control_df = control_df[
        ~((control_df['epoch_utc'] >= CAMERA_CLOUD_START) &
          (control_df['epoch_utc'] <= CAMERA_CLOUD_END))
    ].reset_index(drop=True)

    experimental_df = experimental_df[
        ~((experimental_df['epoch_utc'] >= CAMERA_CLOUD_START) &
          (experimental_df['epoch_utc'] <= CAMERA_CLOUD_END))
    ].reset_index(drop=True)

    print(f"\nAfter removing cloud period (epoch {CAMERA_CLOUD_START} to {CAMERA_CLOUD_END}):")
    print(f"  Control: {control_df.shape}")
    print(f"  Experimental: {experimental_df.shape}")

    print(f"\nTime range:")
    print(f"  Start: {control_df['local_iso8601'].min()}")
    print(f"  End: {control_df['local_iso8601'].max()}")

    pixel_cols = [col for col in control_df.columns if col.startswith('pix_')]
    if KEEP_ONLY_MIDDLE_BAND:
        pixel_cols = [col for col in pixel_cols if int(col.strip('pix_')) in set(range(256, 512))]
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

def plot_sample_frames_during_dispersion(control_df, experimental_df, pixel_cols, plots_dir):
    """
    Plot sample thermal camera frames from the dispersion window.

    Args:
        control_df: Control camera dataframe
        experimental_df: Experimental camera dataframe
        pixel_cols: List of pixel column names
        plots_dir: Directory to save plots
    """
    im_height = 8 if KEEP_ONLY_MIDDLE_BAND else 24
    control_dispersion_df = control_df[
        (control_df['epoch_utc'] >= DISPERSION_START_EPOCH) &
        (control_df['epoch_utc'] <= DISPERSION_END_EPOCH)
    ].reset_index(drop=True)

    experimental_dispersion_df = experimental_df[
        (experimental_df['epoch_utc'] >= DISPERSION_START_EPOCH) &
        (experimental_df['epoch_utc'] <= DISPERSION_END_EPOCH)
    ].reset_index(drop=True)

    print(f"\nFrames within dispersion window:")
    print(f"  Control: {len(control_dispersion_df)} frames")
    print(f"  Experimental: {len(experimental_dispersion_df)} frames")

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

    plt.suptitle('Thermal Camera Sample Frames During CaCO3 Dispersion (11:42 AM - 12:50 PM PST)', fontsize=14)
    plt.tight_layout()
    output_path = plots_dir / 'thermal_camera_sample_frames.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sample frames to {output_path}")

def plot_temporal_analysis(control_df, experimental_df, pixel_cols, plots_dir):
    """
    Plot temporal analysis of thermal camera data using temperature anomalies.
    Anomalies are calculated as difference from each pixel's temporal mean.

    Args:
        control_df: Control camera dataframe
        experimental_df: Experimental camera dataframe
        pixel_cols: List of pixel column names
        plots_dir: Directory to save plots
    """
    print(f"\nCaCO3 Dispersion Window:")
    print(f"  Start: {DISPERSION_START_DT} -> Epoch: {DISPERSION_START_EPOCH}")
    print(f"  End:   {DISPERSION_END_DT} -> Epoch: {DISPERSION_END_EPOCH}")

    # Calculate pixel-wise temporal means, before spraying
    control_pixel_means = control_df[control_df['epoch_utc'] < DISPERSION_START_EPOCH][pixel_cols].mean(axis=0)
    experimental_pixel_means = experimental_df[experimental_df['epoch_utc'] < DISPERSION_START_EPOCH][pixel_cols].mean(axis=0)

    # Calculate temperature anomalies (difference from pixel mean)
    control_anomalies = control_df[pixel_cols].sub(control_pixel_means, axis=1)
    experimental_anomalies = experimental_df[pixel_cols].sub(experimental_pixel_means, axis=1)

    # Calculate mean anomaly across all pixels for each time point
    control_mean_anomaly = control_anomalies.mean(axis=1)
    experimental_mean_anomaly = experimental_anomalies.mean(axis=1)

    # Filter to dispersion window
    control_dispersion = control_df[
        (control_df['epoch_utc'] >= DISPERSION_START_EPOCH) &
        (control_df['epoch_utc'] <= DISPERSION_END_EPOCH)
    ]
    experimental_dispersion = experimental_df[
        (experimental_df['epoch_utc'] >= DISPERSION_START_EPOCH) &
        (experimental_df['epoch_utc'] <= DISPERSION_END_EPOCH)
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
        print(f"Temperature difference between control and experimental: {experimental_disp_anomaly_values.mean() - control_disp_anomaly_values.mean():.3f} C")
    
    # Create temporal analysis plots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Mean temperature anomaly plot
    ax1.axvspan(DISPERSION_START_EPOCH, DISPERSION_END_EPOCH, alpha=0.2, color='green',
                label='CaCO3 Dispersion\n(11:42 AM - 12:50 PM PST)')
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
                label=f'Control (mean={control_disp_anomaly_values.mean():.3f}°C)', edgecolor='black')
        ax2.hist(experimental_disp_anomaly_values, bins=50, alpha=0.6, color='red',
                label=f'Experimental (mean={experimental_disp_anomaly_values.mean():.3f}°C)', edgecolor='black')
        ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Temperature Anomaly (C)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Temperature Anomalies During Dispersion Window')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.savefig(plots_dir / 'thermal_camera_temporal_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved temporal analysis to {plots_dir / 'thermal_camera_temporal_analysis.png'}")

def explore_thermal_camera_data():
    """
    Main function to explore and visualize thermal camera data.
    """
    control_path = Path('data/2025-11-07/thermal_cameras/control.csv')
    experimental_path = Path('data/2025-11-07/thermal_cameras/experimental.csv')
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)

    # Load data
    control_df, experimental_df, pixel_cols = load_thermal_camera_data(
        control_path, experimental_path
    )

    # Plot sample frames during dispersion
    plot_sample_frames_during_dispersion(
        control_df, experimental_df, pixel_cols, plots_dir
    )

    # Plot temporal analysis
    plot_temporal_analysis(
        control_df, experimental_df, pixel_cols, plots_dir
    )

if __name__ == '__main__':
    explore_thermal_camera_data()
