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

def calculate_effective_sample_size(data):
    """
    Calculate Effective Sample Size (N_eff) correcting for temporal autocorrelation.
    Uses the Lag-1 autocorrelation coefficient (rho).
    Formula: N_eff = N * (1 - rho) / (1 + rho)
    """
    n = len(data)
    if n < 2:
        return n
    
    # Calculate Lag-1 autocorrelation
    rho = data.autocorr(lag=1)
    
    # If rho is negative or negligible, just use N (conservative)
    # If rho is very high (~1), N_eff approaches 0.
    if pd.isna(rho) or rho < 0:
        rho = 0
        
    n_eff = n * (1 - rho) / (1 + rho)
    return max(1.0, n_eff) # Ensure at least 1 degree of freedom


def analyze_dispersion_effect(control_df, experimental_df, pixel_cols, plots_dir, config):
    """
    Performs Difference-in-Differences (DiD) analysis and plots the results.
    Corrects p-values for temporal autocorrelation.
    """
    print(f"\n--- Statistical Analysis (Difference-in-Differences) ---")

    # 1. Calculate Spatial Means (Frame-level averages)
    # We only care about the aggregate frame temperature for the statistical test
    control_df = control_df.copy()
    experimental_df = experimental_df.copy()
    
    control_df['spatial_mean'] = control_df[pixel_cols].mean(axis=1)
    experimental_df['spatial_mean'] = experimental_df[pixel_cols].mean(axis=1)

    # 2. Merge dataframes on timestamp to ensure exact frame alignment
    # (Assuming timestamps are close enough or using 'epoch_utc' for exact match)
    merged_df = pd.merge(
        control_df[['epoch_utc', 'spatial_mean']], 
        experimental_df[['epoch_utc', 'spatial_mean']], 
        on='epoch_utc', 
        suffixes=('_ctrl', '_exp')
    )

    # 3. Create the "Difference Series" (Experimental - Control)
    # This removes shared environmental noise (e.g., sun going behind a cloud affects both)
    merged_df['diff'] = merged_df['spatial_mean_exp'] - merged_df['spatial_mean_ctrl']

    # 4. Define Time Windows
    # Pre-Dispersion: Start of active camera -> Start of Dispersion
    # Dispersion: Start of Dispersion -> End of Dispersion
    pre_mask = (merged_df['epoch_utc'] < config['dispersion_start_epoch'])
    disp_mask = (merged_df['epoch_utc'] >= config['dispersion_start_epoch']) & \
                (merged_df['epoch_utc'] <= config['dispersion_end_epoch'])

    pre_data = merged_df.loc[pre_mask, 'diff']
    disp_data = merged_df.loc[disp_mask, 'diff']

    if len(pre_data) == 0 or len(disp_data) == 0:
        print("Error: Not enough data in Pre or During windows for analysis.")
        return

    # 5. Calculate Statistics
    mean_pre = pre_data.mean()
    mean_disp = disp_data.mean()
    
    # The "Effect Size" is the shift in the difference
    did_estimate = mean_disp - mean_pre 

    print(f"Time Windows:")
    print(f"  Pre-Dispersion N frames: {len(pre_data)}")
    print(f"  Dispersion N frames:     {len(disp_data)}")
    
    print(f"\nTemperature Deltas (Experimental - Control):")
    print(f"  Baseline Delta (Pre):    {mean_pre:.3f} C")
    print(f"  Dispersion Delta:        {mean_disp:.3f} C")
    print(f"  Observed Impact (DiD):   {did_estimate:.3f} C")

    # 6. Statistical Significance with Autocorrelation Correction
    # We compare the distribution of the 'diff' variable Pre vs During.
    
    # Calculate Effective Sample Sizes
    n_eff_pre = calculate_effective_sample_size(pre_data)
    n_eff_disp = calculate_effective_sample_size(disp_data)
    
    print(f"\nAutocorrelation Correction:")
    print(f"  Pre-dispersion N_eff:    {n_eff_pre:.1f} (Raw: {len(pre_data)})")
    print(f"  Dispersion N_eff:        {n_eff_disp:.1f} (Raw: {len(disp_data)})")
    
    # Welch's t-test using summary statistics and N_eff
    # t = (mean1 - mean2) / sqrt(var1/N1 + var2/N2)
    var_pre = pre_data.var(ddof=1)
    var_disp = disp_data.var(ddof=1)
    
    std_error = np.sqrt((var_pre / n_eff_pre) + (var_disp / n_eff_disp))
    t_stat = (mean_disp - mean_pre) / std_error
    
    # Degrees of freedom (Welch-Satterthwaite equation)
    # Simplified approximation usually sufficient, but let's be rigorous:
    num = ((var_pre / n_eff_pre) + (var_disp / n_eff_disp))**2
    den = ((var_pre / n_eff_pre)**2 / (n_eff_pre - 1)) + \
          ((var_disp / n_eff_disp)**2 / (n_eff_disp - 1))
    df_eff = num / den
    
    # Two-tailed p-value
    p_value = stats.t.sf(np.abs(t_stat), df_eff) * 2

    print(f"\nSignificance Test (Welch's t-test on Difference Series):")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value:     {p_value:.6f}")
    
    if p_value < 0.05:
        print("  Result: SIGNIFICANT change in temperature relationship.")
    else:
        print("  Result: NO significant change detected.")

    # 7. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Raw Temperatures
    ax1.plot(merged_df['epoch_utc'], merged_df['spatial_mean_ctrl'], label='Control', color='blue', alpha=0.6)
    ax1.plot(merged_df['epoch_utc'], merged_df['spatial_mean_exp'], label='Experimental', color='red', alpha=0.6)
    
    # Add shading for dispersion
    ax1.axvspan(config['dispersion_start_epoch'], config['dispersion_end_epoch'], 
                color='gray', alpha=0.2, label='Dispersion Window')
    
    ax1.set_ylabel('Mean Frame Temp (°C)')
    ax1.set_title('Raw Thermal Camera Temperatures')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference Series (The variable we actually tested)
    ax2.plot(merged_df['epoch_utc'], merged_df['diff'], color='purple', alpha=0.8, label='Exp - Ctrl')
    
    # Plot mean lines
    # Pre-mean line
    pre_epochs = merged_df.loc[pre_mask, 'epoch_utc']
    if len(pre_epochs) > 0:
        ax2.hlines(mean_pre, pre_epochs.min(), pre_epochs.max(), colors='green', linestyles='--', lw=2, label=f'Baseline Mean ({mean_pre:.2f}C)')
        
    # Disp-mean line
    disp_epochs = merged_df.loc[disp_mask, 'epoch_utc']
    if len(disp_epochs) > 0:
        ax2.hlines(mean_disp, disp_epochs.min(), disp_epochs.max(), colors='orange', linestyles='--', lw=2, label=f'Dispersion Mean ({mean_disp:.2f}C)')

    ax2.axvspan(config['dispersion_start_epoch'], config['dispersion_end_epoch'], 
                color='gray', alpha=0.2)
    
    ax2.set_ylabel('Temp Difference (Exp - Ctrl) (°C)')
    ax2.set_xlabel('Epoch UTC')
    ax2.set_title(f'Difference Analysis (DiD Estimate: {did_estimate:.3f}°C)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = plots_dir / 'thermal_did_analysis.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved analysis plot to {output_path}")


def explore_thermal_camera_data(date):
    config = load_config(date)
    
    control_path = Path(f'data/{date}/thermal_cameras/control.csv')
    experimental_path = Path(f'data/{date}/thermal_cameras/experimental.csv')
    plots_dir = Path(f'plots/thermal_camera/{date}')
    plots_dir.mkdir(parents=True, exist_ok=True)

    control_df, experimental_df, pixel_cols = load_thermal_camera_data(
        control_path, experimental_path, config
    )

    plot_sample_frames_during_dispersion(
        control_df, experimental_df, pixel_cols, plots_dir, config
    )

    analyze_dispersion_effect(
        control_df, experimental_df, pixel_cols, plots_dir, config
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explore thermal camera data for a specific date')
    parser.add_argument('date', type=str, help='Date in format YYYY-MM-DD')
    args = parser.parse_args()

    explore_thermal_camera_data(args.date)