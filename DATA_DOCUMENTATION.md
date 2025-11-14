# Field Experiments Data Documentation

## Overview

This document provides comprehensive documentation of the data collected during the 2025-11-07 field experiment. The experiment involved dispersing CaCO3 on cropland and monitoring temperature and particulate matter distributions using multiple sensor types.
Sensors were clustered into groups at 12 different locations which are shown in the plot that is attached. Sensors may also be referred to as pyranometers in this discussion. Sensors can measure particulate matter and wind and temperature. We also have drone multi-spectral footage of the crop land, which is also in the plot.

## CaCO3 Dispersion Window

The active CaCO3 dispersion occurred during the following time window:

**Start Time:**
- Local: 2025-11-07 11:42:00 PST (UTC-8)
- UTC: 2025-11-07 19:42:00 UTC

**End Time:**
- Local: 2025-11-07 12:50:00 PST (UTC-8)
- UTC: 2025-11-07 20:50:00 UTC

**Duration:** 71.0 minutes (1 hour 11 minutes)

**Important Note on Timezones:**
- The dispersion window times above are given in **PST (Pacific Standard Time, UTC-8)**
- Data files use **PDT (Pacific Daylight Time, UTC-7)** in the `local_iso8601` column (indicated by "-0700" offset)
- **Always use the `epoch_utc` column for time synchronization across datasets** to avoid timezone confusion

This dispersion window is highlighted in visualizations to enable analysis of pre-dispersion, during-dispersion, and post-dispersion conditions.

## Data Directory Structure

```
data/2025-11-07/
├── drone_imaging/          # 10 thermal orthomosaic .tif files
├── pyranometers/           # pyranometer_sensors.csv
└── thermal_cameras/        # control.csv and experimental.csv
```

## 1. Drone Imaging Data

### File Details
- **Location**: `data/2025-11-07/drone_imaging/`
- **File count**: 10 .tif files
- **Format**: GeoTIFF (EPSG:32610 - UTM Zone 10N)
- **File naming**: `YYYY-MM-DDTHHMMPST_ORTHO_TIR_Celsius.tif`

### Time Series
Drone flights occurred at the following times (PST):
1. 11:01 AM
2. 11:09 AM
3. 11:20 AM
4. 11:51 AM
5. 11:59 AM
6. 12:14 PM
7. 12:23 PM
8. 12:32 PM
9. 1:01 PM
10. 1:12 PM

### Spatial Coverage
- **Bounds**: Approximately 620837-621200 meters east, 4247280-4247557 meters north (UTM)
- **Coverage area**: ~360m x ~277m
- **CRS**: EPSG:32610 (WGS 84 / UTM zone 10N)

### Temperature Ranges
- **Minimum**: -9999°C (no-data value used for masked areas)
- **Maximum observed**: 49.19°C (at 1:01 PM)
- **Typical range**: 32-49°C during midday flights

### Key Observations
- Temperature increases throughout the day as expected
- No-data values (-9999) indicate areas outside field coverage
- Peak temperatures occur around 1:01 PM (49.19°C)
- Later flight at 1:12 PM shows cooler temperatures (32.74°C max), possibly due to cloud cover
- Flights at 11:51 AM and 11:59 AM occurred during the CaCO3 dispersion window (11:42 AM - 12:53 PM PST)
- Flights at 12:14 PM, 12:23 PM, 12:32 PM, and 1:01 PM occurred during or immediately after dispersion

### Sensor Overlay
All 94 sensor locations have been mapped to the thermal imagery using coordinate transformation from WGS84 (lat/lon) to UTM Zone 10N. See `plots/tif_with_sensors_*.png` for visualizations.

## 2. Pyranometer Sensor Data

### File Details
- **Location**: `data/2025-11-07/pyranometers/pyranometer_sensors.csv`
- **Total rows**: 1,860,933
- **Number of devices**: 94 unique units (UNIT_01 through UNIT_100, with some gaps)
- **Time range**: 10:58:22 AM - 5:51:18 PM (local time from data files, approximately 7 hours)

### Data Columns

#### Core Measurements (all devices)
- **Device_ID**: Sensor unit identifier (e.g., UNIT_01)
- **epoch_utc**: Unix timestamp (seconds since 1970-01-01)
- **local_iso8601**: Local time in ISO8601 format (PST, UTC-7)
- **t_plus_s**: Seconds since device boot
- **temp_C**: Temperature in Celsius
  - Range: 20.1 - 37.9°C
  - Mean: 23.96°C
  - Std: 1.94°C
- **rh_pct**: Relative humidity percentage
  - Range: 31.2 - 72.4%
  - Mean: 59.4%
  - Std: 4.48%
- **pyr_V**: Pyranometer voltage reading
  - Range: 0.0 - 0.554 V
  - Mean: 0.063 V
  - Std: 0.045 V

#### Calibrated Data
- **slope**: Calibration slope
- **intercept**: Calibration intercept
- **ghi_calibrated**: Calibrated Global Horizontal Irradiance values

#### Particulate Matter (PM) Sensors
Available on subset of devices (18,911 rows with PM data out of 1,860,933 total):
- **sps_pm1_0**: PM1.0 concentration
- **sps_pm2_5**: PM2.5 concentration
- **sps_pm4_0**: PM4.0 concentration
- **sps_pm10**: PM10 concentration
- **sps_nc0_5**: Number concentration 0.5μm
- **sps_nc1_0**: Number concentration 1.0μm
- **sps_nc2_5**: Number concentration 2.5μm
- **sps_nc4_0**: Number concentration 4.0μm
- **sps_nc10**: Number concentration 10μm
- **sps_typ_particle_um**: Typical particle size in micrometers

#### Wind Sensors
Available on 5 devices (75,828 rows with wind data):
- **wind_dir_deg**: Wind direction in degrees
- **wind_dir_cardinal**: Cardinal direction (N, NE, E, SE, etc.)
- **wind_speed_kph**: Wind speed in kilometers per hour

#### Metadata
- **source_file**: Original source file (format: row.column.device_id)

### Sensor Locations
Sensor coordinates are available in `sensor_coordinates/sensor_coordinates.json` in WGS84 format (latitude, longitude).

### Key Observations
- Data collected at ~1 Hz sampling rate
- Temperature shows expected diurnal variation
- Pyranometer voltages correlate with solar irradiance
- PM data shows spike around epoch 1762544520-1762548780 (during CaCO3 dispersion window: 11:42 AM - 12:53 PM PST)
- Wind data provides context for particle dispersion patterns
- Sensor data spans the entire dispersion window and provides before/during/after measurements

## 3. Thermal Camera Data

### File Details
- **Control**: `data/2025-11-07/thermal_cameras/control.csv`
  - Rows: 7,880 frames
  - Time range: 10:36:05 AM - 3:02:00 PM (local time from data files)
  - Duration: ~4.4 hours

- **Experimental**: `data/2025-11-07/thermal_cameras/experimental.csv`
  - Rows: 59,606 frames
  - Time range: 10:33:30 AM (Nov 7) - 8:17:32 PM (Nov 8) (local time from data files)
  - Duration: ~34 hours (7x longer than control)

### Data Format

#### Time Columns
- **epoch_utc**: Unix timestamp (primary time reference)
- **local_iso8601**: Local timezone (may be off by 1 hour, use epoch_utc)
- **t_plus_s**: Seconds since camera boot

#### Thermal Image Data
- **pix_0 through pix_767**: 768 pixel temperature values
  - Image dimensions: 32 × 24 pixels
  - Pixel ordering: Row-major (left-to-right, top-to-bottom)
  - Camera specifications:
    - 110° field of view
    - Distance from vines: ~2.5 meters
    - Target: Grape vines (thermal images of vegetation)

### Temperature Statistics

**Control Camera:**
- Min: 7.14°C
- Max: 35.46°C
- Mean: 23.37°C

**Experimental Camera:**
- Min: 8.10°C
- Max: 34.88°C
- Mean: 19.66°C

### Key Observations
- Thermal cameras capture fine-scale temperature variations in grape vines
- Images show clear thermal patterns with cooler vegetation and warmer backgrounds
- Control region shows shorter monitoring period (ends at 3:02 PM local time)
- Experimental region monitored for extended duration (continues to next day at 8:17 PM local time) to capture full treatment effect
- Some frames show checkered patterns indicating camera initialization or calibration issues
- Mean temperature decreases over time as expected for diurnal cooling
- CaCO3 dispersion window (11:42 AM - 12:53 PM PST, epoch 1762544520-1762548780) falls within both camera recording periods
- During dispersion window (2102 control frames, 2085 experimental frames):
  - Control: mean 24.23°C, std dev 4.99°C, range 9.01-34.20°C
  - Experimental: mean 23.55°C, std dev 5.72°C, range 8.10-34.88°C
- Post-dispersion, experimental region shows gradual cooling into overnight period

### Coordinate System
Unlike drone imaging and pyranometers, thermal camera data is NOT georeferenced. These are fixed-position time-series thermal images of specific vine locations (control vs. experimental).

## Background Context

### Experiment Objectives
Monitor the effects of CaCO3 dispersion on cropland temperature and particulate matter distribution.

### Confounding Variables

1. **Cloud Cover**: May obscure parts of field transiently during tests
2. **Historical CaCO3 Deposition**: Areas with previous tests have higher reflectivity
3. **Drone Flight Time**:
   - Complete orthomosaic: ~8 minutes per pass
   - East-West leg: ~40-48 seconds
   - Time lag means first regions may appear cooler than final regions
4. **PM Sensor Limitations**: Areas can be shaded without particles nearby, though sun position (south of field) should ensure plume falls on sensors while shading them

## Visualization Outputs

All plots are saved in the `plots/` directory:

### Drone Imaging with Sensors
- **Files**: `tif_with_sensors_*.png` (10 files)
- **Content**: Thermal orthomosaics with sensor locations overlaid
- **Features**:
  - Cyan markers indicate sensor positions
  - Sensor numbers labeled
  - Temperature colormap (hot)
  - Coordinate system properly transformed

### Pyranometer Exploration
- **File**: `pyranometer_exploration.png`
- **Subplots**:
  1. Temperature over time (5 sample devices)
  2. Relative humidity over time
  3. Pyranometer voltage readings
  4. Calibrated GHI values
  5. PM2.5 concentrations
  6. Wind speed measurements

### Thermal Camera Analysis
- **Sample Frames**: `thermal_camera_sample_frames.png`
  - 6 sample frames each from control and experimental cameras
  - Shows temporal evolution of thermal patterns

- **Temporal Analysis**: `thermal_camera_temporal_analysis.png`
  - Mean temperature over time (control vs experimental overlaid)
  - Temperature variability (std dev) over time (control vs experimental overlaid)
  - CaCO3 dispersion window highlighted in green (11:42 AM - 12:53 PM PST)

## Data Access Scripts

Exploration scripts are available in `scratch/`:

1. **plot_tif_with_sensors.py**: Visualize drone thermal imagery with sensor locations
2. **explore_pyranometer.py**: Analyze pyranometer time series data
3. **explore_thermal_cameras.py**: Explore thermal camera imagery

Run with: `uv run python scratch/<script_name>.py`

## Notes for Analysis

1. Use `epoch_utc` for time synchronization across all data sources
2. Coordinate transformation required for overlaying sensor data on drone imagery
3. PM data is sparse - available on subset of sensors only
4. Wind data available on 5 sensors only
5. Thermal camera data is not georeferenced - position information must come from experimental setup documentation
6. No-data values in drone imagery are represented as -9999°C
7. Some thermal camera frames show artifacts - quality filtering may be needed

