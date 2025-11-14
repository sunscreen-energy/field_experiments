# Field Experiments Analysis

Analysis code for field experiments measuring temperature effects of calcium carbonate dispersion on cropland.

## Python Files

### Main Analysis Scripts

**drone_footage.py**
Provides the `DroneFootageAnalysis` class for processing drone thermal imagery TIF files, extracting temperature data within defined regions, and analyzing spatial temperature patterns before, during, and after dispersion events. Includes methods for spatial decorrelation, region querying, and visualization of temperature changes.

**plot_tif_dispersion_interval.py**
Creates normalized temperature difference plots comparing drone thermal imagery during the dispersion interval against baseline periods (before and/or after). Overlays sensor locations with PM2.5 concentration labels and the emission site to visualize the spatial cooling effect.

**explore_thermal_cameras.py**
Analyzes continuous thermal camera data from control and experimental vine locations throughout the dispersion event. Generates temporal plots showing differential warming effects between vines under the dispersion plume versus control vines.

### Correlation Analysis

**correlations/correlation_ambient_sensors.py**
Computes correlations between ambient air temperature (mean across all sensors) and pyranometer sensor variables including PM concentrations, humidity, and wind data. Generates scatter plots showing relationships between ambient conditions and measured variables.

**correlations/correlation_drone_sensors.py**
Pairs drone-measured ground temperatures with pyranometer sensor readings at matching locations and times to analyze correlations between ground temperature and sensor variables like PM2.5. Uses spatial and temporal windowing to match measurements from the two data sources.

**correlations/correlation_pyr_voltage.py**
Analyzes correlations between raw pyranometer voltage readings and various environmental measurements to validate sensor calibration and identify relationships with meteorological variables.

### Supporting Files

**sensor_coordinates/sensor_coordinates.py**
Defines sensor location mappings including field corner coordinates, normalized sensor positions, and groupings of sensor IDs by physical location. Used to generate lat/lon coordinates for each sensor device.

**utils.py**
Shared utility functions including sensor data loading, coordinate transformations (lat/lon to UTM), timestamp parsing for drone imagery, statistical functions (mean centering, correlation computation), and plotting helpers. Defines key constants like dispersion start/end times.

## Quickstart

Run the main analysis:
```bash
uv run python drone_footage.py
uv run python plot_tif_dispersion_interval.py
uv run python explore_thermal_cameras.py
```

Generate correlation plots:
```bash
uv run python correlations/correlation_drone_sensors.py
uv run python correlations/correlation_ambient_sensors.py
```

## Data Requirements

- Drone thermal imagery: `data/2025-11-07/drone_imaging/*.tif`
- Pyranometer sensor data: `data/2025-11-07/pyranometer_corrected.csv`
- Thermal camera data: `data/2025-11-07/thermal_cameras/control.csv` and `experimental.csv`
