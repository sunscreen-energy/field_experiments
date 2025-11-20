import numpy as np
import matplotlib.pyplot as plt
import rasterio
import rasterio.mask
import rasterio.transform
import rasterio.windows

from pathlib import Path
from scipy.stats import linregress
from shapely.geometry import mapping, Polygon, box
from utils import (
    parse_tif_timestamp,
    latlon_to_utm,
    compute_regression_pvalue,
    DISPERSION_START_EPOCH,
    DISPERSION_END_EPOCH
)

CORNERS = {
    'topleft':  (38.367917, -121.616448),
    'topright': (38.367281, -121.613817),
    'botleft':  (38.365171, -121.616451),
    'botright': (38.365120, -121.613840)
}

BUFFER_METERS = 20

# Emission site coordinates (UTM)
EMISSION_SITE_X = 620915
EMISSION_SITE_Y = 4247510


class DroneFootageAnalysis:
    def __init__(self, corners, buffer_m, tif_pattern, dispersion_start_epoch, dispersion_end_epoch):
        """
        Initialize drone footage analysis.

        Args:
            corners: Dict with corner coordinates in lat/lon format
            buffer_m: Buffer distance in meters (inward buffer applied)
            tif_pattern: Glob pattern for TIF files (e.g., 'data/2025-11-07/drone_imaging/*.tif')
            dispersion_start_epoch: Epoch timestamp for start of dispersion
            dispersion_end_epoch: Epoch timestamp for end of dispersion
        """
        self.corners = corners
        self.buffer_m = buffer_m
        self.dispersion_start = dispersion_start_epoch
        self.dispersion_end = dispersion_end_epoch

        self.polygon = self._create_buffered_polygon()

        tif_files = sorted(Path().glob(tif_pattern))
        if not tif_files:
            raise ValueError(f"No TIF files found matching pattern: {tif_pattern}")

        self.data = {'before': [], 'during': [], 'after': []}
        self.timestamps = {'before': [], 'during': [], 'after': []}

        self._load_data(tif_files)

        self.decorrelation_params = {
            'latitude': None,
            'longitude': None
        }

        self.query_regions = []

    def _create_buffered_polygon(self):
        """Create a polygon from corners and apply an inward buffer."""
        corner_order = ['topleft', 'topright', 'botright', 'botleft']
        utm_coords = [latlon_to_utm(self.corners[c][0], self.corners[c][1]) for c in corner_order]

        polygon = Polygon(utm_coords)
        buffered_polygon = polygon.buffer(-self.buffer_m)

        print(f"Original area: {polygon.area:.0f} m²")
        print(f"Buffered area: {buffered_polygon.area:.0f} m²")

        return buffered_polygon

    def _load_data(self, tif_files):
        """Load and aggregate TIF data into before/during/after timeframes."""
        print(f"\nLoading {len(tif_files)} TIF files...")

        for tif_file in tif_files:
            epoch, dt = parse_tif_timestamp(tif_file)

            if epoch < self.dispersion_start:
                timeframe = 'before'
            elif epoch <= self.dispersion_end:
                timeframe = 'during'
            else:
                timeframe = 'after'

            print(f"  {timeframe.upper()}: {Path(tif_file).name} (epoch: {epoch}, time: {dt})")

            temps = self._extract_temps_in_polygon(tif_file)
            self.data[timeframe].extend(temps)
            self.timestamps[timeframe].append(epoch)

        for timeframe in ['before', 'during', 'after']:
            print(f"{timeframe.upper()}: {len(self.data[timeframe])} measurements from {len(self.timestamps[timeframe])} passes")

    def _extract_temps_in_polygon(self, tif_path):
        """
        Extract all temperature values within the polygon using
        efficient vectorized masking.
        """

        with rasterio.open(tif_path) as src:
            polygon_geojson = [mapping(self.polygon)]
            try:
                out_image, out_transform = rasterio.mask.mask(src, polygon_geojson, crop=True)
            
            except ValueError as e:
                if "Input shapes do not overlap" in str(e):
                    return []
                else:
                    raise

            nodata_val = src.nodata # Value for missing data
            data = out_image.squeeze(0)

            if nodata_val is not None:
                final_mask = (data != nodata_val)
            else:
                final_mask = True

            rows, cols = np.where(final_mask)

            if not rows.size:
                return []

            temps = data[rows, cols]

            xs, ys = rasterio.transform.xy(out_transform, rows, cols, offset='center')

            return list(zip(xs, ys, temps))

    def extract_temps_near_sensor(self, tif_path, sensor_lat, sensor_lon, radius_m=10, apply_decorrelation=True):
        """
        Extract temperature values within a square window around a sensor location from a TIF file.
        Optionally apply spatial decorrelation.

        Args:
            tif_path: Path to the TIF file
            sensor_lat: Sensor latitude
            sensor_lon: Sensor longitude
            radius_m: Half-width of square in meters (Manhattan distance, default: 10)
            apply_decorrelation: Whether to apply spatial decorrelation (default: True)

        Returns:
            numpy array of temperature values within square window, or empty array if none found
        """
        with rasterio.open(tif_path) as src:
            sensor_x, sensor_y = latlon_to_utm(sensor_lat, sensor_lon, src.crs)
            transform = src.transform

            col, row = ~transform * (sensor_x, sensor_y)
            col, row = int(col), int(row)

            if col < 0 or col >= src.width or row < 0 or row >= src.height:
                return np.array([])

            pixel_width = abs(transform.a)
            pixel_radius = int(np.ceil(radius_m / pixel_width))

            row_start = max(0, row - pixel_radius)
            row_end = min(src.height, row + pixel_radius + 1)
            col_start = max(0, col - pixel_radius)
            col_end = min(src.width, col + pixel_radius + 1)

            if row_end <= row_start or col_end <= col_start:
                return np.array([])

            window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
            data = src.read(1, window=window)

            rows, cols = np.meshgrid(
                np.arange(row_start, row_end),
                np.arange(col_start, col_end),
                indexing='ij'
            )

            xs, ys = rasterio.transform.xy(transform, rows.flatten(), cols.flatten(), offset='center')
            x_coords = np.array(xs)
            y_coords = np.array(ys)
            temps = data.flatten()

            valid_mask = temps > -1000
            temps = temps[valid_mask]
            x_coords = x_coords[valid_mask]
            y_coords = y_coords[valid_mask]

            if apply_decorrelation and len(temps) > 0 and hasattr(self, 'x_ref') and hasattr(self, 'y_ref'):
                if self.decorrelation_params['latitude'] is not None:
                    params = self.decorrelation_params['latitude']
                    y_meters = y_coords - self.y_ref
                    temps = temps - (params['slope'] * y_meters)

                if self.decorrelation_params['longitude'] is not None:
                    params = self.decorrelation_params['longitude']
                    x_meters = x_coords - self.x_ref
                    temps = temps - (params['slope'] * x_meters)

        return temps

    def decorrelate_spatial(self, timeframes=['before'], direction='latitude'):
        """
        Compute correlation with spatial direction and store decorrelation parameters.

        Args:
            timeframes: List of timeframes to include ('before', 'during', 'after')
            direction: Direction to decorrelate ('latitude' or 'longitude')
        """
        if direction not in ['latitude', 'longitude']:
            raise ValueError(f"Invalid direction: {direction}, must be 'latitude' or 'longitude'")

        all_data = []
        for tf in timeframes:
            all_data.extend(self.data[tf])

        if len(all_data) == 0:
            print("No data available for decorrelation")
            return

        x_utm = np.array([d[0] for d in all_data])
        y_utm = np.array([d[1] for d in all_data])
        temps = np.array([d[2] for d in all_data])

        if not hasattr(self, 'x_ref') or not hasattr(self, 'y_ref'):
            self.x_ref = np.mean(x_utm)
            self.y_ref = np.mean(y_utm)
            print(f"\nSet reference coordinates: x_ref={self.x_ref:.2f}, y_ref={self.y_ref:.2f}")

        if direction == 'latitude':
            d_meters = y_utm - self.y_ref
        elif direction == 'longitude':
            d_meters = x_utm - self.x_ref

        slope, intercept, r_value, p_value, std_err = linregress(d_meters, temps)
        r_squared = r_value ** 2
        n_points = len(d_meters)

        p_value_computed = compute_regression_pvalue(r_squared, n_points)

        self.decorrelation_params[direction] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value_computed,
            'n': n_points
        }

        print(f"\n{direction} decorrelation (timeframes: {timeframes}):")
        print(f"  Line of best fit: y = {slope:.6f}x + {intercept:.3f}")
        print(f"  R² = {r_squared:.6f}")
        print(f"  p-value = {p_value_computed:.4e}")
        print(f"  n = {n_points}")

        return self.decorrelation_params

    def query(self, x_min, x_max, y_min, y_max, name=None):
        """
        Query a rectangular region and compute mean temperature.

        Args:
            x_min, x_max, y_min, y_max: UTM coordinates defining rectangle
            name: Optional name for the region
        """

        per_timeframe_temps = {}
        region_box = box(x_min, y_min, x_max, y_max)

        for timeframe in ['before', 'during', 'after']:
            x_utm = np.array([d[0] for d in self.data[timeframe]])
            y_utm = np.array([d[1] for d in self.data[timeframe]])
            temps = np.array([d[2] for d in self.data[timeframe]])

            mask = (
                (x_utm >= x_min) & (x_utm <= x_max) &
                (y_utm >= y_min) & (y_utm <= y_max)
            )
            temps_in_region = temps[mask]
            x_utm = x_utm[mask]
            y_utm = y_utm[mask]

            if self.decorrelation_params['latitude'] is not None:
                params = self.decorrelation_params['latitude']
                y_meters = y_utm - self.y_ref
                temps_in_region -= (params['slope'] * y_meters)

            if self.decorrelation_params['longitude'] is not None:
                params = self.decorrelation_params['longitude']
                x_meters = x_utm - self.x_ref
                temps_in_region -= (params['slope'] * x_meters)

            if len(temps_in_region) > 0:
                per_timeframe_temps[timeframe] = np.mean(temps_in_region)

        region_name = name or f"Region {len(self.query_regions) + 1}"
        print(f"\n{region_name}:")
        for timeframe, mean_temp in per_timeframe_temps.items():
            print(f"  {timeframe.upper()}: Mean temp = {mean_temp:.3f}°C")

        self.query_regions.append({
            'name': region_name,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'box': region_box,
            'temps': per_timeframe_temps
        })

        return per_timeframe_temps

    def plot(self, timeframe='before', output_path='plots/drone_footage_analysis.png'):
        """
        Plot temperature data with query regions highlighted.

        Args:
            timeframe: Which timeframe to plot ('before', 'during', 'after')
            output_path: Path to save the plot
        """

        data = self.data[timeframe]

        if len(data) == 0:
            print(f"No data available for timeframe: {timeframe}")
            return

        x_utm = np.array([d[0] for d in data])
        y_utm = np.array([d[1] for d in data])
        temps = np.array([d[2] for d in data])
        subsample_mask = np.random.uniform(size=x_utm.shape) < 1e6 / x_utm.shape[0]
        x_utm, y_utm, temps = x_utm[subsample_mask], y_utm[subsample_mask], temps[subsample_mask]

        decorrelated_temps = temps.copy()

        if self.decorrelation_params['latitude'] is not None:
            params = self.decorrelation_params['latitude']
            y_meters = y_utm - self.y_ref
            decorrelated_temps -= (params['slope'] * y_meters)

        if self.decorrelation_params['longitude'] is not None:
            params = self.decorrelation_params['longitude']
            x_meters = x_utm - self.x_ref
            decorrelated_temps -= (params['slope'] * x_meters)

        fig, ax = plt.subplots(figsize=(12, 10))

        scatter = ax.scatter(
            x_utm, y_utm, c=decorrelated_temps, s=1,
            cmap='hot', alpha=0.5, edgecolors='none', rasterized=True
        )

        for region in self.query_regions:
            box_obj = region['box']
            x_coords, y_coords = box_obj.exterior.xy
            ax.plot(x_coords, y_coords, 'cyan', linewidth=2)
            if timeframe == 'during':
                text = f"{region['name']}\n+{region['temps']['during'] - region['temps']['before']:.3f}°C"
            elif timeframe == 'after':
                text = f"{region['name']}\n{region['temps']['after'] - region['temps']['during']:.3f}°C"
            else:
                text = region['name']
            ax.text((region['x_min'] + region['x_max']) / 2,
                   (region['y_min'] + region['y_max']) / 2,
                   text, 
                   color='cyan', fontsize=10,
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        poly_x, poly_y = self.polygon.exterior.xy
        ax.plot(poly_x, poly_y, 'lime', linewidth=2, linestyle='--', label='Analysis region')

        # Plot emission site
        ax.scatter(EMISSION_SITE_X, EMISSION_SITE_Y, c='red', s=200, marker='*',
                  edgecolors='white', linewidths=2, alpha=1.0, zorder=10,
                  label='Emission Site')

        cbar = plt.colorbar(scatter, ax=ax)
        decorr_label = []
        if self.decorrelation_params['latitude'] is not None:
            decorr_label.append('latitude-decorrelated')
        if self.decorrelation_params['longitude'] is not None:
            decorr_label.append('longitude-decorrelated')

        if decorr_label:
            cbar.set_label(f'Temperature (C) ({", ".join(decorr_label)})', fontsize=12)
        else:
            cbar.set_label('Temperature (C)', fontsize=12)

        ax.set_xlabel('UTM X (meters)', fontsize=12)
        ax.set_ylabel('UTM Y (meters)', fontsize=12)
        ax.set_title(f'Drone Footage Analysis - {timeframe.upper()} Dispersion', fontsize=14)
        ax.legend(fontsize=10)
        ax.set_aspect('equal')

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nPlot saved to {output_path}")


if __name__ == '__main__':
    analysis = DroneFootageAnalysis(
        corners=CORNERS,
        buffer_m=BUFFER_METERS,
        tif_pattern='data/2025-11-07/drone_imaging/*.tif',
        dispersion_start_epoch=DISPERSION_START_EPOCH,
        dispersion_end_epoch=DISPERSION_END_EPOCH
    )
    analysis.decorrelate_spatial(timeframes=['before'], direction='latitude')
    analysis.decorrelate_spatial(timeframes=['before'], direction='longitude')

    experimental_region_1 = analysis.query(
        x_min=620935,
        x_max=620980,
        y_min=4247400,
        y_max=4247480,
        name='Experimental 1'
    )
    mixed_region_1 = analysis.query(
        x_min=620905,
        x_max=620930,
        y_min=4247360,
        y_max=4247475,
        name='Mixed 1'
    )
    mixed_region_2 = analysis.query(
        x_min=621000,
        x_max=621062,
        y_min=4247400,
        y_max=4247457,
        name='Mixed 2'
    )
    mixed_region_3 = analysis.query(
        x_min=620935,
        x_max=620980,
        y_min=4247350,
        y_max=4247390,
        name='Mixed 3'
    )
    mixed_region_4 = analysis.query(
        x_min=621000,
        x_max=621070,
        y_min=4247350,
        y_max=4247390,
        name='Mixed 4'
    )
    control_region_1 = analysis.query(
        x_min=620915,
        x_max=621058,
        y_min=4247300,
        y_max=4247335,
        name='Control 1'
    )

    analysis.plot(timeframe='before', output_path='plots/drone_footage_analysis/before.png')
    analysis.plot(timeframe='during', output_path='plots/drone_footage_analysis/during.png')
    analysis.plot(timeframe='after', output_path='plots/drone_footage_analysis/after.png')