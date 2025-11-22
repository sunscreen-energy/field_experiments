"""
Classes for working with drone footage data.
Provides utilities for loading, processing, and analyzing drone imagery.
"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
from shapely.geometry import Polygon
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.ticker as ticker
from scipy.spatial import ConvexHull
from PIL import Image

# Project root directory (field_experiments)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DroneFlightPass:
    """
    Represents a single drone flight pass with multiple imaging channels.
    All channels are aligned to a master reference raster's spatial grid
    """
    def __init__(
        self, 
        timestamp: str, 
        master_reference_path: str | Path, 
        data_dir: str | Path, 
        verbose: bool = True
    ):
        """
        Initialize a drone flight pass for a specific timestamp.

        Args:
            timestamp: Timestamp string in format "YYYY-MM-DDTHHMMPST" (e.g., "2025-11-07T1133PST")
            master_reference_path: Path to the master reference raster that defines the target grid
            data_dir: Directory containing drone imagery files
            verbose: If True, print information about available channels and plots alignment
        """
        self.timestamp = timestamp
        self.master_reference_path = Path(master_reference_path)
        self.data_dir = Path(data_dir)

        with rasterio.open(self.master_reference_path) as src:
            self.master_transform = src.transform
            self.master_crs = src.crs
            self.master_shape = (src.height, src.width)
            self.master_profile = src.profile.copy()
            self.master_nodata = src.nodata

        self.data = None
        self.channel_map = {}
        self.transform = self.master_transform
        self.crs = self.master_crs
        self.bounds = None
        self.nodata_mask = None

        self._load_flight_data(verbose)

    def _warp_to_master(self, src_path: Path) -> tuple[np.ndarray, float | None]:
        """
        Warp a source raster to match the master reference grid.

        Args:
            src_path: Path to the source raster file

        Returns:
            Tuple of (aligned_array, nodata_value)
        """
        with rasterio.open(src_path) as src:
            # Determine if this is thermal data (needs float32 for decimal preservation)
            is_thermal = "TIR" in src_path.stem or "Celsius" in src_path.stem
            dest_dtype = np.float32 if is_thermal else src.dtypes[0]

            dest_shape = (src.count, *self.master_shape)
            dest_array = np.zeros(dest_shape, dtype=dest_dtype)
            for band_idx in range(src.count):
                reproject(
                    source=rasterio.band(src, band_idx + 1),
                    destination=dest_array[band_idx],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=self.master_transform,
                    dst_crs=self.master_crs,
                    resampling=Resampling.bilinear
                )

            return dest_array, src.nodata

    def _load_flight_data(self, verbose: bool):
        """Load all TIF files matching this timestamp and align to master grid."""
        pattern = f"{self.timestamp}_ORTHO_*.tif"
        tif_files = sorted(self.data_dir.glob(pattern))

        if not tif_files:
            raise ValueError(f"No TIF files found for timestamp {self.timestamp}")

        all_channels = []
        channel_idx = 0

        with rasterio.open(self.master_reference_path) as src:
            master_data = src.read()
            if master_data.ndim == 2:
                master_data = master_data[np.newaxis, :, :]

            nodata_value = src.nodata
            if nodata_value is not None:
                self.nodata_mask = master_data[0] == nodata_value
            else:
                self.nodata_mask = np.zeros(self.master_shape, dtype=bool)

        # Process all files for this timestamp
        for tif_path in tif_files:
            channel_type = tif_path.stem.split("_ORTHO_")[1]

            # Load and align data
            if tif_path == self.master_reference_path:
                aligned_image = master_data
                nodata_value = self.master_nodata
            else:
                aligned_image, nodata_value = self._warp_to_master(tif_path)

            # Update nodata mask (union of invalid pixels across all channels)
            if nodata_value is not None:
                self.nodata_mask |= (aligned_image[0] == nodata_value)

            num_bands = aligned_image.shape[0]

            if num_bands == 1:
                self.channel_map[channel_type] = channel_idx
                channel_idx += 1
            else:
                for band_idx in range(num_bands):
                    self.channel_map[f"{channel_type}_band{band_idx}"] = channel_idx
                    channel_idx += 1

            all_channels.append(aligned_image)

        self.data = np.concatenate(all_channels, axis=0)

        # Set bounds based on master transform and shape
        self.bounds = rasterio.transform.array_bounds(
            self.master_shape[0], self.master_shape[1], self.transform
        )

        if verbose:
            print(f"Loaded drone flight pass for {self.timestamp}")
            print(f"Available channels ({len(self.channel_map)}):")
            for channel_name, idx in sorted(self.channel_map.items(), key=lambda x: x[1]):
                print(f"  {idx:2d}: {channel_name}")
            print(f"Data shape: {self.data.shape} (channels, height, width)")

    def create_mask_from_polygon(self, polygon_coords: list[tuple[float, float]]) -> np.ndarray:
        """
        Create a binary mask from polygon coordinates.

        Args:
            polygon_coords: List of (lon, lat) coordinate tuples defining polygon vertices

        Returns:
            Binary mask array with True for pixels inside the polygon
        """
        polygon = Polygon(polygon_coords)

        mask = geometry_mask(
            [polygon],
            out_shape=self.master_shape,
            transform=self.transform,
            invert=True
        )

        return mask

    def create_mask_from_feature(self, channel: str, threshold: float, greater_than: bool = False) -> np.ndarray:
        """
        Create a binary mask based on channel values and threshold.

        Args:
            channel: Channel name to use for masking
            threshold: Threshold value
            greater_than: If True, mask where values > threshold; otherwise mask where values <= threshold

        Returns:
            Binary mask array
        """
        if channel not in self.channel_map:
            raise ValueError(f"Channel '{channel}' not found. Available: {list(self.channel_map.keys())}")

        channel_idx = self.channel_map[channel]
        channel_data = self.data[channel_idx]

        if greater_than:
            mask = channel_data > threshold
        else:
            mask = channel_data <= threshold

        return mask

    def query_regions(self, regions: list[list[np.ndarray]], channel: str, mode: str = 'mean') -> list[float]:
        """
        Query statistics for multiple regions in a specific channel.

        Args:
            regions: List of region definitions, where each region is a list of masks to AND together
            channel: Channel name to query
            mode: Aggregation mode ('mean', 'min', 'max', 'median')

        Returns:
            List of aggregated values for each region
        """
        if channel not in self.channel_map:
            raise ValueError(f"Channel '{channel}' not found. Available: {list(self.channel_map.keys())}")

        channel_idx = self.channel_map[channel]
        channel_data = self.data[channel_idx]

        results = []

        for region_masks in regions:
            combined_mask = np.ones(self.master_shape, dtype=bool)
            for mask in region_masks:
                combined_mask &= mask

            combined_mask &= ~self.nodata_mask

            values = channel_data[combined_mask]

            if len(values) == 0:
                results.append(np.nan)
            else:
                if mode == 'mean':
                    results.append(float(np.nanmean(values)))
                elif mode == 'min':
                    results.append(np.nanmin(values))
                elif mode == 'max':
                    results.append(np.nanmax(values))
                elif mode == 'median':
                    results.append(np.nanmedian(values))
                else:
                    raise ValueError(f"Unknown mode: {mode}")

        return results

    def visualize_channel_distribution(self, region: list[np.ndarray], channel: str, bins: int = 50):
        """
        Plot a histogram of channel values within a specified region.

        Args:
            region: List of masks that define the region (combined with bitwise AND)
            channel: Channel name to visualize
            bins: Number of histogram bins (default: 50)
        """
        if channel not in self.channel_map:
            raise ValueError(f"Channel '{channel}' not found. Available: {list(self.channel_map.keys())}")

        channel_idx = self.channel_map[channel]
        channel_data = self.data[channel_idx]

        # Combine masks to define the region
        combined_mask = np.ones(self.master_shape, dtype=bool)
        for mask in region:
            combined_mask &= mask

        # Exclude nodata pixels
        combined_mask &= ~self.nodata_mask

        # Extract values from the region
        values = channel_data[combined_mask]

        if len(values) == 0:
            print(f"Warning: No valid pixels found in region for channel '{channel}'")
            return

        # Create histogram
        fig, ax = plt.subplots(figsize=(10, 6))

        counts, bin_edges, patches = ax.hist(values, bins=bins, edgecolor='black', alpha=0.7)

        ax.set_xlabel(f'{channel} Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{self.timestamp} - {channel} Distribution\n({len(values):,} pixels)', fontsize=14)

        # Add statistics text
        stats_text = f'Mean: {np.mean(values):.3f}\nMedian: {np.median(values):.3f}\nStd: {np.std(values):.3f}'
        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)

        plt.tight_layout()
        plt.show()

    def visualize(self, regions: list[list[np.ndarray]], region_names: list[str],
                channel: str, mean_center: bool = False, colormap: str = "coolwarm",
                vmin: float | None = None, vmax: float | None = None, show_fig: bool = True):
        """
        Visualize a channel with region overlays using Geographic Coordinates.

        Args:
            regions: List of region definitions.
            region_names: Names for each region
            channel: Channel to visualize
            mean_center: If True, subtract mean from data before visualization
            colormap: Matplotlib colormap name
            vmin: Minimum value for colormap (None for auto)
            vmax: Maximum value for colormap (None for auto)
        """
        if channel not in self.channel_map:
            raise ValueError(f"Channel '{channel}' not found. Available: {list(self.channel_map.keys())}")

        channel_idx = self.channel_map[channel]
        channel_data = self.data[channel_idx].copy()

        masked_data = np.ma.masked_where(self.nodata_mask, channel_data)

        if mean_center:
            masked_data = masked_data - np.ma.mean(masked_data)

        fig, ax = plt.subplots(figsize=(14, 10))

        height, width = self.master_shape
        left_lon, top_lat = rasterio.transform.xy(self.transform, 0, 0, offset='center')
        right_lon, bottom_lat = rasterio.transform.xy(self.transform, height, width, offset='center')
        
        geo_extent = [left_lon, right_lon, top_lat, bottom_lat]

        # --- 2. Plot Image with Geographic Extent ---
        im = ax.imshow(masked_data, cmap=colormap, vmin=vmin, vmax=vmax,
                    extent=geo_extent, aspect='auto')

        for region_masks, region_name in zip(regions, region_names):
            combined_mask = np.ones(self.master_shape, dtype=bool)
            for mask in region_masks:
                combined_mask &= mask

            # Get pixel indices (rows, cols)
            points = np.column_stack(np.where(combined_mask))

            if len(points) >= 3:
                hull = ConvexHull(points)
                hull_indices = points[hull.vertices] # These are (row, col) pixels
                
                # --- 3. Convert Polygon Pixels to Lat/Lon ---
                hull_geo_coords = []
                for row, col in hull_indices:
                    lon, lat = rasterio.transform.xy(self.transform, row, col, offset='center')
                    hull_geo_coords.append([lon, lat])
                
                hull_geo_coords = np.array(hull_geo_coords)

                # Create polygon using Longitude (x) and Latitude (y)
                polygon = MplPolygon(hull_geo_coords, fill=False, edgecolor='black', linewidth=2)
                ax.add_patch(polygon)

                # Calculate centroid in Lat/Lon for the text label
                centroid_lon = np.mean(hull_geo_coords[:, 0])
                centroid_lat = np.mean(hull_geo_coords[:, 1])

                ax.text(centroid_lon, centroid_lat, region_name,
                        fontsize=12, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # --- 4. Format Single Axis ---
        ax.set_xlabel('Longitude (°E)', fontsize=12)
        ax.set_ylabel('Latitude (°N)', fontsize=12)
        ax.set_title(f'{self.timestamp} - {channel}', fontsize=14)

        # Format ticks to show precision (decimals)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(channel, fontsize=12)

        plt.tight_layout()
        if show_fig:
            plt.show()

        return fig

    def strip_band(self, polygon_coords: list[tuple[float, float]], num_bands: int) -> list[list[tuple[float, float]]]:
        """
        Divide a polygon vertically into horizontal strips of equal height.

        Args:
            polygon_coords: List of (lon, lat) coordinate tuples defining polygon vertices
            num_bands: Number of horizontal bands to create

        Returns:
            List of polygon coordinate lists, one for each band (intersection of original polygon with each band)
        """
        polygon = Polygon(polygon_coords)
        min_y = min(coord[1] for coord in polygon_coords)
        max_y = max(coord[1] for coord in polygon_coords)

        band_height = (max_y - min_y) / num_bands

        bands = []
        for i in range(num_bands):
            band_min_y = min_y + i * band_height
            band_max_y = min_y + (i + 1) * band_height

            min_x = min(coord[0] for coord in polygon_coords)
            max_x = max(coord[0] for coord in polygon_coords)

            band_rect = Polygon([
                (min_x, band_min_y),
                (max_x, band_min_y),
                (max_x, band_max_y),
                (min_x, band_max_y)
            ])

            intersection = polygon.intersection(band_rect)

            if intersection.is_empty:
                continue
            elif hasattr(intersection, 'exterior'):
                band_coords = list(intersection.exterior.coords[:-1])
            else:
                continue

            bands.append(band_coords)

        return bands


class FlightDifference:
    """
    Computes and visualizes differences between two drone flight passes.
    """

    def __init__(self, before_timestamp: str, after_timestamp: str, master_reference_path: str | Path, data_dir: str | Path | None = None):
        """
        Initialize flight difference analyzer.

        Args:
            before_timestamp: Timestamp for the earlier flight
            after_timestamp: Timestamp for the later flight
            master_reference_path: Path to the master reference raster that defines the target grid
            data_dir: Directory containing drone imagery files (defaults to PROJECT_ROOT/data/2025-11-07/drone_imaging)
        """
        self.before = DroneFlightPass(before_timestamp, master_reference_path, data_dir, verbose=False)
        self.after = DroneFlightPass(after_timestamp, master_reference_path, data_dir, verbose=False)

        print(f"Initialized FlightDifference: {before_timestamp} -> {after_timestamp}")
        print(f"Common channels: {set(self.before.channel_map.keys()) & set(self.after.channel_map.keys())}")

    def create_mask_from_polygon(self, polygon_coords: list[tuple[float, float]]) -> np.ndarray:
        """Create mask from polygon using the before flight pass."""
        return self.before.create_mask_from_polygon(polygon_coords)

    def create_mask_from_feature(self, channel: str, threshold: float, greater_than: bool = False) -> np.ndarray:
        """Create mask from feature using the before flight pass."""
        return self.before.create_mask_from_feature(channel, threshold, greater_than)

    def strip_band(self, polygon_coords: list[tuple[float, float]], num_bands: int) -> list[list[tuple[float, float]]]:
        """Divide polygon into horizontal bands."""
        return self.before.strip_band(polygon_coords, num_bands)

    def query_regions(self, regions: list[list[np.ndarray]], channel: str, mode: str = 'mean') -> list[float]:
        """
        Query regional differences (after - before) for a channel.

        Args:
            regions: List of region definitions
            channel: Channel name to query
            mode: Aggregation mode

        Returns:
            List of difference values for each region
        """
        before_values = self.before.query_regions(regions, channel, mode)
        after_values = self.after.query_regions(regions, channel, mode)

        differences = [after - before for before, after in zip(before_values, after_values)]

        return differences

    def visualize(self, regions: list[list[np.ndarray]], region_names: list[str],
                  channel: str, mean_center: bool = False, colormap: str = "coolwarm",
                  vmin: float | None = None, vmax: float | None = None, show_fig=True):
        """
        Visualize pixel-wise difference between flights.

        Args:
            regions: List of region definitions
            region_names: Names for each region
            channel: Channel to visualize
            mean_center: If True, subtract mean from difference before visualization
            colormap: Matplotlib colormap name
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap
        """
        if channel not in self.before.channel_map or channel not in self.after.channel_map:
            raise ValueError(f"Channel '{channel}' not found in both flights")

        before_idx = self.before.channel_map[channel]
        after_idx = self.after.channel_map[channel]

        before_data = self.before.data[before_idx].astype(float)
        after_data = self.after.data[after_idx].astype(float)

        valid_mask = ~(self.before.nodata_mask | self.after.nodata_mask)

        difference = np.full_like(before_data, np.nan)
        difference[valid_mask] = after_data[valid_mask] - before_data[valid_mask]

        masked_diff = np.ma.masked_invalid(difference)

        if mean_center:
            masked_diff = masked_diff - np.ma.mean(masked_diff)

        fig, ax = plt.subplots(figsize=(14, 10))

        height, width = self.after.master_shape
        left_lon, top_lat = rasterio.transform.xy(self.after.transform, 0, 0, offset='center')
        right_lon, bottom_lat = rasterio.transform.xy(self.after.transform, height, width, offset='center')

        geo_extent = [left_lon, right_lon, top_lat, bottom_lat]

        im = ax.imshow(masked_diff, cmap=colormap, vmin=vmin, vmax=vmax,
                      extent=geo_extent, aspect='auto')

        for region_masks, region_name in zip(regions, region_names):
            combined_mask = np.ones(self.before.master_shape, dtype=bool)
            for mask in region_masks:
                combined_mask &= mask

            points = np.column_stack(np.where(combined_mask))

            if len(points) >= 3:
                hull = ConvexHull(points)
                hull_indices = points[hull.vertices]

                hull_geo_coords = []
                for row, col in hull_indices:
                    lon, lat = rasterio.transform.xy(self.after.transform, row, col, offset='center')
                    hull_geo_coords.append([lon, lat])
                
                hull_geo_coords = np.array(hull_geo_coords)

                polygon = MplPolygon(hull_geo_coords, fill=False, edgecolor='black', linewidth=2)
                ax.add_patch(polygon)

                centroid_row = np.mean(hull_geo_coords[:, 0])
                centroid_col = np.mean(hull_geo_coords[:, 1])

                ax.text(centroid_col, centroid_row, region_name,
                       fontsize=10, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_xlabel('Longitude (°E)', fontsize=12)
        ax.set_ylabel('Latitude (°N)', fontsize=12)
        ax.set_title(f'{channel} Difference: {self.after.timestamp} - {self.before.timestamp}', fontsize=14)

        # Format ticks to show precision (decimals)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.15)
        cbar.set_label(f'{channel} Difference', fontsize=12)

        plt.tight_layout()
        if show_fig:
            plt.show()
        
        return fig
