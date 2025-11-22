Design some classes with utility methods to work with the drone data. These classes will be in `src/drone_footage.py`
Use the rasterio library

## Preprocessing step (will be in `DroneFlightPass` class)
One very important thing to keep in mind is all drone flight passes need to be aligned and have the same shape
`data/2025-11-07/drone_imaging/2025-11-07_block_boundary/2025-11-07_block_boundary.shp` can help you obtain that shape

Here's roughly the approach to take for this

```
import rasterio
import geopandas as gpd
from rasterio.mask import mask

shapefile = gpd.read_file("data/.../2025-11-07_block_boundary.shp")
geoms = shapefile.geometry.values

def preprocess_layer(tif_path, shapes):
    with rasterio.open(tif_path) as src:
        out_image, out_transform = mask(src, shapes, crop=True, nodata=True)
        return out_image, src.profile
```
We need to preprocess all of the tif files that match a single timestamp as a single flight, e.g.
(
    data/2025-11-07/drone_imaging/2025-11-07T1133PST_ORTHO_MS.tif,
    data/2025-11-07/drone_imaging/2025-11-07T1133PST_ORTHO_NDRE.tif,
    data/2025-11-07/drone_imaging/2025-11-07T1133PST_ORTHO_NDVI.tif
)

These might have different resolutions, though, so we can't (yet) concatenate them along the channel dimension.

Here's how to resolve that issue:

Write a script in scratch/ that opens all of the *ORTHO_TIR_Celsius.tif files. Process them with the preprocess_layer function described above and determine the coarsest resolution. Based on this value, define RESOLUTION_HEIGHT and RESOLUTION_WIDTH as constants at the top of the `src/drone_footage.py` file. The file in scratch should not be referred to or imported from in `src/drone_footage.py`


## Single flight class: `DroneFlightPass`
(thermal, multispectral, ndvi, etc).
The first class to design will consider only a single flight
Remember when opening tif files we need to use the preprocessing step with the .shp file.
Every tif file should be opened and manipulated with to RESOLUTION_HEIGHT and RESOLUTION_WIDTH, the constants defined at the top of the file.
For this, good options are rasterio's out_shape or reproject functionality to force the returned array to match (RESOLUTION_HEIGHT, RESOLUTION_WIDTH). Use Resampling.bilinear or Resampling.average depending on what's necessary.

All drone data for a single flight should now have the same H, W. 
Concatenate along the channel dimension for a numpy array (Channels, Height, Width)
Maintain a dictionary of channel_name to index into the channel dimension.
For each timestamp, there will be several different tif files whose values need to be corre

Internally, the class should use coordinates as points of reference. I.e. polygons will be specified in terms of geographical coordinates. 

Upon initialization with verbose=True by default, the class should print what forms of imagery/channels are available. 
The __init__ should also recognize the tif nodata value and keep it as a mask for all future operations.

Here are some methods the class should provide:
`create_mask_from_polygon(self, polygon_coords: list[tuple[float, float]]) --> np.ndarray` 
Makes a mask of the np array based on the polygon coordinates. Make very clean code using `from rasterio.features import geometry_mask` and `from shapely.geometry import Polygon` that translates from geographical coordinate space to pixel space in a neat vectorized manner.

`create_mask_from_feature(self, channel: str, threshold : float, greater_than : bool = False) --> np.ndarray`
It should be possible to create a mask according to a channel and threshold value.

`query_regions(self, regions: list[list[np.ndarray]], channel: str, mode='mean') --> List[float]` 
For each region, identify the mean value of a specific channel. Write code such that modes 'min', 'max', 'median' could be supported with only 1-2 lines of extra code (but do not include them yet)
A region is a `list[np.ndarray]` masks that should be undergo bit-wise AND to define a single set of pixels. We can pass in multiple regions with `regions : list[list[np.ndarray]]`

`visualize(self, regions: list[list[np.ndarray]], region_names : list[str], channel : str, mean_center : bool = False, colormap="coolwarm", vmin=None, vmax=None)`
I should be able to visualize any channel. I should have an option to visualize the channel using any matplotlib colormap
The pixels should have color based on their value for that channel and the colormap and vmin and vmax
However I would also like to visualize the regions, so here is the strategy for that: Take the convex hull of all pixels within a region and add the border onto the plot as well. At the center of each region, the region name should be present.
Use matplotlib.pyplot for visualizations
Center the visualized data with with something like `arr -= arr.mean()` if `mean_center=True`
When visualizing the x and y axis should show both distance in meters and the geographical coordinates. For coordinates keep in mind you'll need 6 decimal points b/c its a small area.
For the dual-axis requirement in visualize: The primary axis should be the indices/pixels converted to relative meters from the top-left. You can use matplotlib.ticker or secondary_xaxis to simply display the corresponding Lat/Lon based on the affine transform bounds. It doesn't need to be a full map projection, just a labeled axis.


`strip_band(self, polygon, num_bands)` that takes in a polygon and divides it vertically into `num_bands` equal height horizontal strips. 
This returns the strips as a list of polyons.

There should be no use of for loops when it is possible to use np vectorized operations.


## Flight difference class `FlightDifference`
This class adds functionality that identifies the difference between two flights. It maintains two `DroneFlightPass` classes, which I'll refer to as components in this description
One flight should be referred to as 'before' internally and one flight should be referred to as 'after'
The initialization of components should happen with verbose = False.

`create_mask_from_polygon`, `strip_band` and `create_mask_from_feature` should call the respective methods in the components
`query_regions` should get the mean for both components and do a (after - before) calculation before returning the results.

`visualize` will take in the same arguments and attempt to show a pixelwise difference.
Ensure mathematical operations handle NaN values gracefully (e.g., using np.nanmean or masking invalid values before subtraction) so that a valid pixel in one flight isn't discarded just because the other flight has a nodata value in that spot


## Light sample usage
With `if __name__ == __main__` you can show some sample usage, but it doesn't need to be too involved/comprehensive


! --- Apply matching ----

We need to make some edits to `src/drone_footage.py`
Lets from a **"Geometric Clipping"** strategy (forcing all images into an abstract bounding box) to a **"Reference Grid Alignment"** strategy.

Instead of defining an arbitrary `RESOLUTION_HEIGHT/WIDTH`, I will pass a specific file path (the "Master Raster," likely a high-res NDVI flight) to the class. I will also pass the data dir with all flights for that day. The class will read that file's spatial grid (transform, CRS, dimensions) and force every subsequent drone flight—regardless of its native resolution—to warp into that exact pixel grid.

-----

### Phase 1: API & Initialization Refactor

  * [ ] **Update `__init__` arguments**

      * **Remove:** `bbox` argument.
      * **Add:** `master_reference_path` (Path/str). This allows you to swap between any flight as the "truth" depending on the date.
      * **Task:** Load and store the `master_profile`, `master_transform`, `master_crs`, and `master_shape` immediately upon instantiation.

  * [ ] **Remove Geometry Creation**

      * **Delete:** The `Polygon(bbox)` logic.
      * **Reason:** We are no longer cropping to a vector; we are warping to a raster extent. Cropping happens implicitly because data outside the master raster's extent becomes `nodata`.

### Phase 2: The Alignment Engine (The Heavy Lifting)

  * [ ] **Create helper: `_warp_to_master(src_path)`**

      * **Goal:** Replace your current `_preprocess_layer`.
      * **Pseudocode:**
        ```python
        def _warp_to_master(self, src_path):
            # 1. Open source file
            # 2. Create destination array (np.float32) with shape = self.master_shape
            # 3. rasterio.warp.reproject(
            #      source=src_band,
            #      destination=dest_array,
            #      dst_transform=self.master_transform,
            #      dst_crs=self.master_crs,
            #      resampling=Resampling.bilinear  <-- Vital for Thermal
            #    )
            # 4. Return aligned_array
        ```

  * [ ] **Handle `dtype` management**

      * **Task:** Ensure the destination array in `_warp_to_master` creates `float32` arrays for Thermal data (to preserve decimals) even if the master raster is `uint8`.

  * [ ] **Implement Sub-pixel Registration**

      * **Goal:** Fix the "few pixels off" error you mentioned using OpenCV.
      * **Function:** `_calculate_pixel_shift(master_arr, worker_arr)`
      * **Pseudocode:**
        ```python
        # 1. Normalize both arrays to 0-1 range
        # 2. Run cv2.phaseCorrelate(master, worker)
        # 3. Return (dx, dy) tuple
        # 4. Apply scipy.ndimage.shift if offset > threshold
        ```
      * *Note:* Only run this on the "structural" channels (e.g., align Thermal to NDVI), not on individual bands of the same flight.

### Phase 3: Data Loading & Organization

  * [ ] **Update `_load_flight_data` loop**

      * **Task:** Iterate through file list.
      * **Condition:** If `filepath == master_reference_path`:
          * Load directly (no warping needed, just read it).
      * **Else:**
          * Call `_warp_to_master(filepath)`.
          * (Optional) Call `_calculate_pixel_shift` and shift the result.

  * [ ] **Refactor Channel Mapping**

      * **Task:** Keep your dictionary logic (`channel_map`), but ensure it tracks the new aligned arrays.

### Phase 4: Masking & Safety

  * [ ] **Refactor `nodata_mask`**

      * **Old way:** Boolean mask based on the first loaded file.
      * **New way:** The mask should primarily derive from the **Master Raster**.
      * **Task:** If `master_raster_value == nodata`, then `mask = True`.
      * **Edge Case:** If a thermal flight covers *less* area than the NDVI flight (smaller swath), update the mask to be the **union** of invalid pixels (if NDVI is invalid OR Thermal is invalid -\> Pixel is invalid).

  * [ ] **Add Alignment Validation (Sanity Check)**

      * **Task:** Add a `verify_alignment()` method.
      * **Action:** Generate a temporary low-res JPG overlaying the Master (Green channel) and a Thermal layer (Red channel) to visually check if trees/rows line up.4
        * Save to plots/{date}/verify_tif_alignment

