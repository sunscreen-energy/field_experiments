! - Drone only - !

Read field_experiments/scratch/DATA_DOCUMENTATION.md to get context for what we're trying to do in this repo

Let's focus on the drone imaging data. There are 3 time intervals of importance to us: The time during dispersion, the time before dispersion, and the time after dispersion.

Read `scratch/plot_tif_with_sensors.py` to get a sense of how we're visualizing drone data.
Write a new file `scratch/plot_tif_dispersion_interval.py`. 
Implement roughly the following pseudocode

```
# Let mask_of_all_pix be all pixels that are valid in every file
# Let im be the tensor with temperature readings

captured_pix = im[mask_of_all_pix] # Filter for only valid pixels

temporal_difference = captured_pix[times WITHIN dispersion interval].mean(axis = time_axis) -  captured_pix[times BEFORE dispersion interval].mean(axis = time_axis)
temporal_difference -= temporal_difference.mean()

plot(temporal_difference) # as a single PNG --> `tif_dispersion_effect.png`
```
Also include a constant flag at the top of the file.
Instead of subtracting `captured_pix[times BEFORE dispersion interval].mean(axis = time_axis)`, I may consider subtracting `captured_pix[times OUTSIDE (before and after) dispersion interval].mean(axis = time_axis)`. Make a small if else statement based on the flag that lets me toggle between these behaviors

You can keep the points indicating the location of the pyranometer. However restructure the code so that it acknowledges that there are multiple sensors in each location. The label of each sensor location should be the average PM 2.5 concentration at that location, within the dispersion interval from the sensors that can read PM concentrations.

! If this works we'd expect to see a "cloud of cooling".

! drone and sensor correlation

Read `field_experiments/DATA_DOCUMENTATION.md` to get context for what we're trying to do in this repo

Read `plot_tif_dispersion_interval.py` because there's useful functions there.
Lets conduct a test of correlation between each of the pyranometer variables and the temperature variables.

There are basically 2 temperature variables:

Each set of sensors reads temperature, and their mean at a given time can be taken. This is ambient air temperature.

For each drone pass, there are ground temperature readings. Consider only the temperature readings within a 10 m radius of the sensors. Capture all of the pyranometer measurements at +/- 1 min from these times, compute the mean, and identify correlations between variables of the pyranometer and drone temperature measurements.  

In both cases, handle missing data gracefully

Write three python scripts in the current directory for this task:
correlation_ambient_sensors.py # Computes correlations of all pyranometer variables to the pyranometer temperature measurements. Outputs should be scatterplots. Plots should go in plots/ambient_air_correlations/{variable}_r_squared_{r_squared_value}.png  

correlation_drone_sensors.py # Computes correlations of all pyranometer variables to drone temperature measurements. Outputs should be scatterplots. Plots should go in plots/drone_correlations/{variable}_r_squared_{r_squared_value}.png  

utils.py # Shared functions. You can refactor `plot_tif_dispersion_interval.py` to use this utils.py file too, so there's less code duplication

At each time step, the temperature will mostly be dependent on time of day. So make sure you mean center the temperature readings before constructing correlations with each of the pyranometer variables.

! What if there is correlation with N/S

Read `field_experiments/DATA_DOCUMENTATION.md` to get context for what we're trying to do in this repo. I'm curious if there is correlation between the lattitude and the temperature. I'm also curious if theres a correlation between the longitude and the temperature. Here, when I refer to temperature, I mean the temperature measured by the drone in the tif files.
Read `utils.py` to see if there's any helpful helper functions you can use. 

All of these correlations should be done with drone data prior to the dispersion interval.

The correlation plot should be similar to those produced by the other files matching correlation*.py

The x axis should be meters. The y axis should be temperature.

Write a helper fn in utils To determine if a given, R squared value and N are statistically significant. add this information in the form of p value to the plot.

The corners of the region I wish to examine are given by: 

corners = {
    'topleft':  (38.367917, -121.616448),
    'topright': (38.367281, -121.613817),
    'botleft':  (38.363171, -121.616451),
    'botright': (38.363120, -121.613840)
}

Which you can declare as a constant at the top of the file. The edges are surrounded by rivers and other confounding variables, so please add a buffer region of 20 m to shrink the effective polygon.
We should consider only the area where the drone footage overlaps this shrunken polygon


! De-correlate data by latitude, organize into classes

It appears there is some signficant correlation with latitude and longitude.

1. mv correlation_lat_lon_temperature.py to drone_footage.py

2. create a class within drone footage analysis that takes in as input the polygon region, buffer distance, .tif file pattern, and timestamps for dispersion interval.

3. The class should be instantiated in the `if __name__=="__main__"` section

4. the class should store an aggregation of the tif data prior to, during, and after the dispersion interval. It should also store the timestamps of each ofc.

5. The class should have a method(s) to compute the correlation w.r.t longitude and/or latitude and de-correlate the data by using the line of best fit. There should be an option to determine correlation using any number of {before, during, after} the dispersion interval. It should not plot the correlations with the longitude or latitude via scatteplot, but should instead print the line of best fit, p value, n, and r squared.

6. The class should allow querying rectangular regions. When a rectangular region is specified, the mean temperature within that region should be printed. Also, the region should be saved for plotting.

7. The class should have a plot method. The plot method plots the rectangular query regions and the temperature data (de-correlated by longitude or latitude if necessary)

8. You may rely on any functions in `utils.py`

9. The end of the file should have the following logic

```pseudocode
if __name__ == "__main__":
    class = InstantiateClass(args from constants in original file or docs)
    class.decorrelate_lattitude(timeframes=["before"])
    class.decorrelate_longitude(timeframes=["before"])
    class.query(some sample region)
    class.query(some other sample region)
    class.plot()

```