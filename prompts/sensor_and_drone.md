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
