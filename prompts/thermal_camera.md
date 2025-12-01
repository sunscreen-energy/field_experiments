As you know, the thermal camera data contains two images, one pointing two crops within the experimental region, and one pointing to crops within the control region. One issue, though is there may have been slight differences in the angle of the thermal camera between control and experimental. It's possible that the control camera captured slightly more "sky" or "ground" pixels versus the experimental camera. Sky pixels tend to be cooler and ground pixels tend to be warmer than leaf pixels, which is what we're actually trying to measure. We'd like to conduct a scientifically rigorous analysis of how much calcium carbonate contributed to reductions in temperatures. 

Let's continue to iterate on the thermal camera temporal plot. First of all, only include the time region, where both the control and experimental cameras are on. The cameras appear to be active from 540000 epoch UTC to 552000 epoch UTC so do not plot or analyze any times outside of this range. Have these time values declared as a constant at the top of the file.

For each pixel a mean should be taken along the time dimension. then in the temporal analysis plot the mean difference from the pixel mean should be computed.

Obviously the control and experimental means should be computed separately.

Within the dispersion window, the mean temperature differences should be computed.
Plot 2 histograms on the same axes, the contents of the histograms should be the pixel differences in temperature from the mean for the control and experimental versions.

! --- 

The file `src/explore_thermal_cameras.py` constructs the graphs we'd like to see from thermal camera readings. However it assumes we only have 1 date/experiment during which we collected thermal camera data.
Thermal camera data should be expected to be in `data/YYYY-MM-DD/thermal_cameras`. Currently only 2025-11-07 has that data, but this will change soon.

There's a lot of hardcoded values in the python script. 
Instead make this a YAML file that goes in the `data/YYYY-MM-DD/thermal_cameras` dir.
The options for this config file should be:
* drop_cloud
    * If true also the start and end time of the cloud
* mean based on after

Just delete the option in the code for KEEP_ONLY_MIDDLE_BAND, its kinda pointless.

All times for the YAML file should be in "YYYY-MM-DD-HH-MM-SS" format, with an option for timezone as well, which will usually be -8 (PST) or -7 PDT. Make a comment in the YAML file explaining this.

Internally, the explore thermal cameras should use the Epoch UTC format, which is seconds since 1970. However this should not be needed for the YAML input.

Output files should go to `plots/thermal_camera/{date}/filename.png`. Use os.makedirs(... exist_ok=True) to ensure that exists. Test your new implementation with the 2025-11-07 data and fix any bugs that arise.