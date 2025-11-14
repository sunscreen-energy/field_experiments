# Field Experiment Data Analysis
## Methodology: Region Definition

The primary analysis method involved constructing regions based on their likely influence from the aerosol dispersion event. Weather sensors indicated primarily **northwesterly winds** blowing across the field during the dispersion. Based on this wind data, we constructed six regions: **Experimental 1**, **Mixed 1-4** and **Control 1**, ranked by their potential exposure to the dispersion plume.

---
## Temperature Gains During Experiment:
<img src="plots/drone_footage_analysis/during.png" alt="Temperature gains during experiment" width="570">

## Temperature Analysis Results

The following table details the mean temperatures before, during, and after the dispersion event for each defined region. It also includes the calculated temperature change for the "During" and "After" periods.

| Region | Temp (Before) | Temp (During) | Temp (After) | Change (During - Before) | Change (After - During) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Experimental 1** | 12.364 C | 13.880 C | 12.320 C | **+1.516 C** | **-1.560 C** |
| **Mixed 1** | 11.828 C | 13.507 C | 11.281 C | +1.679 C | -2.226 C |
| **Mixed 2** | 12.386 C | 14.011 C | 12.192 C | +1.625 C | -1.819 C |
| **Mixed 3** | 12.433 C | 14.619 C | 10.956 C | +2.186 C | -3.663 C |
| **Mixed 4** | 12.435 C | 14.491 C | 11.033 C | +2.056 C | -3.458 C |
| **Control 1** | 12.556 C | 14.746 C | 10.551 C | **+2.190 C** | **-4.195 C** |

> As shown in the table, regions most affected by the CaCO3 dispersion (e.g., Experimental 1) appear to have a smaller temperature increase during the event compared to control regions.

## Thermal Camera Analysis

Thermal cameras were deployed to continuously monitor vine temperatures in both experimental and control locations throughout the dispersion event. The temporal analysis reveals a clear differential warming effect between the two sites.

<img src="plots/thermal_camera_temporal_analysis.png" alt="Thermal camera temporal analysis" width="750">

> During the experiment, the control vine temperature increased by 0.867 C, while the experimental vine under the dispersion plume increased by only 0.342 C. This 0.525 C difference provides direct evidence of the shading effect from the CaCO3 aerosol plume.

## Conclusion

Generally, a direct shading effect from the calcium carbonate (CaCO3) plume is observable and consistent with the measured wind directions. A region of strong plume concentration appears to increase in temperature less than the surrounding regions. Furthermore, the less cooling an area received, the further away it was from the plume's highest concentration.

Thermal cameras placed in control and experimental regions show slight cooling of vines. Vines are expected to show smaller temperature fluctuations due to homeostatic efforts by the plant.

In this experiment, clouds appeared after the dispersion interval, causing temperatures to normalize. We observed that regions *outside* the dispersion area cooled down more than the regions *inside* the dispersion area. This is consistent with our hypothesis: we are providing a direct shading effect, and once that shading effect dissipates, temperatures normalize.

Either our intervention produced a cooling effect, or the area of our intervention was already more resistant to temperature changes from solar radiation.

---

## Other Notes

* **Region Selection:** Regions were intentionally chosen away from the border of the main analysis area. Surrounding features, such as rivers or roads, could produce confounding effects on the temperature data.

* **Correlation Analysis:**
    * We explored the correlation between particulate matter (PM) measured by sensors, ambient air temperature, and drone-sensed ground temperature.
    * We found a **statistically significant negative correlation** between ambient air temperature and the PM sensed by the drone.
    * However, we did not observe a significant correlation between particulate matter and the drone-sensed *ground* temperature.

* **Potential Causes & Caveats:**
    * The dataset is limited, with only approximately 10 drone runs and 5 sensors within the drone region, n = 50 was not enough to determine significance.
    * Sensor location was not ideal:
        * One sensor is in a region that is consistently cooler than the rest of the field.
        * One sensor was positioned near the edge of the plume.
        * Other sensors were located far from the direct shading plume.

## Appendix

<img src="plots/drone_footage_analysis/after.png" alt="Temperature after experiment" width="550">

> After our experiment, the experimental region does appear to cool down less, as temperatures are normalizing in the absence of our shading effect.

<img src="plots/ambient_air_correlations/sps_pm2_5_r_squared_0.0014.png" alt="PM 2.5 correlation with ambient air temperature" width="750">

> A significant correlation between PM 2.5 and ambient air temperatures, where we see our dispersion event is causing high PM readings that correspond to incremental decreases in temperature.

<img src="plots/drone_correlations/sps_pm2_5_r_squared_0.0117.png" alt="PM 2.5 correlation with ground temperature" width="750">

> An insignificant correlation between PM 2.5 and ground temperatures, likely becuase of a few outliers, low n, and poor sensor placement

<img src="plots/tif_dispersion_effect_before.png" alt="Normalized Pixel Wise Differences" width="750">

> This plot gives a clear picture of the temperature differences and is normalized to account for local variation. As you can see, a direct shading effect from a cloud blown by north westerly winds is evident. 
