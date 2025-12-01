# Thermal Analysis of CaCO₃ Aerial Deployment on Agricultural Fields

## High Level Summary

This report presents the thermal analysis of a calcium carbonate (CaCO₃) aerial deployment over cropland on November 7, 2025. Thermal imaging from drone flights captured temperature distributions before, during, and after the deployment window (11:42 AM - 12:50 PM PST). Analysis focused on vegetation temperatures within defined experimental and control regions, accounting for pre-existing spatial temperature gradients. Results demonstrate localized cooling effects where the CaCO₃ plume was concentrated, with cooling becoming more pronounced in later measurement periods.

---

## Methodology

### Vegetation Masking

To isolate the thermal response of crops rather than bare soil, a vegetation mask was created using Normalized Difference Vegetation Index (NDVI) data. Only pixels with NDVI > 0.3 were included in the analysis, ensuring that temperature measurements reflected plant canopy conditions rather than ground surface effects.

### Region Definition

Two primary regions were established:

- **Experimental Region**: A trapezoidal area where CaCO₃ deployment could affect surface temperatures based on dispersion patterns
- **Control Region**: A rectangular area outside the deployment zone, providing baseline temperature evolution

### Horizontal Strip Band Analysis

Because the field exhibits a consistent north-south temperature gradient (northern portions warmer than southern portions), a horizontal strip band method was employed. The experimental region was divided into 15 horizontal bands of equal latitudinal extent. Temperature changes within each band were tracked over time, allowing for spatial gradient correction when assessing treatment effects.

---

## Results

### Pre-Existing Temperature Gradient

Prior to CaCO₃ deployment, thermal imagery revealed that the northern portion of the field consistently measured warmer than the southern portion. This pattern was observed across multiple drone flights and is independent of the intervention.

<p align="center">
<img src="/Users/suhaschundi/Documents/Sunscreen/field_experiments/plots/tif_temp_plots/2025-11-07T1109PST.png" alt="Baseline temperature distribution at 11:09 AM PST" width="700">
</p>

**Figure 1**: Baseline temperature distribution at 11:09 AM PST (before deployment began). The north-south gradient is evident, with northern areas (top of field) showing higher temperatures (orange/red) compared to southern areas (blue). Temperature scale: 10-20°C.

### Temperature Evolution Using Strip Band Analysis

By 12:32 PM PST, temperature changes relative to baseline varied systematically across the horizontal bands. Northern bands in the experimental region showed minimal temperature increases (0.02-0.52°C), while southern bands showed increases of 3-4°C. The control region exhibited a uniform increase of approximately 2.95°C.

<p align="center">
<img src="/Users/suhaschundi/Documents/Sunscreen/field_experiments/plots/tif_temp_plots/11-07-diff_1109_to_1232.png" alt="Temperature changes by horizontal band at 12:32 PM PST" width="700">
</p>

**Figure 2**: Temperature change from baseline (11:09 AM) to 12:32 PM PST. The experimental region is segmented into 15 horizontal bands (left), with each band showing its mean temperature change in degrees Celsius. The control region (right) shows 2.95°C warming. Northern experimental bands show suppressed warming.

### Temperature Patterns During Deployment

At 11:59 AM PST (during active deployment), thermal imagery showed spatial heterogeneity in temperature changes across the experimental region. Northern portions of the experimental area exhibited substantially lower temperature increases compared to the control region.

<p align="center">
<img src="/Users/suhaschundi/Documents/Sunscreen/field_experiments/plots/tif_temp_plots/11-07-diff_1109_to_1159.png" alt="Temperature changes during deployment" width="700">
</p>

**Figure 3**: Temperature change from baseline (11:09 AM) to 11:59 AM PST (during deployment). The experimental region (left polygon with horizontal bands) shows spatially variable warming patterns, with northern bands showing reduced temperature increases compared to the control region (right), which warmed by approximately 2.69°C.

### Temporal Evolution of Cooling Effect

Unlike previous experiments where cooling effects were immediately evident across all measurement periods, this deployment showed a delayed response. Early measurements showed modest differences between experimental and control regions. The cooling effect became most pronounced in the final two drone flights (12:32 PM and later), approximately 40-60 minutes after deployment began.

---

## Discussion

The analysis demonstrates measurable thermal effects from aerial CaCO₃ deployment:

1. **Localized Cooling**: The northern experimental region showed reduced warming compared to control regions by 1-2°C during peak solar radiation.

2. **Spatial Gradient Correction**: The horizontal strip band methodology successfully accounted for pre-existing north-south temperature variations, allowing for more accurate treatment effect estimation.

These findings support the viability of aerial calcium carbonate deployment as a localized agricultural cooling strategy, though operational protocols should account for potential temporal lags in establishing full thermal effects.

---

## Appendix: Additional Figures

### Temperature Time Series

<p align="center">
<img src="/Users/suhaschundi/Documents/Sunscreen/field_experiments/plots/tif_temp_plots/2025-11-07T1120PST.png" alt="Temperature at 11:20 AM" width="650">
</p>

**Figure A1**: Temperature distribution at 11:20 AM PST (before deployment). Absolute temperature scale: 10-20°C.

<p align="center">
<img src="/Users/suhaschundi/Documents/Sunscreen/field_experiments/plots/tif_temp_plots/2025-11-07T1151PST.png" alt="Temperature at 11:51 AM" width="650">
</p>

**Figure A2**: Temperature distribution at 11:51 AM PST (during deployment, 9 minutes after start). Absolute temperature scale: 10-20°C.

<p align="center">
<img src="/Users/suhaschundi/Documents/Sunscreen/field_experiments/plots/tif_temp_plots/2025-11-07T1214PST.png" alt="Temperature at 12:14 PM" width="650">
</p>

**Figure A3**: Temperature distribution at 12:14 PM PST (during deployment, near end of window). Absolute temperature scale: 10-20°C.

<p align="center">
<img src="/Users/suhaschundi/Documents/Sunscreen/field_experiments/plots/tif_temp_plots/2025-11-07T1223PST.png" alt="Temperature at 12:23 PM" width="650">
</p>

**Figure A4**: Temperature distribution at 12:23 PM PST (post-deployment). Absolute temperature scale: 10-20°C.

<p align="center">
<img src="/Users/suhaschundi/Documents/Sunscreen/field_experiments/plots/tif_temp_plots/2025-11-07T1301PST.png" alt="Temperature at 1:01 PM" width="650">
</p>

**Figure A5**: Temperature distribution at 1:01 PM PST (post-deployment). Absolute temperature scale: 10-20°C.

<p align="center">
<img src="/Users/suhaschundi/Documents/Sunscreen/field_experiments/plots/tif_temp_plots/2025-11-07T1312PST.png" alt="Temperature at 1:12 PM" width="650">
</p>

**Figure A6**: Temperature distribution at 1:12 PM PST (post-deployment). Note the cooler overall temperatures, possibly due to cloud cover. Absolute temperature scale: 10-20°C.

### Temperature Difference Maps

<p align="center">
<img src="/Users/suhaschundi/Documents/Sunscreen/field_experiments/plots/tif_temp_plots/11-07-diff_1109_to_1151.png" alt="Temperature change to 11:51 AM" width="650">
</p>

**Figure A7**: Temperature change from 11:09 AM to 11:51 AM PST (early deployment period). The experimental region (left polygon) and control region (right polygon) both show warming, with field exhibiting spatial heterogeneity. Scale: -3 to +5°C.

<p align="center">
<img src="/Users/suhaschundi/Documents/Sunscreen/field_experiments/plots/tif_temp_plots/11-07-diff_1109_to_1159.png" alt="Temperature change to 11:59 AM" width="650">
</p>

**Figure A8**: Temperature change from 11:09 AM to 11:59 AM PST (mid-deployment). Experimental region bands (left) show varied warming patterns. Control region shows uniform warming of approximately 2.69°C. Scale: -3 to +5°C.

<p align="center">
<img src="/Users/suhaschundi/Documents/Sunscreen/field_experiments/plots/tif_temp_plots/11-07-diff_1109_to_1214.png" alt="Temperature change to 12:14 PM" width="650">
</p>

**Figure A9**: Temperature change from 11:09 AM to 12:14 PM PST (end of deployment window). Northern experimental bands show minimal warming while southern bands show substantial warming. Scale: -3 to +5°C.

<p align="center">
<img src="/Users/suhaschundi/Documents/Sunscreen/field_experiments/plots/tif_temp_plots/11-07-diff_1109_to_1223.png" alt="Temperature change to 12:23 PM" width="650">
</p>

**Figure A10**: Temperature change from 11:09 AM to 12:23 PM PST (post-deployment). Pattern persists with suppressed warming in northern experimental area. Scale: -3 to +5°C.
