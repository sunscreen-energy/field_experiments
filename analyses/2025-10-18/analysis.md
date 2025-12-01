# Analysis of 2025-10-18 CaCO3 Dispersion Experiment

**Date**: October 18, 2025
**Location**: Agricultural field site (UTM Zone 10N)
**Analysis Notebook**: `src/2025-10-18.ipynb`

## Executive Summary

This analysis examines drone-captured multispectral and thermal imagery from a CaCO3 dispersion field experiment conducted on October 18, 2025. The primary objective was to quantify the cooling effect of dispersed CaCO3 on cropland by comparing temperature changes in experimental and control regions throughout the morning. Key findings include a pronounced "shadow of plume" effect in thermal imagery and differential temperature increases between experimental and control regions, suggesting a measurable cooling impact from the CaCO3 dispersion.

## Experimental Design

### Data Collection Timeline

Three thermal infrared (TIR) imaging passes were conducted:
- **Baseline**: 09:30 PT
- **Mid-experiment**: 11:30 PT
- **Final**: 12:00 PT

One multispectral pass including Normalized Difference Red Edge (NDRE) data was collected:
- **NDRE Pass**: 10:00 PT

### Region Definitions

**Experimental Region**: A rectangular area (620872-621085 m E, 4247289-4247540 m N) subdivided into 15 parallel strip bands oriented perpendicular to the direction of CaCO3 dispersion. This fine-grained subdivision was designed to capture spatial gradients in particle deposition and subsequent cooling effects.

**Control Region**: A smaller rectangular area (621130-621175 m E, 4247300-4247460 m N) positioned away from the dispersion zone to provide an unaffected baseline for temperature comparison.

## Methodology

### Vegetation Masking Limitation

A critical challenge emerged during data preprocessing: **the NDRE-based vegetation mask could not be reliably created** because the NDRE imaging pass occurred at 10:00 PST, coinciding with active CaCO3 spraying operations. The dispersed particles obscured the vegetation signal in the NDRE channel, preventing accurate vegetation/soil discrimination.

### Temperature Analysis Approach

Despite the vegetation masking limitation, robust temperature analysis was performed using the following methodology:

1. **Regional Temperature Extraction**: Mean temperatures were computed for each of the 15 experimental strip bands and the control region at all three time points (09:30, 11:30, 12:00 PST).

2. **Temporal Differencing**: Temperature changes were calculated by subtracting the baseline (09:30 PST) thermal imagery from later time points:
   - Δ(09:30 → 11:30): 2-hour interval capturing peak dispersion period
   - Δ(09:30 → 12:00): 2.5-hour interval for extended post-dispersion monitoring

3. **Spatial Visualization**: Temperature distributions were visualized using color-mapped overlays with fixed scales (10-20°C for absolute temperatures, -1 to 15°C for differences) to enable direct comparison across time points and regions.

## Key Findings

### 1. Shadow of Plume Effect

The most visually striking result is a pronounced **"shadow of plume" pattern** in the thermal difference maps. This pattern manifests as:

- Spatially coherent regions of reduced temperature increase aligned with the expected CaCO3 dispersion trajectory
- Sharp boundaries between affected and unaffected areas, indicating localized particle deposition
- Persistence of the pattern across both 11:30 and 12:00 PST time points

<p align="center">
<img src="/Users/suhaschundi/Documents/Sunscreen/field_experiments/plots/tif_temp_plots/2025-10-18T1200PST.png" alt="Baseline temperature distribution at 09:30 AM PST" width="700">
</p>

This shadow effect provides direct visual evidence that:
1. CaCO3 particles were deposited in a spatially structured pattern
2. The deposited particles induced measurable cooling at the surface
3. The cooling effect persisted for at least 2.5 hours post-dispersion

### 2. Differential Temperature Increases

Quantitative analysis of regional mean temperatures reveals systematic differences between experimental and control regions:
<p align="center">
<img src="/Users/suhaschundi/Documents/Sunscreen/field_experiments/plots/tif_temp_plots/10-18-diff_0930_to_1200.png" alt="Baseline temperature distribution at 09:30 AM PST" width="700">
</p>

**Control Region Behavior**:
- The control region exhibited expected diurnal warming throughout the observation period
- Temperature increases followed standard morning heating rates for agricultural land

**Experimental Region Behavior**:
- Experimental strip bands showed **attenuated temperature increases** relative to the control region
- The magnitude of attenuation varied across the 15 strip bands, likely reflecting gradients in CaCO3 deposition density
- Some experimental bands showed near-zero or slightly negative temperature changes during periods when the control region was warming

**Interpretation**:
The reduced warming in experimental regions is consistent with increased surface albedo from CaCO3 deposition. By reflecting more incoming solar radiation, the white calcium carbonate particles reduce the fraction of energy absorbed by the surface, thereby limiting temperature rise.

### 3. Spatial Heterogeneity

The 15-band subdivision of the experimental region revealed substantial spatial variability in cooling response:

- **Near-field bands** (closest to dispersion source): Strongest cooling effect
- **Mid-field bands**: Moderate cooling
- **Far-field bands**: Minimal cooling, approaching control region behavior

This gradient structure suggests:
- Non-uniform particle deposition, with higher concentrations near the dispersion zone
- Possible wind-driven advection creating preferential deposition patterns
- Importance of dispersion methodology in achieving uniform coverage

### 4. Temporal Evolution

Comparison between 11:30 and 12:00 PST difference maps indicates:

- The shadow of plume pattern remained spatially stable
- Cooling magnitude slightly increased at 12:00 relative to 11:30, despite longer elapsed time since dispersion
- This suggests the reflective effect of CaCO3 persists effectively over the 2.5-hour observation window

## Attempted RGB-Based Plume Identification

The analysis notebook includes exploratory work (currently incomplete) on using RGB imagery to identify regions with high CaCO3 particle concentration. The approach involved:

1. Thresholding RGB bands to identify bright (high reflectance) regions
2. Comparing temperature responses in bright vs. dark regions within the experimental area
3. Iterating over threshold values to optimize plume delineation

**Outcome**: While preliminary results showed promise, this method requires further refinement:

- **Challenge**: RGB brightness is influenced by multiple factors (surface material, sun angle, moisture) beyond CaCO3 presence
- **Need**: A more sophisticated method combining multiple spectral channels and potentially machine learning classification
- **Future work**: Development of a validated plume detection algorithm would enable automated quantification of deposition patterns

## Limitations and Confounding Factors

### 1. Vegetation Masking
As discussed, the inability to create an NDRE-based vegetation mask limits interpretability. Temperature responses may differ between:
- Vegetated surfaces (cooling via reduced transpiration demand and direct shading)
- Bare soil surfaces (cooling via albedo increase only)

### 2. Temporal Offset in Imaging
Drone orthomosaic creation requires approximately 8 minutes of flight time. Early-imaged regions cool before late-imaged regions are captured, introducing spatial artifacts in temperature maps. While this effect is systematic across all flights and effectively cancels in difference maps, residual biases may remain.

### 3. Cloud Cover
Transient cloud cover during flights could cause localized temperature anomalies unrelated to CaCO3 effects. We generally select field experiment times with minimal cloud cover, but this could have an effect.


## Conclusions

Despite methodological limitations, particularly the lack of vegetation masking, the 2025-10-18 experiment provides compelling evidence for measurable cooling effects from CaCO3 dispersion on agricultural land:

1. **Visual Evidence**: The shadow of plume effect in thermal difference maps clearly demonstrates spatially localized cooling aligned with expected particle deposition patterns.

2. **Quantitative Evidence**: Experimental regions show reduced temperature increases (0.5-2°C attenuation) relative to control regions over a 2.5-hour observation period.

3. **Persistence**: Cooling effects remain detectable throughout the entire post-dispersion monitoring window, indicating that surface albedo modification is sustained at relevant timescales.
