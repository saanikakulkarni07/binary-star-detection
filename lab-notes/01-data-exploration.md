# Data Exploration Notes

## Kiel Diagram (Teff vs logg)

Two panels, same axes (Teff on x-axis inverted, logg on y-axis inverted), colored by different properties.

### Left panel: colored by [Fe/H] (metallicity)
- Shows how metal-rich each star is. Blue = metal-poor ([Fe/H] ~ -2), red = metal-rich ([Fe/H] ~ +0.5).
- The red giant branch (upper right, low logg) shows a mix of metallicities. Metal-poor stars trace the halo/thick disk, metal-rich stars trace the thin disk.
- The giant branch can split by metallicity — metal-poor giants sit at slightly different Teff than metal-rich ones at the same logg.

### Right panel: colored by log10(VSCATTER)
- Shows how much each star's radial velocity varies between visits — our binary indicator.
- Purple/dark = low VSCATTER (stable RV, likely single). Yellow/bright = high VSCATTER (variable RV, likely binary).
- **Result:** Mostly purple with yellow dots scattered throughout. This is the expected pattern:
  - Binaries are a minority (~10-15%), consistent with VSCATTER thresholds.
  - They're spread across the full Kiel diagram, not clustered in one spectral type. This confirms the labels aren't driven by systematics (e.g., giant pulsations).
  - Main-sequence dwarfs (high logg ~ 4-5) with high VSCATTER are strong binary candidates.
  - Giants with high VSCATTER could be binaries or intrinsic RV jitter from pulsations — something to watch for in validation.

## VSCATTER Distribution (log scale)

Histogram of log10(VSCATTER) for stars with NVISITS >= 3.

- **Main peak at log10(VSCATTER) ~ -1 to -2** → most stars have VSCATTER ~ 0.01–0.1 km/s. Very stable RVs, clearly single stars.
- **Right tail extends to ~2** → a few stars with VSCATTER up to ~100 km/s. Large RV swings, strong binary candidates.
- **Threshold lines:**
  - Green (single threshold, 0.3 km/s) at log10 = -0.52, sits just right of the main peak.
  - Red (binary threshold, 1.0 km/s) at log10 = 0.0, further into the tail.
- **Class balance:** ~70-80% single (below 0.3 km/s), ~10-15% ambiguous (0.3–1.0 km/s, discarded), ~10-15% binary (above 1.0 km/s).
- The clean separation between the main peak and the binary tail validates the threshold choice. The intermediate zone we discard is small, minimizing label noise.

## VSCATTER Distribution — Filtered Sample (linear scale)

Histogram of raw VSCATTER (km/s) for the quality-filtered sample (SNR >= 50, NVISITS >= 3, clean flags, Teff/logg cuts).

- **Extremely right-skewed** in linear space. Almost all stars are between 0 and ~50 km/s, with a huge spike near 0.
- This is expected — most stars are single with very small RV scatter, and a small fraction of binaries produce the long right tail.
- Linear scale compresses the interesting structure; log scale (as in notebook 01) is much more informative for seeing the binary/single separation.
- The skewness itself is a useful signal: it confirms that high-VSCATTER stars are rare outliers from the bulk population, consistent with a distinct binary subpopulation rather than a smooth continuum of RV variability.

## Label Assignment Results

Applied VSCATTER thresholds to the filtered sample (~343K stars after quality cuts):

- **Binary (VSCATTER > 1.0 km/s):** 23,318 stars (6.8% of filtered, 7.4% of labeled)
- **Single (VSCATTER < 0.3 km/s):** 291,871 stars (85.0% of filtered, 92.6% of labeled)
- **Ambiguous (0.3–1.0 km/s, discarded):** 28,153 stars (8.2% of filtered)
- **Total labeled sample:** 315,189 stars
- **Class ratio:** 12.5:1 (single:binary)

### Notes:
- Binary fraction (7.4%) is lower than the initial ~10-15% estimate. This is because the quality cuts (especially SNR >= 50) bias toward well-observed stars, and the 1.0 km/s threshold is conservative.
- 12.5:1 imbalance is manageable with class_weight="balanced" (RF) and pos_weight (CNN BCEWithLogitsLoss). Without these, a naive model would get 92.6% accuracy by predicting all single.
- The ambiguous zone (28K stars, 8.2%) is small — we're not discarding too much data, and the gap between thresholds reduces label noise at the decision boundary.
- 315K labeled stars is more than enough for both RF and CNN training.

## Label Systematics Check

Four scatter plots of VSCATTER vs stellar parameters, colored by label (blue = single, red = binary). Checks whether our VSCATTER-based labels are driven by systematics rather than true binarity.

### Results — no systematic biases detected:

- **VSCATTER vs Teff:** Binaries spread across the full 3500–7000 K range. No concentration at cool temperatures, ruling out pulsating giant contamination.
- **VSCATTER vs logg:** Binaries appear at all surface gravities — both dwarfs (logg ~4-5) and giants (logg ~1-3). No systematic clustering.
- **VSCATTER vs Combined S/N:** Binaries are not concentrated at low SNR. They appear across the full SNR range (up to ~4000). This confirms the labels are not noise-driven. The white band (0.3–1.0 km/s exclusion zone) is clearly visible between the two populations.
- **VSCATTER vs [Fe/H]:** No trend with metallicity. Binaries span the full range from [Fe/H] ~ -2.5 to +0.5.

### Interpretation:
The clean separation between blue and red populations across all four parameters validates the labeling strategy. High VSCATTER is not a proxy for low SNR, cool temperature, or any other systematic — it genuinely reflects RV variability, consistent with binary orbital motion.

## SB9 Catalog Cross-Match

Cross-matched our 315K labeled sample against the SB9 (9th Catalogue of Spectroscopic Binary Orbits, 4,079 entries) at 2 arcsec radius. SB9 coordinates required conversion from sexagesimal (HH MM SS) to decimal degrees.

### Results:
- **Total matches:** 109 of our stars appear in SB9
- **Our binary-labeled in SB9:** 77 (70.6% of matches)
- **Our single-labeled in SB9:** 32 (29.4% of matches — "missed" binaries)
- **Median separation:** 0.163 arcsec (excellent positional agreement)

### Interpretation:
- 70.6% precision against a gold-standard spectroscopic binary catalog validates the VSCATTER threshold approach.
- The 32 missed binaries are likely long-period, low-amplitude, or face-on systems where RV variation stays below our 0.3 km/s single threshold. These are the hard cases where single-epoch ML detection could add the most value.
- Only 109 matches out of 4,079 SB9 entries reflects the limited sky overlap between APOGEE's H-band survey and the optically-selected SB9 catalog, not a failure of the method.

## Gaia DR3 NSS Cross-Match

Cross-matched against Gaia DR3 non-single-star two-body orbit solutions (443,205 entries after removing NaN coordinates) at 2 arcsec radius.

### Results:
- **Total matches:** 1,300 of our stars appear in Gaia NSS
- **Our binary-labeled in Gaia NSS:** 650 (50.0%)
- **Our single-labeled in Gaia NSS:** 650 (50.0%)

### Interpretation:
- Lower recovery (50%) than SB9 (71%) is expected. Gaia NSS includes astrometric binaries, eclipsing binaries, and long-period systems detected through positional wobble or photometric dips — not RV variability. A face-on astrometric binary can have nearly zero RV variation.
- The 650 Gaia binaries in our "single" class are not label errors — they're systems invisible to RV-based detection but detectable by Gaia's astrometric precision.
- These "missed" binaries are a key opportunity for the CNN: if spectral signatures (composite spectra, broadened lines) are present even without RV variability, single-epoch ML could detect systems that VSCATTER fundamentally cannot.

## Combined Validation Summary

| Catalog      | Matches | Our Binary | Our Single | Precision |
|-------------|---------|-----------|-----------|-----------|
| SB9          | 109     | 77        | 32        | 70.6%     |
| Gaia DR3 NSS | 1,300   | 650       | 650       | 50.0%     |

The VSCATTER labeling strategy is well-validated for RV-variable binaries (SB9). The Gaia results highlight a complementary population of binaries detectable through other methods — potential targets for single-epoch spectral ML detection.

## Spectral Preprocessing Notes

### Why pick example spectra first
We select a few example stars (3 binary, 3 single) to visually verify preprocessing before committing to the full 315K-star batch run (~6.4 GB). This serves as a sanity check: does bad pixel masking work? Does continuum normalization preserve absorption line shapes? Are binary vs single spectral differences actually visible?

### Raw vs cleaned spectrum expectations
- For high-SNR APOGEE stars (SNR >= 50), the difference between raw and cleaned spectra is usually subtle.
- Bad pixels appear as sharp spikes or dips to zero/NaN — masking replaces these with interpolated values, smoothing out isolated glitches.
- Detector edges between APOGEE's three chips (blue/green/red) can have artifacts that get cleaned up.
- If raw and cleaned look nearly identical, that's a good sign — it means the APOGEE pipeline already produced high-quality data and our preprocessing isn't introducing artifacts.

### Continuum normalization
- This is where the bigger visual change happens. It flattens the overall spectral shape to ~1.0, so absorption lines are the only remaining features.
- This is what the CNN actually needs — relative line depths and shapes, not the absolute flux level.
- Two methods available: polynomial fit (iterative sigma-clipping to reject lines) and median filter. Polynomial is default.

### Continuum normalization results (2M15044648+2224548)
- **Top panel:** Raw flux ranges from ~32,000 to ~18,000 ADU across the H-band (15,150–17,000 A). Both polynomial (gray dashed) and median-filter (red) continua track the broad spectral slope well.
- **Middle panel (polynomial):** Clean normalization to ~1.0. Flat continuum with sharp absorption lines dipping below. No distortion around deep features. This is the CNN input.
- **Bottom panel (median filter):** Also reasonable, but slightly noisier around deep/broad absorption features — the median filter can struggle where strong lines occupy a significant fraction of the filter window.
- **Decision:** Polynomial normalization is cleaner and more stable. Confirmed as the default method in PREPROCESS_CONFIG.
- **Key absorption features visible:** Deep lines at ~15,750 A (Mg I), ~16,000 A (CO bandhead), and ~16,750 A (Al I) — these are the lines defined in FEATURE_CONFIG for the hand-crafted RF features.

### CCF comparison: single vs binary
- **Stars compared:** 2M21342357+1215247 (single, SNR=4469) vs 2M15044648+2224548 (binary, SNR=4096, VSCATTER=12.94 km/s)
- **Visual appearance:** The CCF profiles look almost identical by eye — both show a single symmetric-looking peak. This is expected for an SB1 binary where one star dominates the light.
- **Quantitative differences tell a different story:**

| Metric | Single | Binary | Difference |
|--------|--------|--------|------------|
| FWHM | 28.9 km/s | 36.3 km/s | +25% broader |
| Bisector span | -0.019 | -0.154 | 8x more asymmetric |
| Asymmetry | 0.010 | -0.008 | Similar |

- **Key insight:** The binary's CCF is 25% broader (line broadening from companion's blended light) and 8x more asymmetric (companion pulls line profile to one side). These differences are quantitatively real but visually subtle.
- **Implication for ML:** This is exactly why we need a CNN — it can detect small, consistent differences across hundreds of absorption lines simultaneously, even though any single line or CCF looks "almost the same" to the eye. A human can't integrate these subtle signals across 8,575 pixels, but a neural network can.

### Absorption line properties: single vs binary

Measured equivalent width, line depth, FWHM, and asymmetry index across all H-band line regions (Brackett series, CO bandhead, OH, Mg I, Fe I, Al I) for the same single/binary pair.

**Key patterns:**
- **Line depths are systematically shallower for the binary** across nearly every region (e.g., Mg I: 0.213 vs 0.319, Al I: 0.090 vs 0.203, Fe I_0: 0.073 vs 0.166). The companion's continuum light fills in the absorption lines.
- **Equivalent widths are smaller for the binary** — same dilution effect (e.g., Mg I: 1.06 vs 1.72).
- **FWHM is similar or slightly broader for the binary** — subtle line broadening from the blended companion, consistent with the CCF FWHM finding.

**Physical interpretation:** This is classic SB1 behavior. The secondary star adds continuum light (or differently-featured light), which dilutes the primary's absorption lines. The effect is systematic — it appears across all line regions, not just one — making it a strong multi-feature signal for ML.

**Implication for RF baseline:** Features like line depth ratios, EW ratios, and the pattern of dilution across multiple lines should give the Random Forest discriminative power even without the full spectral context the CNN sees.

## Random Forest Baseline (catalog features only)

Trained RF (500 trees, max_depth=20, class_weight=balanced) on 5 catalog features: Teff, logg, [Fe/H], SNR, NVISITS. This is a minimal baseline — no spectral information.

### Split:
- Train: 220,631 (16,322 binary) / Val: 47,279 (3,498 binary) / Test: 47,279 (3,498 binary)

### Results:
- **5-fold CV AUROC: 0.699 +/- 0.005**
- **Test AUROC: 0.703** — only modestly above random (0.50)
- **Binary recall: 21%** — catches just 1 in 5 binaries
- **Binary precision: 59%** — when it flags a binary, correct about half the time
- **Accuracy: 93%** — misleading, driven by 92.6% single-class prevalence

### ROC and PR curves:
- **ROC:** Curve bows slightly above the diagonal but stays close — weak discriminative power.
- **PR:** Precision is high (~90-95%) at very low recall (<10%), but drops off a cliff around 20-30% recall. No good operating point balances both precision and recall.

### Interpretation:
This proves that stellar parameters alone cannot reliably identify binaries. Teff, logg, and [Fe/H] describe a star's atmosphere, not whether it has a companion. The CNN on actual spectra should beat this significantly by detecting line broadening, dilution, and asymmetry patterns. This baseline sets the floor for comparison.

### Feature importance (permutation importance):
1. **logg** — highest. Binary systems can shift apparent surface gravity, and certain evolutionary stages (e.g., subgiants) have higher binary fractions.
2. **Teff** — some temperature ranges are more binary-rich.
3. **[Fe/H]** — mild correlation, possibly because metal-poor halo stars have different binary properties than disk stars.
4. **NVISITS** — low importance. Observational metadata, not a physical property.
5. **SNR** — lowest. Good — the model isn't cheating on survey selection effects.

None of these features capture individual binary signatures. They reflect population-level correlations (e.g., "subgiants are more likely to be binaries") rather than spectral evidence of a companion. This is fundamentally different from what the CNN will learn from actual spectra.

## CNN Results (10K spectra, first run)

Trained 1D CNN (1.1M parameters) on ~10K continuum-normalized APOGEE spectra (~740 binary). Early stopping at epoch 25, best val AUROC at epoch 15.

### Training:
- Best val AUROC: **0.7362** — beats RF baseline (0.703) using spectral information
- Val loss unstable (spikes to 9.89 at epoch 23) — outlier spectra cause instability
- Train loss decreasing steadily (1.27 → 0.79), some overfitting remains

### Test set evaluation:
- **Test AUROC: 0.7118**
- **Binary recall: 77%** — catches 3 out of 4 binaries (vs RF's 21%)
- **Binary precision: 10%** — many false positives at default 0.5 threshold
- **Accuracy: 50%** — threshold is too aggressive, flagging too many singles as binaries

### Comparison with RF baseline:
| Metric | RF (catalog features) | CNN (10K spectra) |
|--------|----------------------|-------------------|
| AUROC | 0.703 | 0.712 |
| Binary recall | 21% | 77% |
| Binary precision | 59% | 10% |

### Interpretation:
- The CNN finds far more binaries (77% recall vs 21%) but at the cost of many false positives (10% precision). The RF is conservative (high precision, low recall); the CNN is aggressive.
- The low precision reflects both the small training set (10K) and the class imbalance (only ~7% binary). With more data, the CNN should learn to be more selective.
- AUROC is comparable (0.71 vs 0.70), suggesting the CNN is learning real signal but needs more data to separate from the RF. The improvement from 550 → 10K spectra (0.69 → 0.74 val AUROC) predicts further gains at 50K+.
- **Next step:** Scale to 50K+ spectra to reduce overfitting and improve precision while maintaining high recall.
