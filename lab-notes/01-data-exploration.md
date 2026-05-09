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
