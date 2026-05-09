"""Hand-crafted spectral features for binary detection."""

import numpy as np
from scipy.signal import correlate
from scipy.optimize import curve_fit
from config import FEATURE_CONFIG


def _gaussian(x, amp, mu, sigma, offset):
    """1D Gaussian with constant offset."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def compute_ccf(flux, wavelength, template_flux, rv_range=(-300, 300), rv_step=1.0):
    """Compute the cross-correlation function between an observed spectrum and a template.

    Parameters
    ----------
    flux : np.ndarray (n_pixels,)
        Observed, continuum-normalized spectrum.
    template_flux : np.ndarray (n_pixels,)
        Template spectrum on the same wavelength grid.
    rv_range : tuple
        Min and max RV in km/s.
    rv_step : float
        RV step in km/s.

    Returns
    -------
    dict with keys:
        - 'rv_grid': np.ndarray — RV values in km/s
        - 'ccf': np.ndarray — CCF values
        - 'rv_peak': float — RV at CCF peak (km/s)
        - 'fwhm': float — FWHM of CCF peak (km/s)
        - 'bisector_span': float — bisector velocity span (km/s)
        - 'asymmetry': float — asymmetry index
    """
    c_kms = 2.998e5  # speed of light in km/s
    log_wave = np.log(wavelength)
    dlog = np.median(np.diff(log_wave))

    rv_grid = np.arange(rv_range[0], rv_range[1], rv_step)
    shift_pixels = (rv_grid / c_kms) / dlog  # RV shift in pixel units

    # Subtract mean for proper cross-correlation
    f_obs = flux - np.nanmean(flux)
    f_tmpl = template_flux - np.nanmean(template_flux)

    # Compute CCF via spectral shifting
    ccf = np.zeros(len(rv_grid))
    for i, sp in enumerate(shift_pixels):
        shifted = np.interp(
            np.arange(len(f_tmpl)) - sp,
            np.arange(len(f_tmpl)),
            f_tmpl,
        )
        ccf[i] = np.nansum(f_obs * shifted)

    # Normalize
    ccf /= np.max(np.abs(ccf)) if np.max(np.abs(ccf)) > 0 else 1.0

    # Fit Gaussian to find peak
    peak_idx = np.argmax(ccf)
    rv_peak = rv_grid[peak_idx]

    # Fit around peak (+/- 50 km/s)
    fit_mask = np.abs(rv_grid - rv_peak) < 50
    try:
        popt, _ = curve_fit(
            _gaussian, rv_grid[fit_mask], ccf[fit_mask],
            p0=[ccf[peak_idx], rv_peak, 10.0, 0.0],
            maxfev=5000,
        )
        fwhm = 2.355 * abs(popt[2])  # sigma to FWHM
        rv_peak = popt[1]
    except RuntimeError:
        fwhm = np.nan

    # Bisector span (difference in RV between upper and lower halves of CCF peak)
    bisector_span = _compute_bisector_span(rv_grid, ccf, rv_peak)

    # Asymmetry: ratio of area on blue vs red side of peak
    blue_mask = (rv_grid < rv_peak) & (rv_grid > rv_peak - 100)
    red_mask = (rv_grid > rv_peak) & (rv_grid < rv_peak + 100)
    blue_area = np.trapezoid(ccf[blue_mask], rv_grid[blue_mask]) if blue_mask.any() else 0
    red_area = np.trapezoid(ccf[red_mask], rv_grid[red_mask]) if red_mask.any() else 0
    asymmetry = (blue_area - red_area) / (blue_area + red_area) if (blue_area + red_area) > 0 else 0.0

    return {
        "rv_grid": rv_grid,
        "ccf": ccf,
        "rv_peak": rv_peak,
        "fwhm": fwhm,
        "bisector_span": bisector_span,
        "asymmetry": asymmetry,
    }


def _compute_bisector_span(rv_grid, ccf, rv_peak):
    """Compute the bisector velocity span of the CCF peak."""
    peak_val = ccf.max()
    min_val = np.median(ccf)
    levels = np.linspace(min_val + 0.2 * (peak_val - min_val),
                         min_val + 0.8 * (peak_val - min_val), 10)

    bisector_rvs = []
    for level in levels:
        # Find blue and red crossings
        above = ccf >= level
        crossings = np.where(np.diff(above.astype(int)))[0]
        if len(crossings) >= 2:
            # Interpolate crossing RVs
            rv_blue = np.interp(level, ccf[crossings[0]:crossings[0]+2],
                                rv_grid[crossings[0]:crossings[0]+2])
            rv_red = np.interp(level, ccf[crossings[-1]:crossings[-1]+2][::-1],
                               rv_grid[crossings[-1]:crossings[-1]+2][::-1])
            bisector_rvs.append((rv_blue + rv_red) / 2)

    if len(bisector_rvs) >= 4:
        # Span = difference between top and bottom bisector points
        return bisector_rvs[-1] - bisector_rvs[0]
    return np.nan


def measure_line_properties(flux, wavelength, region):
    """Measure properties of an absorption line in a given wavelength region.

    Parameters
    ----------
    flux : np.ndarray — continuum-normalized spectrum
    wavelength : np.ndarray
    region : tuple (wl_min, wl_max) in Angstroms

    Returns
    -------
    dict with: equivalent_width, line_depth, fwhm, asymmetry_index
    """
    mask = (wavelength >= region[0]) & (wavelength <= region[1])
    if mask.sum() < 5:
        return {"equivalent_width": np.nan, "line_depth": np.nan,
                "fwhm": np.nan, "asymmetry_index": np.nan}

    wl = wavelength[mask]
    fl = flux[mask]

    # Equivalent width (integral of 1 - flux)
    ew = np.trapezoid(1.0 - fl, wl)

    # Line depth
    line_depth = 1.0 - np.min(fl)

    # FWHM of the absorption line
    half_depth = 1.0 - line_depth / 2
    above_half = fl <= half_depth
    if above_half.any():
        indices = np.where(above_half)[0]
        fwhm = wl[indices[-1]] - wl[indices[0]]
    else:
        fwhm = np.nan

    # Asymmetry: compare blue and red halves
    min_idx = np.argmin(fl)
    blue_ew = np.trapezoid(1.0 - fl[:min_idx+1], wl[:min_idx+1]) if min_idx > 0 else 0
    red_ew = np.trapezoid(1.0 - fl[min_idx:], wl[min_idx:]) if min_idx < len(fl)-1 else 0
    total_ew = blue_ew + red_ew
    asymmetry_index = (blue_ew - red_ew) / total_ew if total_ew > 0 else 0.0

    return {
        "equivalent_width": ew,
        "line_depth": line_depth,
        "fwhm": fwhm,
        "asymmetry_index": asymmetry_index,
    }


def extract_handcrafted_features(flux, wavelength, template_flux=None):
    """Extract all hand-crafted features from a single spectrum.

    Parameters
    ----------
    flux : np.ndarray — continuum-normalized spectrum
    wavelength : np.ndarray
    template_flux : np.ndarray, optional — template for CCF computation

    Returns
    -------
    dict
        Feature name -> value mapping.
    """
    features = {}

    # CCF-based features (if template provided)
    if template_flux is not None:
        ccf_result = compute_ccf(flux, wavelength, template_flux)
        features["ccf_fwhm"] = ccf_result["fwhm"]
        features["ccf_bisector_span"] = ccf_result["bisector_span"]
        features["ccf_asymmetry"] = ccf_result["asymmetry"]
        features["ccf_rv"] = ccf_result["rv_peak"]

    # Line-based features
    for line_name, regions in FEATURE_CONFIG["line_regions"].items():
        for j, region in enumerate(regions):
            props = measure_line_properties(flux, wavelength, region)
            suffix = f"_{j}" if len(regions) > 1 else ""
            features[f"{line_name}{suffix}_ew"] = props["equivalent_width"]
            features[f"{line_name}{suffix}_depth"] = props["line_depth"]
            features[f"{line_name}{suffix}_fwhm"] = props["fwhm"]
            features[f"{line_name}{suffix}_asym"] = props["asymmetry_index"]

    # Global spectral features
    features["spectral_rms"] = np.std(flux[np.isfinite(flux)])
    features["spectral_skew"] = _skewness(flux[np.isfinite(flux)])

    return features


def _skewness(x):
    """Compute skewness of an array."""
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return np.mean(((x - m) / s) ** 3)
