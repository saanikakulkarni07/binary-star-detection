"""Spectral preprocessing: continuum normalization, masking, batch preparation."""

import numpy as np
from scipy.ndimage import median_filter
from config import PREPROCESS_CONFIG


def mask_bad_pixels(flux, bitmask, fill_method="interpolate"):
    """Replace bad pixels (flagged in APOGEE bitmask) with interpolated values.

    Parameters
    ----------
    flux : np.ndarray (n_pixels,)
    bitmask : np.ndarray (n_pixels,) or None
    fill_method : str
        'interpolate' for linear interpolation, 'zero' for zeroing out.

    Returns
    -------
    np.ndarray (n_pixels,)
        Cleaned flux array.
    """
    if bitmask is None:
        return flux.copy()

    cleaned = flux.copy()
    bad = (bitmask != 0) | ~np.isfinite(flux)

    if not np.any(bad):
        return cleaned

    if fill_method == "interpolate":
        good_idx = np.where(~bad)[0]
        bad_idx = np.where(bad)[0]
        if len(good_idx) > 1:
            cleaned[bad_idx] = np.interp(bad_idx, good_idx, cleaned[good_idx])
        else:
            cleaned[bad] = 0.0
    else:
        cleaned[bad] = 0.0

    return cleaned


def continuum_normalize(flux, wavelength, method="polynomial", order=4, sigma_clip=3.0, n_iter=5):
    """Fit and divide out the pseudo-continuum.

    Parameters
    ----------
    flux : np.ndarray (n_pixels,)
    wavelength : np.ndarray (n_pixels,)
    method : str
        'polynomial' or 'median_filter'.
    order : int
        Polynomial order (if method='polynomial').
    sigma_clip : float
        Sigma threshold for iterative clipping of absorption lines.
    n_iter : int
        Number of sigma-clipping iterations.

    Returns
    -------
    np.ndarray (n_pixels,)
        Continuum-normalized flux (centered around 1.0).
    """
    if method == "median_filter":
        kernel_size = 501  # wide enough to smooth over lines
        continuum = median_filter(flux, size=kernel_size)
        continuum[continuum <= 0] = 1.0
        return flux / continuum

    # Polynomial fit with iterative sigma-clipping
    mask = np.isfinite(flux) & (flux > 0)
    x = wavelength.copy()
    # Normalize x to [-1, 1] for numerical stability
    x_norm = 2 * (x - x.min()) / (x.max() - x.min()) - 1

    for _ in range(n_iter):
        if mask.sum() < order + 1:
            break
        coeffs = np.polyfit(x_norm[mask], flux[mask], order)
        continuum = np.polyval(coeffs, x_norm)
        residual = flux - continuum
        std = np.std(residual[mask])
        # Clip absorption lines (below continuum) more aggressively
        mask = mask & (residual > -sigma_clip * std) & (residual < sigma_clip * 2 * std)

    continuum = np.polyval(coeffs, x_norm)
    continuum[continuum <= 0] = 1.0
    return flux / continuum


def prepare_spectrum(flux, wavelength, bitmask=None):
    """End-to-end preprocessing of a single APOGEE spectrum.

    Parameters
    ----------
    flux : np.ndarray (n_pixels,)
    wavelength : np.ndarray (n_pixels,)
    bitmask : np.ndarray (n_pixels,), optional

    Returns
    -------
    np.ndarray (n_pixels,)
        Clean, continuum-normalized spectrum.
    """
    cleaned = mask_bad_pixels(flux, bitmask)
    normalized = continuum_normalize(
        cleaned, wavelength,
        method=PREPROCESS_CONFIG["continuum_method"],
        order=PREPROCESS_CONFIG["continuum_poly_order"],
    )
    return normalized


def batch_prepare_spectra(flux_array, wavelength, bitmask_array=None, n_workers=1):
    """Process multiple spectra.

    Parameters
    ----------
    flux_array : np.ndarray (n_spectra, n_pixels)
    wavelength : np.ndarray (n_pixels,)
    bitmask_array : np.ndarray (n_spectra, n_pixels), optional
    n_workers : int
        Number of parallel workers (1 = sequential).

    Returns
    -------
    np.ndarray (n_spectra, n_pixels)
        Preprocessed spectra matrix.
    """
    from tqdm import tqdm

    n_spectra = flux_array.shape[0]
    result = np.empty_like(flux_array)

    for i in tqdm(range(n_spectra), desc="Preprocessing spectra"):
        bitmask_i = bitmask_array[i] if bitmask_array is not None else None
        result[i] = prepare_spectrum(flux_array[i], wavelength, bitmask_i)

    return result
