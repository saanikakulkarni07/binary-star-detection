"""APOGEE DR17 data access and loading utilities."""

import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import download_file
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_CONFIG, LABEL_CONFIG


def download_allstar_catalog(cache_dir=None):
    """Download the APOGEE DR17 allStarLite catalog.

    On Fornax, checks for the file in the SDSS shared data directory first.
    Falls back to HTTP download with astropy caching.

    Returns
    -------
    astropy.table.Table
        The allStarLite catalog.
    """
    cache_dir = cache_dir or DATA_CONFIG["local_cache"]
    local_path = Path(cache_dir) / "allStarLite-dr17-synspec_rev1.fits"

    # Check if already cached locally
    if local_path.exists():
        print(f"Loading cached catalog from {local_path}")
        return Table.read(local_path)

    # On Fornax, check shared SDSS data mount
    fornax_path = Path(
        "/data/sdss/dr17/apogee/spectro/aspcap/"
        "dr17/synspec_rev1/allStarLite-dr17-synspec_rev1.fits"
    )
    if fornax_path.exists():
        print(f"Loading from Fornax shared data: {fornax_path}")
        return Table.read(fornax_path)

    # Download via HTTP
    print("Downloading allStarLite catalog (~200 MB)...")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cached = download_file(DATA_CONFIG["allstar_url"], cache=True)
    # Copy to local cache for faster reloads
    import shutil
    shutil.copy(cached, local_path)
    print(f"Saved to {local_path}")
    return Table.read(local_path)


def load_allstar_catalog(path=None):
    """Load the allStarLite catalog from a local FITS file.

    Parameters
    ----------
    path : str or Path, optional
        Path to the FITS file. Defaults to data/allStarLite-dr17-synspec_rev1.fits.

    Returns
    -------
    astropy.table.Table
    """
    path = path or Path(DATA_CONFIG["local_cache"]) / "allStarLite-dr17-synspec_rev1.fits"
    return Table.read(path)


def get_visit_spectra(apogee_id, telescope="apo25m", field=None, cache_dir=None):
    """Load multi-visit spectra from an apStar file for a given star.

    Parameters
    ----------
    apogee_id : str
        APOGEE ID (e.g., '2M00000068+5710233').
    telescope : str
        Telescope name ('apo25m' or 'lco25m').
    field : str, optional
        Field name. If None, tries to resolve from the allStar catalog.
    cache_dir : str, optional
        Local cache directory.

    Returns
    -------
    dict with keys:
        - 'flux': np.ndarray (n_visits, 8575)
        - 'error': np.ndarray (n_visits, 8575)
        - 'wavelength': np.ndarray (8575,)
        - 'bitmask': np.ndarray (n_visits, 8575)
        - 'rv_per_visit': np.ndarray (n_visits,)
        - 'mjd_per_visit': np.ndarray (n_visits,)
        - 'snr_per_visit': np.ndarray (n_visits,)
    """
    cache_dir = cache_dir or DATA_CONFIG["local_cache"]
    filename = f"apStar-dr17-{apogee_id}.fits"
    local_path = Path(cache_dir) / "apstar" / filename

    if not local_path.exists():
        if field is None:
            raise ValueError(
                f"Field required to download apStar for {apogee_id}. "
                "Pass field= or use resolve_apstar_field()."
            )
        url = DATA_CONFIG["apstar_url_template"].format(
            telescope=telescope, field=field, apogee_id=apogee_id
        )
        print(f"Downloading {filename}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cached = download_file(url, cache=True)
        import shutil
        shutil.copy(cached, local_path)

    with fits.open(local_path) as hdul:
        # HDU 1: flux (row 0 = combined, rows 1+ = individual visits)
        flux_all = hdul[1].data.astype(np.float32)
        error_all = hdul[2].data.astype(np.float32)
        bitmask = hdul[3].data

        # Build wavelength from header WCS
        header = hdul[1].header
        n_pix = header["NAXIS1"]
        crval = header["CRVAL1"]
        cdelt = header["CDELT1"]
        crpix = header.get("CRPIX1", 1)
        log_wave = crval + cdelt * (np.arange(n_pix) - (crpix - 1))
        wavelength = 10.0**log_wave  # APOGEE uses log10(lambda) in Angstroms

        # Row 0 is the combined spectrum; rows 1: are individual visits
        n_visits = flux_all.shape[0] - 1
        if n_visits < 1:
            raise ValueError(f"No individual visits found for {apogee_id}")

        flux_visits = flux_all[1:, :]
        error_visits = error_all[1:, :]
        bitmask_visits = bitmask[1:, :] if bitmask.ndim > 1 else None

        # Per-visit RVs and MJDs from the RV table
        # DR17 apStar files vary in HDU layout and column names
        rv_per_visit = np.full(n_visits, np.nan)
        mjd_per_visit = np.full(n_visits, np.nan)
        snr_per_visit = np.full(n_visits, np.nan)

        for hdu_idx in range(len(hdul)):
            if hdul[hdu_idx].data is None:
                continue
            try:
                colnames = [c.upper() for c in hdul[hdu_idx].columns.names]
            except (AttributeError, TypeError):
                continue

            # Look for the HDU containing per-visit RV data
            has_rv = any(c in colnames for c in ["VHELIO", "VHELIO_AVG", "BC", "VREL"])
            has_time = any(c in colnames for c in ["MJD", "JD", "BJDOBS", "MJD-OBS"])
            if has_rv or has_time:
                rv_tab = hdul[hdu_idx].data
                actual_cols = hdul[hdu_idx].columns.names
                col_map = {c.upper(): c for c in actual_cols}

                for rv_key in ["VHELIO", "VREL", "BC"]:
                    if rv_key in col_map:
                        rv_per_visit = np.array(rv_tab[col_map[rv_key]], dtype=np.float64)[:n_visits]
                        break

                for mjd_key in ["MJD", "JD", "BJDOBS", "MJD-OBS"]:
                    if mjd_key in col_map:
                        mjd_per_visit = np.array(rv_tab[col_map[mjd_key]], dtype=np.float64)[:n_visits]
                        break

                for snr_key in ["SNR", "SN"]:
                    if snr_key in col_map:
                        snr_per_visit = np.array(rv_tab[col_map[snr_key]], dtype=np.float32)[:n_visits]
                        break
                break

    return {
        "flux": flux_visits,
        "flux_combined": flux_all[0, :],
        "error": error_visits,
        "wavelength": wavelength,
        "bitmask": bitmask_visits,
        "rv_per_visit": rv_per_visit,
        "mjd_per_visit": mjd_per_visit,
        "snr_per_visit": snr_per_visit,
        "n_visits": n_visits,
    }


def validate_allstar_entry(row):
    """Check whether an allStar catalog entry passes quality cuts.

    Parameters
    ----------
    row : astropy.table.Row or dict
        A single row from the allStarLite catalog.

    Returns
    -------
    bool
        True if the star passes all quality cuts.
    """
    # S/N cut
    if row["SNR"] < LABEL_CONFIG["snr_min"]:
        return False

    # Minimum visits
    if row["NVISITS"] < LABEL_CONFIG["min_visits"]:
        return False

    # Bad star flags (bit 23 = STAR_BAD)
    if row["STARFLAG"] & (1 << 23):
        return False

    # Temperature range
    teff = row["TEFF"]
    tmin, tmax = LABEL_CONFIG["teff_range"]
    if teff < tmin or teff > tmax:
        return False

    # Surface gravity range
    logg = row["LOGG"]
    gmin, gmax = LABEL_CONFIG["logg_range"]
    if logg < gmin or logg > gmax:
        return False

    return True


def filter_allstar_catalog(catalog):
    """Apply quality cuts to the full allStar catalog.

    Parameters
    ----------
    catalog : astropy.table.Table
        The full allStarLite catalog.

    Returns
    -------
    astropy.table.Table
        Filtered catalog with quality cuts applied.
    dict
        Waterfall summary of cuts applied.
    """
    waterfall = {"initial": len(catalog)}

    mask = catalog["SNR"] >= LABEL_CONFIG["snr_min"]
    waterfall["snr_cut"] = mask.sum()

    mask &= catalog["NVISITS"] >= LABEL_CONFIG["min_visits"]
    waterfall["nvisits_cut"] = mask.sum()

    mask &= (catalog["STARFLAG"] & (1 << 23)) == 0
    waterfall["starflag_cut"] = mask.sum()

    tmin, tmax = LABEL_CONFIG["teff_range"]
    mask &= (catalog["TEFF"] >= tmin) & (catalog["TEFF"] <= tmax)
    waterfall["teff_cut"] = mask.sum()

    gmin, gmax = LABEL_CONFIG["logg_range"]
    mask &= (catalog["LOGG"] >= gmin) & (catalog["LOGG"] <= gmax)
    waterfall["logg_cut"] = mask.sum()

    waterfall["final"] = mask.sum()
    return catalog[mask], waterfall
