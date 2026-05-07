"""Classical radial velocity analysis: CCF measurement, period search, orbit fitting."""

import numpy as np
from scipy.optimize import curve_fit, differential_evolution
from astropy.timeseries import LombScargle


def compute_ccf_rv(flux, wavelength, template_flux, rv_grid=None):
    """Measure radial velocity via cross-correlation with a template.

    Parameters
    ----------
    flux : np.ndarray (n_pixels,)
    wavelength : np.ndarray (n_pixels,)
    template_flux : np.ndarray (n_pixels,)
    rv_grid : np.ndarray, optional
        RV values to evaluate (km/s). Default: -500 to 500 in 0.5 km/s steps.

    Returns
    -------
    dict with: rv, rv_err, ccf_peak_height, ccf, rv_grid
    """
    c_kms = 2.998e5
    if rv_grid is None:
        rv_grid = np.arange(-500, 500, 0.5)

    log_wave = np.log(wavelength)
    dlog = np.median(np.diff(log_wave))
    shift_pixels = (rv_grid / c_kms) / dlog

    f_obs = flux - np.nanmean(flux)
    f_tmpl = template_flux - np.nanmean(template_flux)

    ccf = np.zeros(len(rv_grid))
    for i, sp in enumerate(shift_pixels):
        shifted = np.interp(
            np.arange(len(f_tmpl)) - sp,
            np.arange(len(f_tmpl)),
            f_tmpl,
        )
        ccf[i] = np.nansum(f_obs * shifted)

    ccf /= np.max(np.abs(ccf)) if np.max(np.abs(ccf)) > 0 else 1.0

    # Fit Gaussian to peak
    peak_idx = np.argmax(ccf)
    fit_mask = np.abs(rv_grid - rv_grid[peak_idx]) < 30
    try:
        def gauss(x, a, mu, sig, c):
            return a * np.exp(-0.5 * ((x - mu) / sig) ** 2) + c

        popt, pcov = curve_fit(
            gauss, rv_grid[fit_mask], ccf[fit_mask],
            p0=[ccf[peak_idx], rv_grid[peak_idx], 5.0, 0.0],
            maxfev=5000,
        )
        rv = popt[1]
        rv_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else np.nan
        peak_height = popt[0] + popt[3]
    except (RuntimeError, ValueError):
        rv = rv_grid[peak_idx]
        rv_err = np.nan
        peak_height = ccf[peak_idx]

    return {
        "rv": rv,
        "rv_err": rv_err,
        "ccf_peak_height": peak_height,
        "ccf": ccf,
        "rv_grid": rv_grid,
    }


def measure_per_visit_rvs(visit_spectra, template_flux):
    """Measure RV for each visit of a multi-epoch star.

    Parameters
    ----------
    visit_spectra : dict
        Output of data.get_visit_spectra().
    template_flux : np.ndarray (n_pixels,)

    Returns
    -------
    dict with: mjd, rv, rv_err (all np.ndarray of length n_visits)
    """
    n_visits = visit_spectra["n_visits"]
    rvs = np.zeros(n_visits)
    rv_errs = np.zeros(n_visits)

    for i in range(n_visits):
        result = compute_ccf_rv(
            visit_spectra["flux"][i],
            visit_spectra["wavelength"],
            template_flux,
        )
        rvs[i] = result["rv"]
        rv_errs[i] = result["rv_err"]

    return {
        "mjd": visit_spectra["mjd_per_visit"],
        "rv": rvs,
        "rv_err": rv_errs,
    }


def period_search(mjd, rv, rv_err, min_period=0.5, max_period=1000):
    """Run a Lomb-Scargle periodogram on RV measurements.

    Parameters
    ----------
    mjd : np.ndarray — observation dates
    rv : np.ndarray — radial velocities (km/s)
    rv_err : np.ndarray — RV uncertainties
    min_period : float — minimum period to search (days)
    max_period : float — maximum period to search (days)

    Returns
    -------
    dict with: best_period, best_power, fap, frequency, power
    """
    ls = LombScargle(mjd, rv, rv_err)
    frequency, power = ls.autopower(
        minimum_frequency=1.0 / max_period,
        maximum_frequency=1.0 / min_period,
    )
    best_freq = frequency[np.argmax(power)]
    best_period = 1.0 / best_freq
    fap = ls.false_alarm_probability(power.max())

    return {
        "best_period": best_period,
        "best_power": power.max(),
        "fap": fap,
        "frequency": frequency,
        "power": power,
    }


def keplerian_rv(t, period, K, ecc, omega, T0, gamma):
    """Compute Keplerian RV curve.

    Parameters
    ----------
    t : np.ndarray — times (MJD)
    period : float — orbital period (days)
    K : float — semi-amplitude (km/s)
    ecc : float — eccentricity (0-1)
    omega : float — argument of periastron (radians)
    T0 : float — time of periastron passage (MJD)
    gamma : float — systemic velocity (km/s)

    Returns
    -------
    np.ndarray — predicted RV values
    """
    # Mean anomaly
    M = 2 * np.pi * (t - T0) / period
    M = M % (2 * np.pi)

    # Solve Kepler's equation iteratively
    E = M.copy()
    for _ in range(50):
        E = M + ecc * np.sin(E)

    # True anomaly
    nu = 2 * np.arctan2(
        np.sqrt(1 + ecc) * np.sin(E / 2),
        np.sqrt(1 - ecc) * np.cos(E / 2),
    )

    return gamma + K * (np.cos(nu + omega) + ecc * np.cos(omega))


def fit_orbit(mjd, rv, rv_err, period_guess):
    """Fit a Keplerian orbit to RV measurements.

    Parameters
    ----------
    mjd, rv, rv_err : np.ndarray
    period_guess : float — initial period estimate (days)

    Returns
    -------
    dict with: period, K, ecc, omega, T0, gamma, reduced_chi2, params_err
    """
    gamma_guess = np.median(rv)
    K_guess = (np.max(rv) - np.min(rv)) / 2

    bounds = [
        (period_guess * 0.8, period_guess * 1.2),  # period
        (0.1, K_guess * 3),                         # K
        (0.0, 0.9),                                  # ecc
        (0, 2 * np.pi),                              # omega
        (mjd.min(), mjd.max()),                       # T0
        (gamma_guess - 50, gamma_guess + 50),         # gamma
    ]

    def chi2(params):
        model = keplerian_rv(mjd, *params)
        return np.sum(((rv - model) / rv_err) ** 2)

    result = differential_evolution(chi2, bounds, seed=42, maxiter=1000)
    params = result.x
    dof = len(rv) - 6
    reduced_chi2 = result.fun / dof if dof > 0 else np.inf

    return {
        "period": params[0],
        "K": params[1],
        "ecc": params[2],
        "omega": params[3],
        "T0": params[4],
        "gamma": params[5],
        "reduced_chi2": reduced_chi2,
        "success": result.success,
    }
