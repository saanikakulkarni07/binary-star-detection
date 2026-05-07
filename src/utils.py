"""Plotting, metrics, and common helper utilities."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
)
from config import VIS_CONFIG


def plot_spectrum(wavelength, flux, title="", ax=None, label=None, alpha=1.0):
    """Plot a single spectrum."""
    if ax is None:
        fig, ax = plt.subplots(figsize=VIS_CONFIG["figure_size"], dpi=VIS_CONFIG["dpi"])
    ax.plot(wavelength, flux, linewidth=0.5, label=label, alpha=alpha)
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(title)
    if label:
        ax.legend()
    return ax


def plot_rv_curve(mjd, rv, rv_err, title="", model_mjd=None, model_rv=None, ax=None):
    """Plot RV measurements with optional Keplerian model overlay."""
    if ax is None:
        fig, ax = plt.subplots(figsize=VIS_CONFIG["figure_size"], dpi=VIS_CONFIG["dpi"])
    ax.errorbar(mjd, rv, yerr=rv_err, fmt="ko", capsize=3, label="Observed")
    if model_mjd is not None and model_rv is not None:
        ax.plot(model_mjd, model_rv, "r-", linewidth=1.5, label="Keplerian fit")
        ax.legend()
    ax.set_xlabel("MJD")
    ax.set_ylabel("RV (km/s)")
    ax.set_title(title)
    return ax


def plot_roc_pr(y_true, y_probs, labels=None, title_prefix=""):
    """Plot ROC and precision-recall curves for one or more models.

    Parameters
    ----------
    y_true : np.ndarray
    y_probs : dict of {model_name: predicted_probabilities}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=VIS_CONFIG["dpi"])

    for name, probs in y_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auroc = roc_auc_score(y_true, probs)
        ax1.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, probs)
        ax2.plot(recall, precision, label=name)

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"{title_prefix}ROC Curve")
    ax1.legend()

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"{title_prefix}Precision-Recall Curve")
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=("Single", "Binary"), ax=None):
    """Plot a confusion matrix heatmap."""
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=VIS_CONFIG["dpi"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    return ax


def print_metrics(y_true, y_pred, y_probs):
    """Print classification metrics."""
    print(classification_report(y_true, y_pred, target_names=["Single", "Binary"]))
    auroc = roc_auc_score(y_true, y_probs)
    print(f"AUROC: {auroc:.4f}")
    return auroc


def save_figure(fig, name):
    """Save a figure to the figures directory."""
    path = f"{VIS_CONFIG['figures_dir']}{name}.{VIS_CONFIG['save_format']}"
    fig.savefig(path, bbox_inches="tight", dpi=VIS_CONFIG["dpi"])
    print(f"Saved: {path}")


def crossmatch_catalogs(table1, table2, ra1="RA", dec1="DEC",
                        ra2="RA", dec2="DEC", radius_arcsec=2.0):
    """Positional cross-match between two catalogs.

    Parameters
    ----------
    table1, table2 : astropy.table.Table
    ra1, dec1, ra2, dec2 : str — column names for coordinates
    radius_arcsec : float — matching radius

    Returns
    -------
    matched_idx1, matched_idx2 : np.ndarray — indices of matched rows
    separations : astropy.units.Quantity — angular separations
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coords1 = SkyCoord(ra=table1[ra1], dec=table1[dec1], unit="deg")
    coords2 = SkyCoord(ra=table2[ra2], dec=table2[dec2], unit="deg")

    idx2, sep, _ = coords1.match_to_catalog_sky(coords2)
    mask = sep < radius_arcsec * u.arcsec

    matched_idx1 = np.where(mask)[0]
    matched_idx2 = idx2[mask]
    separations = sep[mask]

    return matched_idx1, matched_idx2, separations
