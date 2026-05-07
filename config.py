"""Centralized configuration for binary star detection pipeline."""

# --- Data Access ---
DATA_CONFIG = {
    "survey": "apogee",
    "data_release": "dr17",
    "local_cache": "data/",
    "allstar_url": (
        "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/"
        "dr17/synspec_rev1/allStarLite-dr17-synspec_rev1.fits"
    ),
    "apstar_url_template": (
        "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/"
        "stars/{telescope}/{field}/apStar-dr17-{apogee_id}.fits"
    ),
}

# --- Label Generation ---
LABEL_CONFIG = {
    "rv_scatter_binary_threshold": 1.0,   # km/s — stars above are binary candidates
    "rv_scatter_single_threshold": 0.3,   # km/s — stars below are single
    "min_visits": 3,                       # minimum APOGEE visits for reliable scatter
    "snr_min": 50,                         # minimum combined S/N
    "logg_range": (-0.5, 5.0),
    "teff_range": (3500, 7000),
}

# --- Preprocessing ---
PREPROCESS_CONFIG = {
    "n_pixels": 8575,                      # apStar wavelength grid size
    "continuum_method": "polynomial",
    "continuum_poly_order": 4,
    "mask_bad_pixels": True,
    "normalize": True,
}

# --- Feature Engineering ---
FEATURE_CONFIG = {
    "ccf_template_types": ["F5V", "G2V", "K0III", "K5V", "M2V"],
    "line_regions": {
        "Brackett_series": [(15556, 15575), (16109, 16135), (16407, 16422)],
        "CO_bandhead": [(15975, 16000), (16175, 16215)],
        "OH_lines": [(15505, 15525), (16190, 16210)],
        "Mg_I": [(15740, 15780)],
        "Fe_I": [(15207, 15218), (15395, 15410), (15648, 15660)],
        "Al_I": [(16718, 16730), (16750, 16770)],
    },
}

# --- Model ---
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.15,
    "val_size": 0.15,
    # Random Forest baseline
    "rf_n_estimators": 500,
    "rf_max_depth": 20,
    "rf_class_weight": "balanced",
    # 1D CNN
    "cnn_learning_rate": 1e-3,
    "cnn_batch_size": 128,
    "cnn_epochs": 50,
    "cnn_patience": 10,
    "cnn_dropout": 0.3,
}

# --- Validation ---
VALIDATION_CONFIG = {
    "sb9_catalog_path": "data/sb9_catalog.csv",
    "gaia_nss_table": "gaiadr3.nss_two_body_orbit",
}

# --- Visualization ---
VIS_CONFIG = {
    "figure_size": (10, 6),
    "dpi": 150,
    "save_format": "png",
    "figures_dir": "figures/",
}
