"""
Exoplanet Transit Detection Pipeline
=====================================
Standalone CLI that mirrors `exoplanet_colab.ipynb` methodology end to end:

    data acquisition -> phase folding -> preprocessing -> 30-feature extraction
    -> group-aware train/val/test split -> 9 traditional classifiers + Stacking
    -> DualView CNN (Focal Loss, AdamW, warm restarts, Mixup, 7-pass TTA, Youden)
    -> BiLSTM + soft attention (same schedule) -> SHAP + Grad-CAM
    -> prototype JSON export

Methodology is kept deliberately identical to the notebook so that results
produced here are directly comparable with notebook results.

Usage:
    pip install -r requirements.txt

    # Train on a batch of KOIs (default demo list is used when --ids is omitted)
    python pipeline.py --mode batch --ids "K00001.01,K00007.01,K00017.01"

    # Single KOI / TIC
    python pipeline.py --mode kepler --id "K00001.01"
    python pipeline.py --mode tess   --id 394137592

    # Local CSV (time/flux or phase/flux columns)
    python pipeline.py --mode csv --file my_lightcurve.csv

    # Inference only, reusing previously trained artefacts in saved_models/
    python pipeline.py --mode infer --file my_lightcurve.csv
"""

import argparse
import base64
import io
import json
import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import kurtosis, skew, spearmanr

warnings.filterwarnings("ignore")

# ── Configuration (mirrors the notebook constants) ───────────────────────────
SEQ_LEN         = 2000
LOCAL_LEN       = 200
LOCAL_CENTER    = 0.5
LOCAL_WIDTH     = 0.14
CNN_EPOCHS      = 80
CNN_LR          = 5e-4
CNN_WEIGHT_DECAY = 3e-4
CNN_BATCH_SIZE  = 16
MIXUP_ALPHA     = 0.15
FOCAL_GAMMA     = 2.0
TTA_PASSES      = 7
TTA_NOISE       = 0.001
EARLY_STOP_PATIENCE = 18
TEST_FRACTION   = 0.15
TEST_SEED       = 99
VAL_FRACTION    = 0.20
VAL_SEED        = 42
TRANSIT_WINDOW  = (0.45, 0.55)
APPLY_SG_FILTER = True
NORMALIZE_FLUX  = True
RANDOM_STATE    = 42
PROTOTYPE_POINTS = 400

NASA_TAP   = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
CUMUL_URL  = (
    NASA_TAP + "?query=SELECT+kepoi_name,kepid,koi_pdisposition,"
    "koi_period,koi_duration,koi_depth,koi_prad,koi_sma,koi_teq,"
    "koi_insol,koi_impact,koi_srad,koi_smass,koi_steff,koi_slogg,"
    "koi_kepmag,koi_score,ra,dec+FROM+cumulative&format=csv"
)
TOI_URL    = (
    NASA_TAP + "?query=SELECT+tid,tfopwg_disp,pl_orbper,pl_trandurh,"
    "pl_trandep,pl_rade,pl_orbsmax,pl_eqt,pl_insol,pl_imppar,"
    "st_rad,st_mass,st_teff,st_logg,st_tmag,ra,dec+FROM+toi&format=csv"
)

# Optional dependencies. The notebook treats all of these as optional and so
# does this pipeline; each missing package degrades gracefully.
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

    # The network classes below are declared at import time, so without torch
    # we still need a base class to inherit from. Every real entry point calls
    # _require_torch() first, so this stub is never actually instantiated.
    class _MissingTorchModule:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not installed. Run: pip install torch")

    class _MissingTorchNN:
        Module = _MissingTorchModule
        Sequential = _MissingTorchModule
        Conv1d = _MissingTorchModule
        BatchNorm1d = _MissingTorchModule
        Linear = _MissingTorchModule
        LSTM = _MissingTorchModule
        BatchNorm1d_1d = _MissingTorchModule
        AdaptiveAvgPool1d = _MissingTorchModule
        Dropout = _MissingTorchModule
        GELU = _MissingTorchModule
        Sigmoid = _MissingTorchModule
        Parameter = _MissingTorchModule
        ParameterList = _MissingTorchModule

    nn = _MissingTorchNN()
    torch = None
    F = None


# ─────────────────────────────────────────────
# 1. NASA ARCHIVE TABLES
# ─────────────────────────────────────────────

def _fetch_archive_tables():
    """Download the full cumulative KOI and TOI tables once."""
    import requests

    koi_df = None
    toi_df = None

    print("Downloading cumulative KOI table...", end=" ", flush=True)
    try:
        koi_df = pd.read_csv(io.StringIO(requests.get(CUMUL_URL, timeout=60).text))
        koi_df.columns = [c.strip() for c in koi_df.columns]
        koi_df["kepoi_name"] = koi_df["kepoi_name"].str.strip()
        koi_df = koi_df.set_index("kepoi_name")
        print(f"OK  {len(koi_df)} rows")
    except Exception as exc:
        print(f"WARN  {exc}  (falling back to per-star API)")

    print("Downloading TOI table...", end=" ", flush=True)
    try:
        toi_df = pd.read_csv(io.StringIO(requests.get(TOI_URL, timeout=60).text))
        toi_df.columns = [c.strip() for c in toi_df.columns]
        toi_df["tid"] = toi_df["tid"].astype(str).str.strip()
        toi_df = toi_df.set_index("tid")
        print(f"OK  {len(toi_df)} rows")
    except Exception as exc:
        print(f"WARN  {exc}")

    return koi_df, toi_df


_ARCHIVE_TABLES = None


def archive_tables():
    """Lazily download and cache the archive tables."""
    global _ARCHIVE_TABLES
    if _ARCHIVE_TABLES is None:
        _ARCHIVE_TABLES = _fetch_archive_tables()
    return _ARCHIVE_TABLES


def koi_to_kic(koi_id, koi_df=None):
    """Resolve a KOI ID to its KIC number.

    Raises ValueError when the mapping cannot be established. The previous
    implementation silently substituted a fabricated KIC for unresolved IDs,
    which folded and classified the wrong star.
    """
    if koi_df is None:
        koi_df, _ = archive_tables()

    try:
        if koi_df is not None and koi_id in koi_df.index:
            return str(int(koi_df.loc[koi_id, "kepid"]))
    except Exception:
        pass

    import requests
    try:
        query = f"SELECT kepid FROM cumulative WHERE kepoi_name='{koi_id}'"
        resp = requests.get(
            NASA_TAP, params={"query": query, "format": "json"}, timeout=20
        )
        data = resp.json()
        if data:
            return str(data[0]["kepid"])
    except Exception:
        pass

    raise ValueError(
        f"Could not resolve KIC for {koi_id} — check the KOI ID format. "
        "Refusing to substitute a substitute star."
    )


def fetch_koi_period_and_label(koi_id, koi_df=None):
    """Return (period_days, label) for a KOI ID."""
    if koi_df is None:
        koi_df, _ = archive_tables()

    period, label = 5.0, "unknown"
    try:
        if koi_df is not None and koi_id in koi_df.index:
            row = koi_df.loc[koi_id]
            period = float(row.get("koi_period") or 5.0)
            disp = str(row.get("koi_pdisposition") or "").upper()
        else:
            import requests
            query = (
                "SELECT koi_period,koi_pdisposition FROM cumulative "
                f"WHERE kepoi_name='{koi_id}'"
            )
            resp = requests.get(
                NASA_TAP, params={"query": query, "format": "json"}, timeout=20
            )
            data = resp.json()
            if not data:
                return period, label
            period = float(data[0].get("koi_period") or 5.0)
            disp = str(data[0].get("koi_pdisposition") or "").upper()
        if "CANDIDATE" in disp or "CONFIRMED" in disp:
            label = "planet"
        elif "FALSE" in disp:
            label = "false_positive"
    except Exception as exc:
        print(f"  Archive query failed: {exc}")
    return period, label


def fetch_tic_period_and_label(tic_id, toi_df=None):
    """Return (period_days, label) for a TIC ID."""
    if toi_df is None:
        _, toi_df = archive_tables()

    period, label = 5.0, "unknown"
    try:
        key = str(tic_id)
        if toi_df is not None and key in toi_df.index:
            row = toi_df.loc[key]
            period = float(row.get("pl_orbper") or 5.0)
            disp = str(row.get("tfopwg_disp") or "").upper()
        else:
            import requests
            query = f"SELECT pl_orbper,tfopwg_disp FROM toi WHERE tid={key}"
            resp = requests.get(
                NASA_TAP, params={"query": query, "format": "json"}, timeout=20
            )
            data = resp.json()
            if not data:
                return period, label
            period = float(data[0].get("pl_orbper") or 5.0)
            disp = str(data[0].get("tfopwg_disp") or "").upper()
        if "PC" in disp or "KP" in disp:
            label = "planet"
        elif "FP" in disp:
            label = "false_positive"
    except Exception as exc:
        print(f"  Archive query failed: {exc}")
    return period, label


def star_params_for(sid, mission, koi_df=None, toi_df=None):
    """Assemble the archive star parameters block used by the prototype.

    A locally supplied light curve has no archive identity, so no network call
    is made for it and every field stays null.
    """
    if mission not in ("Kepler", "TESS"):
        return {"source": None, "st_rad": None, "st_mass": None,
                "st_teff": None, "st_logg": None, "st_mag": None,
                "ra": None, "dec": None}

    if koi_df is None or toi_df is None:
        koi_df, toi_df = archive_tables()

    def _num(value):
        if value is None:
            return None
        try:
            text = str(value).strip()
            if text in ("", "nan", "None"):
                return None
            return float(text)
        except Exception:
            return None

    try:
        if mission == "Kepler" and koi_df is not None and sid in koi_df.index:
            row = koi_df.loc[sid]
            return {
                "mission": "Kepler",
                "period":   _num(row.get("koi_period")),
                "duration": _num(row.get("koi_duration")),
                "depth":    _num(row.get("koi_depth")),
                "prad":     _num(row.get("koi_prad")),
                "sma":      _num(row.get("koi_sma")),
                "teq":      _num(row.get("koi_teq")),
                "insol":    _num(row.get("koi_insol")),
                "impact":   _num(row.get("koi_impact")),
                "srad":     _num(row.get("koi_srad")),
                "smass":    _num(row.get("koi_smass")),
                "steff":    _num(row.get("koi_steff")),
                "slogg":    _num(row.get("koi_slogg")),
                "mag":      _num(row.get("koi_kepmag")),
                "score":    _num(row.get("koi_score")),
                "ra":       _num(row.get("ra")),
                "dec":      _num(row.get("dec")),
                "disposition": str(row.get("koi_pdisposition", "") or ""),
            }
        if mission == "TESS" and toi_df is not None:
            tic_num = str(sid).replace("TIC-", "").replace("TIC ", "")
            if tic_num in toi_df.index:
                row = toi_df.loc[tic_num]
                return {
                    "mission": "TESS",
                    "period":   _num(row.get("pl_orbper")),
                    "duration": _num(row.get("pl_trandurh")),
                    "depth":    _num(row.get("pl_trandep")),
                    "prad":     _num(row.get("pl_rade")),
                    "sma":      _num(row.get("pl_orbsmax")),
                    "teq":      _num(row.get("pl_eqt")),
                    "insol":    _num(row.get("pl_insol")),
                    "impact":   _num(row.get("pl_imppar")),
                    "srad":     _num(row.get("st_rad")),
                    "smass":    _num(row.get("st_mass")),
                    "steff":    _num(row.get("st_teff")),
                    "slogg":    _num(row.get("st_logg")),
                    "mag":      _num(row.get("st_tmag")),
                    "ra":       _num(row.get("ra")),
                    "dec":      _num(row.get("dec")),
                    "disposition": str(row.get("tfopwg_disp", "") or ""),
                }
    except Exception:
        pass
    return {}


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def bin_phase_curve(phase, flux, bins=SEQ_LEN):
    """Median-bin samples that are already expressed as phase in [0, 1]."""
    phase = np.asarray(phase, dtype=float)
    flux = np.asarray(flux, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)

    binned = np.empty(bins, dtype=float)
    for i in range(bins):
        # The last bin is closed so phase == 1.0 is never dropped.
        if i == bins - 1:
            mask = (phase >= edges[i]) & (phase <= edges[i + 1])
        else:
            mask = (phase >= edges[i]) & (phase < edges[i + 1])
        binned[i] = np.nanmedian(flux[mask]) if mask.sum() > 0 else 1.0

    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, binned


def phase_fold(time, flux, period, bins=SEQ_LEN, epoch=None):
    """Phase-fold a light curve and median-bin it to a fixed length.

    `epoch` (transit epoch T0, days) is optional. When supplied the folded phase
    is anchored so that phase 0.5 lands on the predicted transit centre:
    phase = (((t - T0 + P/2) mod P) / P). The notebook folds with the default
    `time % period`, which is retained when `epoch` is None.
    """
    if not period or period <= 0:
        raise ValueError("Orbital period must be a positive number of days")

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if epoch is not None:
        phase = (((time - float(epoch) + period / 2.0) % period) / period)
    else:
        phase = (time % period) / period

    return bin_phase_curve(phase, flux, bins=bins)


def preprocess(flux, apply_sg=APPLY_SG_FILTER, normalize=NORMALIZE_FLUX,
               target_len=SEQ_LEN):
    """Interpolate gaps, divide out the Savitzky-Golay trend, normalise, resize."""
    flux = np.asarray(flux, dtype=float)

    nans = np.isnan(flux)
    if nans.any():
        x = np.arange(len(flux))
        flux = np.interp(x, x[~nans], flux[~nans])

    if apply_sg and len(flux) > 51:
        trend = savgol_filter(flux, window_length=51, polyorder=3)
        with np.errstate(divide="ignore", invalid="ignore"):
            flux = np.where(trend != 0, flux / trend, flux)

    if normalize:
        med = np.nanmedian(flux)
        if med != 0:
            flux = flux / med

    if len(flux) != target_len:
        flux = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, len(flux)),
            flux,
        )

    return flux.astype(np.float32)


def extract_local_view(flux_array, center=LOCAL_CENTER, width=LOCAL_WIDTH,
                       out_len=LOCAL_LEN):
    """Zoomed view centred on the transit (phase ~ 0.5).

    Captures ingress/egress and bottom shape that the 2000-point global view
    blurs away. This is the local branch input of the DualView CNN.
    """
    n = len(flux_array)
    lo = max(0, int((center - width / 2.0) * n))
    hi = min(n, int((center + width / 2.0) * n))
    seg = flux_array[lo:hi] if (hi - lo) >= 5 else flux_array
    return np.interp(
        np.linspace(0, 1, out_len), np.linspace(0, 1, len(seg)), seg
    ).astype(np.float32)


def download_kepler(koi_id, koi_df=None):
    """Download and fold a Kepler light curve for a KOI ID."""
    try:
        import lightkurve as lk
    except ImportError:
        raise ImportError("Run: pip install lightkurve")

    kic = koi_to_kic(koi_id, koi_df)
    period, label = fetch_koi_period_and_label(koi_id, koi_df)

    print(f"  Searching MAST for {koi_id} (KIC {kic})...")
    collection = lk.search_lightcurve(
        f"KIC {kic}", mission="Kepler", cadence="long"
    ).download_all()
    if collection is None or len(collection) == 0:
        raise ValueError(f"No light curves found for KIC {kic}")

    lc = collection.stitch().normalize().remove_nans().remove_outliers(sigma=5)
    print(
        f"  KIC {kic} | {len(lc.time)} cadences | "
        f"{len(collection)} quarters | period={period:.3f}d | label={label}"
    )

    _, flux = phase_fold(lc.time.value, lc.flux.value, period)
    return _, flux, period, label


def download_tess(tic_id, toi_df=None):
    """Download and fold a TESS light curve for a TIC ID."""
    try:
        import lightkurve as lk
    except ImportError:
        raise ImportError("Run: pip install lightkurve")

    period, label = fetch_tic_period_and_label(tic_id, toi_df)

    print(f"  Searching MAST for TIC {tic_id}...")
    collection = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS").download_all()
    if collection is None or len(collection) == 0:
        raise ValueError(f"No light curves found for TIC {tic_id}")

    lc = collection.stitch().normalize().remove_nans().remove_outliers(sigma=5)
    print(
        f"  TIC {tic_id} | {len(lc.time)} cadences | "
        f"period={period:.3f}d | label={label}"
    )

    _, flux = phase_fold(lc.time.value, lc.flux.value, period)
    return _, flux, period, label


def resolve_csv_path(directory, stem):
    """Locate a light-curve CSV from a bare name or a full filename.

    `--ids K00001.01` together with `--csv-dir` should work exactly like
    `--ids K00001.01.csv`, so the extension is resolved here.
    """
    candidate = stem if os.path.isabs(stem) else os.path.join(directory, stem)
    if os.path.isfile(candidate):
        return candidate
    for ext in (".csv", ".txt", ".dat", ".CSV"):
        if os.path.isfile(candidate + ext):
            return candidate + ext
    matches = [f for f in os.listdir(directory)
               if os.path.splitext(f)[0].lower() == os.path.basename(stem).lower()]
    if matches:
        return os.path.join(directory, sorted(matches)[0])
    return candidate


def load_csv(filepath):
    """Load a light curve from CSV.

    If the time-like column already spans [0, 1] it is treated as a phase-folded
    curve and median-binned directly. Otherwise a `period` column is required.
    Returns (phase, flux, period, label).
    """
    df = pd.read_csv(filepath, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]

    time_cols = ["time", "btjd", "bkjd", "bjd", "hjd", "t", "cadenceno", "phase"]
    tcol = next((c for c in df.columns if any(n in c for n in time_cols)),
                df.columns[0])
    flux_cols = ["pdcsap_flux", "sap_flux", "flux", "normalized_flux",
                 "rel_flux", "f"]
    fcol = next((c for c in df.columns if any(n in c for n in flux_cols)),
                df.columns[1])
    pcol = next((c for c in df.columns if "period" in c or c == "p"), None)
    lcol = next((c for c in df.columns
                 if c in ("label", "class", "disposition", "y", "target")), None)

    print(f"  Using columns: time='{tcol}', flux='{fcol}'"
          + (f", period='{pcol}'" if pcol else "")
          + (f", label='{lcol}'" if lcol else ""))
    df = df[[tcol, fcol]].dropna().astype(float)
    t = df[tcol].values
    f = df[fcol].values
    f = f / np.nanmedian(f)

    label = "unknown"
    if lcol is not None:
        raw = str(pd.read_csv(filepath, comment="#")[lcol].dropna().iloc[0]).strip()
        low = raw.lower()
        if low in ("planet", "cpc", "confirmed", "1", "true", "pos"):
            label = "planet"
        elif low in ("false_positive", "fp", "not", "0", "false", "neg"):
            label = "false_positive"
        else:
            print(f"  label '{raw}' is not determinate - treated as unknown.")

    if pcol is not None:
        period = float(pd.read_csv(filepath, comment="#")[pcol].dropna().iloc[0])
        centers, binned = phase_fold(t, f, period)
        return centers, binned, period, label

    already_folded = float(t.min()) >= 0.0 and float(t.max()) <= 1.0
    if already_folded:
        # The column already *is* phase, so bin it on that value directly.
        centers, binned = bin_phase_curve(t, f)
        return centers, binned, None, label

    raise ValueError(
        f"CSV time column '{tcol}' spans outside [0, 1] but no orbital period "
        "column is present. Add a 'period' column, or supply a pre-folded curve."
    )


# ─────────────────────────────────────────────
# 3. FEATURE EXTRACTION (30 features)
# ─────────────────────────────────────────────

def extract_features(flux):
    """Extract the 30 tabular features used by every traditional classifier.

    Key names and order match the notebook exactly so that `feat_names` and any
    existing exported JSON stay compatible.
    """
    n = len(flux)
    med = float(np.median(flux))
    mean = float(np.mean(flux))
    phase_arr = np.linspace(0, 1, n)

    depth = float(med - np.min(flux))
    thresh = med - 0.5 * depth
    in_tr = flux < thresh

    duration = float(in_tr.sum() / n)

    out_flux = flux[~in_tr] if (~in_tr).sum() > 3 else flux
    noise = float(np.std(out_flux))
    snr = float(depth / noise) if noise > 0 else 0.0

    mid = n // 2
    even_d = float(med - np.min(flux[:mid]))
    odd_d = float(med - np.min(flux[mid:]))
    even_odd_diff = float(abs(even_d - odd_d))

    s0, s1 = int(0.45 * n), int(0.55 * n)
    sec_reg = flux[s0:s1]
    sec_depth = float(med - np.min(sec_reg)) if len(sec_reg) > 0 else 0.0
    sec_ratio = float(sec_depth / depth) if depth > 0 else 0.0

    flat_frac = float(
        np.sum(flux[in_tr] < (med - 0.8 * depth)) / max(in_tr.sum(), 1)
    )

    in_idx = np.where(in_tr)[0]
    if len(in_idx) > 4:
        half = len(in_idx) // 2
        symmetry = float(
            abs(np.mean(flux[in_idx[:half]]) - np.mean(flux[in_idx[half:]]))
        )
    else:
        symmetry = 0.0

    if in_tr.sum() > 6:
        xi = np.linspace(-1, 1, in_tr.sum())
        coeffs = np.polyfit(xi, flux[in_tr], 2)
        ld_proxy = float(abs(coeffs[0]))
    else:
        ld_proxy = 0.0

    oot_range = (
        float(np.percentile(out_flux, 99) - np.percentile(out_flux, 1))
        if len(out_flux) > 5
        else 0.0
    )

    in_scatter = float(np.std(flux[in_tr])) if in_tr.sum() > 3 else noise
    out_scatter = float(np.std(out_flux)) if len(out_flux) > 3 else noise
    scatter_ratio = float(in_scatter / max(out_scatter, 1e-9))
    if in_tr.sum() > 3:
        in_mean = np.mean(flux[in_tr])
        in_min = np.min(flux[in_tr])
        vshape = float(abs(in_mean - in_min) / max(depth, 1e-9))
    else:
        vshape = 0.0

    fold_quality = float(np.var(flux))
    c_skew = float(skew(flux))
    kurt = float(kurtosis(flux))
    depth_snr = float(depth / max(np.std(flux), 1e-8))

    if len(in_idx) > 4:
        in_flux_pts = flux[in_idx]
        deep_thresh = np.percentile(in_flux_pts, 50)
        deep_pts = in_flux_pts[in_flux_pts <= deep_thresh]
        bottom_flatness = (
            float(np.std(deep_pts) / max(depth, 1e-9)) if len(deep_pts) > 1 else 0.0
        )
    else:
        bottom_flatness = 0.0

    s03 = flux[int(0.27 * n):int(0.33 * n)]
    s07 = flux[int(0.67 * n):int(0.73 * n)]
    sec03_ratio = float((med - np.min(s03)) / max(depth, 1e-9)) if len(s03) > 0 else 0.0
    sec07_ratio = float((med - np.min(s07)) / max(depth, 1e-9)) if len(s07) > 0 else 0.0

    hp_reg = np.concatenate([flux[:int(0.05 * n)], flux[int(0.95 * n):]])
    half_period_depth = (
        float((med - np.min(hp_reg)) / max(depth, 1e-9)) if len(hp_reg) > 0 else 0.0
    )

    if len(out_flux) > 5:
        mad_oot = float(np.median(np.abs(out_flux - np.median(out_flux))))
        snr_robust = float(depth / max(1.4826 * mad_oot, 1e-9))
    else:
        snr_robust = snr

    transit_phase_shift = float(abs(phase_arr[int(np.argmin(flux))] - 0.5))

    if len(out_flux) > 5:
        oot_med = float(np.median(out_flux))
        oot_sig = float(np.std(out_flux))
        outlier_fraction = float(np.mean(np.abs(out_flux - oot_med) > 3 * oot_sig))
    else:
        outlier_fraction = 0.0

    if len(in_idx) > 6:
        n_edge = max(2, len(in_idx) // 6)
        try:
            lslope = abs(float(np.polyfit(in_idx[:n_edge], flux[in_idx[:n_edge]], 1)[0]))
            rslope = abs(float(np.polyfit(in_idx[-n_edge:], flux[in_idx[-n_edge:]], 1)[0]))
            ingress_proxy = float((lslope + rslope) / 2 / max(depth, 1e-9))
        except Exception:
            ingress_proxy = 0.0
    else:
        ingress_proxy = 0.0

    return {
        "transit_depth":       depth,
        "transit_duration":    duration,
        "snr":                 snr,
        "even_odd_diff":       even_odd_diff,
        "centroid_skew":       c_skew,
        "kurtosis":            kurt,
        "std":                 float(np.std(flux)),
        "mean":                mean,
        "p5_p95_range":        float(np.percentile(flux, 95) - np.percentile(flux, 5)),
        "secondary_depth":     sec_depth,
        "secondary_ratio":     sec_ratio,
        "flat_fraction":       flat_frac,
        "transit_symmetry":    symmetry,
        "oot_range":           oot_range,
        "depth_snr":           depth_snr,
        "transit_count":       float(in_tr.sum()),
        "ld_proxy":            ld_proxy,
        "scatter_ratio":       scatter_ratio,
        "vshape_index":        vshape,
        "fold_quality":        fold_quality,
        "in_out_noise_diff":   float(abs(in_scatter - out_scatter)),
        "transit_fill":        float(in_tr.sum() / n),
        "bottom_flatness":     bottom_flatness,
        "sec03_ratio":         sec03_ratio,
        "sec07_ratio":         sec07_ratio,
        "half_period_depth":   half_period_depth,
        "snr_robust":          snr_robust,
        "transit_phase_shift": transit_phase_shift,
        "outlier_fraction":    outlier_fraction,
        "ingress_proxy":       ingress_proxy,
    }


def feature_matrix(curves):
    """Build (X, feat_names) from a list of preprocessed curves."""
    rows = [extract_features(curve) for curve in curves]
    feat_names = list(rows[0].keys())
    X = np.array([[row[k] for k in feat_names] for row in rows], dtype=float)
    return X, feat_names


# ─────────────────────────────────────────────
# 4. NEURAL NETWORK DEFINITIONS
# ─────────────────────────────────────────────

def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("Run: pip install torch")


class ResBlock1D(nn.Module):
    """Residual 1D block: two conv-BN-GELU stacks plus a skip connection."""

    def __init__(self, channels, k=3, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, k, padding=k // 2, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.net(x) + x)


class ViewEncoder(nn.Module):
    """Shared encoder for the 2000-point global and 200-point local views."""

    def __init__(self, out_dim=128):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(4),
        )
        self.res1 = ResBlock1D(32)
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(4),
        )
        self.res2 = ResBlock1D(64)
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.GELU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.GELU(),
            nn.Dropout(0.35),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.res1(x)
        x = self.layer2(x)
        x = self.res2(x)
        x = self.layer3(x)   # Grad-CAM forward-hook target
        return self.proj(self.gap(x))


class DualViewCNN(nn.Module):
    """Dual-view light curve classifier (inspired by AstroNet).

    Global branch: 2000-point phase-folded curve, 128-d embedding.
    Local branch:  200-point transit zoom, 64-d embedding.
    A squeeze-and-excitation style gate reweights the concatenated embedding
    before the 192 -> 64 -> 2 classification head.
    """

    def __init__(self, g_dim=128, l_dim=64):
        super().__init__()
        self.global_enc = ViewEncoder(out_dim=g_dim)
        self.local_enc = ViewEncoder(out_dim=l_dim)
        total = g_dim + l_dim
        self.attn = nn.Sequential(
            nn.Linear(total, total // 4), nn.GELU(),
            nn.Linear(total // 4, total), nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(total, 64), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(64, 2),
        )

    def forward(self, xg, xl):
        zg = self.global_enc(xg)
        zl = self.local_enc(xl)
        z = torch.cat([zg, zl], dim=1)
        z = z * self.attn(z)
        return self.head(z)


class BiLSTMNet(nn.Module):
    """Bidirectional LSTM with a soft-attention context head."""

    def __init__(self, hidden=64, layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=1, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden, 64), nn.Tanh(), nn.Linear(64, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))
        w = torch.softmax(self.attn(out).squeeze(-1), dim=1)
        ctx = (out * w.unsqueeze(-1)).sum(dim=1)
        return self.head(ctx)


class FocalLoss(nn.Module):
    """Multi-class focal loss with class weighting."""

    def __init__(self, alpha=None, gamma=FOCAL_GAMMA, smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits, target):
        log_p = F.log_softmax(logits, dim=-1)
        ce = F.nll_loss(log_p, target, reduction="none")
        p = torch.exp(-ce)
        loss = ((1.0 - p) ** self.gamma) * ce
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = alpha[target] * loss
        if self.smoothing > 0:
            loss = (1.0 - self.smoothing) * loss + self.smoothing * ce.mean()
        return loss.mean()


# ─────────────────────────────────────────────
# 5. AUGMENTATION
# ─────────────────────────────────────────────

def augment_dual(xb_g, xb_l, rng=None):
    """Gaussian noise plus a random circular shift on the global view."""
    rng = rng or np.random
    xb_g = xb_g + torch.randn_like(xb_g) * 0.003
    xb_l = xb_l + torch.randn_like(xb_l) * 0.004
    bound = int(SEQ_LEN * 0.04)
    shift = int(rng.randint(-bound, bound + 1))
    if shift != 0:
        xb_g = torch.roll(xb_g, shifts=shift, dims=-1)
    return xb_g, xb_l


def mixup_batch(xb_g, xb_l, yb, alpha=MIXUP_ALPHA, rng=None):
    """Interpolate a batch against a shuffled copy of itself."""
    rng = rng or np.random
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(xb_g.size(0), device=xb_g.device)
    mg = lam * xb_g + (1 - lam) * xb_g[idx]
    ml = lam * xb_l + (1 - lam) * xb_l[idx]
    return mg, ml, yb, yb[idx], lam


# ─────────────────────────────────────────────
# 6. THRESHOLDING
# ─────────────────────────────────────────────

def find_optimal_threshold(y_true, y_probs):
    """Youden's J statistic: maximise sensitivity + specificity - 1."""
    from sklearn.metrics import roc_curve

    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        j_idx = int(np.argmax(tpr - fpr))
        return min(float(thresholds[j_idx]), 0.5)
    except Exception:
        return 0.5


# ─────────────────────────────────────────────
# 7. DEEP MODEL TRAINING
# ─────────────────────────────────────────────

def class_weights(y):
    """Inverse-frequency class weights for Focal Loss."""
    total = len(y)
    return torch.tensor([
        total / (2.0 * max(int((y == 0).sum()), 1)),
        total / (2.0 * max(int((y == 1).sum()), 1)),
    ], dtype=torch.float32)


def train_dual_cnn(X_global, X_local, y, tr, val, class_w, epochs=CNN_EPOCHS,
                   seed=RANDOM_STATE):
    """Train the DualView CNN and return (model, metrics, threshold, history)."""
    _require_torch()
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score, roc_auc_score)

    np.random.seed(seed)
    torch.manual_seed(seed)

    g_dim = 64 if len(tr) < 30 else 128
    l_dim = 32 if len(tr) < 30 else 64

    model = DualViewCNN(g_dim=g_dim, l_dim=l_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=5e-6
    )
    criterion = FocalLoss(alpha=class_w, gamma=FOCAL_GAMMA)

    Xg_tr = torch.tensor(X_global[tr][:, None, :], dtype=torch.float32)
    Xl_tr = torch.tensor(X_local[tr][:, None, :], dtype=torch.float32)
    yt = torch.tensor(y[tr], dtype=torch.long)
    Xg_v = torch.tensor(X_global[val][:, None, :], dtype=torch.float32)
    Xl_v = torch.tensor(X_local[val][:, None, :], dtype=torch.float32)
    yv = y[val]

    loader = DataLoader(
        TensorDataset(Xg_tr, Xl_tr, yt),
        batch_size=CNN_BATCH_SIZE, shuffle=True, drop_last=False,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  DualViewCNN trainable params: {n_params:,}")

    # Initialise from the starting weights so that load_state_dict below can
    # never receive None when validation F1 never improves.
    best_f1 = 0.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    history = []
    patience = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb_g, xb_l, yb in loader:
            xb_g, xb_l, yb = xb_g.to(DEVICE), xb_l.to(DEVICE), yb.to(DEVICE)
            xb_g, xb_l = augment_dual(xb_g, xb_l)

            optimizer.zero_grad()
            if np.random.rand() > 0.5 and len(yb) > 1:
                xb_g, xb_l, ya, yb2, lam = mixup_batch(xb_g, xb_l, yb)
                logits = model(xb_g, xb_l)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb2)
            else:
                logits = model(xb_g, xb_l)
                loss = criterion(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xg_v.to(DEVICE), Xl_v.to(DEVICE))
            val_loss = F.cross_entropy(
                val_logits, torch.tensor(yv, dtype=torch.long).to(DEVICE)
            ).item()
            val_preds = val_logits.argmax(1).cpu().numpy()

        f1 = float(f1_score(yv, val_preds, zero_division=0))
        acc = float(accuracy_score(yv, val_preds))
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(epoch_loss / max(len(loader), 1), 5),
            "val_loss": round(val_loss, 5),
            "f1": round(f1, 4),
            "acc": round(acc, 4),
        })

        if f1 >= best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(
                f"  Ep {epoch + 1:3d}/{epochs}  "
                f"train_loss={epoch_loss / max(len(loader), 1):.4f}  "
                f"val_loss={val_loss:.4f}  F1={f1:.4f}  Acc={acc:.4f}"
            )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        final_logits = model(Xg_v.to(DEVICE), Xl_v.to(DEVICE))
        y_prob = torch.softmax(final_logits, dim=1)[:, 1].cpu().numpy()

    try:
        auc = (
            0.5 if len(np.unique(yv)) < 2
            else float(roc_auc_score(yv, y_prob))
        )
    except Exception:
        auc = 0.5

    threshold = find_optimal_threshold(yv, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    print(f"  CNN Youden threshold: {threshold:.3f}  (best val F1 {best_f1:.4f})")

    metrics = {
        "accuracy":  float(accuracy_score(yv, y_pred)),
        "precision": float(precision_score(yv, y_pred, zero_division=0)),
        "recall":    float(recall_score(yv, y_pred, zero_division=0)),
        "f1":        float(f1_score(yv, y_pred, zero_division=0)),
        "auc":       auc,
        "mcc":       float(matthews_corrcoef(yv, y_pred)),
    }
    return model, metrics, threshold, history


def train_bilstm(X_global, y, tr, val, class_w, epochs=CNN_EPOCHS,
                 seed=RANDOM_STATE):
    """Train the BiLSTM and return (model, metrics, threshold, history)."""
    _require_torch()
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                 precision_score, recall_score, roc_auc_score)

    np.random.seed(seed + 1)
    torch.manual_seed(seed + 1)

    model = BiLSTMNet().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=5e-6
    )
    criterion = FocalLoss(alpha=class_w, gamma=FOCAL_GAMMA)

    Xb_tr = torch.tensor(X_global[tr], dtype=torch.float32)
    Xb_v = torch.tensor(X_global[val], dtype=torch.float32)
    yb_tr = torch.tensor(y[tr], dtype=torch.long)
    yb_v = y[val]

    loader = DataLoader(
        TensorDataset(Xb_tr, yb_tr),
        batch_size=CNN_BATCH_SIZE, shuffle=True, drop_last=False,
    )
    print(f"  BiLSTM trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_f1 = 0.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    history = []
    patience = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb + torch.randn_like(xb) * 0.002
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vpreds = model(Xb_v.to(DEVICE)).argmax(1).cpu().numpy()
        f1 = float(f1_score(yb_v, vpreds, zero_division=0))
        acc = float(accuracy_score(yb_v, vpreds))
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(epoch_loss / max(len(loader), 1), 5),
            "f1": round(f1, 4),
            "acc": round(acc, 4),
        })

        if f1 >= best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(
                f"  Ep {epoch + 1:3d}/{epochs}  "
                f"train_loss={epoch_loss / max(len(loader), 1):.4f}  "
                f"F1={f1:.4f}  Acc={acc:.4f}"
            )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        y_prob = torch.softmax(model(Xb_v.to(DEVICE)), dim=1)[:, 1].cpu().numpy()

    try:
        auc = 0.5 if len(np.unique(yb_v)) < 2 else float(roc_auc_score(yb_v, y_prob))
    except Exception:
        auc = 0.5

    threshold = find_optimal_threshold(yb_v, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    print(f"  BiLSTM Youden threshold: {threshold:.3f}")

    metrics = {
        "accuracy":  float(accuracy_score(yb_v, y_pred)),
        "precision": float(precision_score(yb_v, y_pred, zero_division=0)),
        "recall":    float(recall_score(yb_v, y_pred, zero_division=0)),
        "f1":        float(f1_score(yb_v, y_pred, zero_division=0)),
        "auc":       auc,
        "mcc":       float(matthews_corrcoef(yb_v, y_pred)),
    }
    return model, metrics, threshold, history


def tta_predict_cnn(model, X_global, X_local, threshold, passes=TTA_PASSES):
    """Average CNN probabilities over `passes` noise-perturbed forward passes.

    Returns (probs, preds) aligned to the full sample axis, so caller indexing
    by sample index is always correct.
    """
    _require_torch()
    model.eval()
    Xg = torch.tensor(X_global[:, None, :], dtype=torch.float32)
    Xl = torch.tensor(X_local[:, None, :], dtype=torch.float32)

    accumulated = []
    with torch.no_grad():
        for _ in range(passes):
            logits = model(
                (Xg + torch.randn_like(Xg) * TTA_NOISE).to(DEVICE),
                (Xl + torch.randn_like(Xl) * TTA_NOISE).to(DEVICE),
            )
            accumulated.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    probs = np.mean(accumulated, axis=0)
    return probs, (probs >= threshold).astype(int)


def tta_predict_bilstm(model, X_global, threshold, passes=TTA_PASSES):
    """Average BiLSTM probabilities over repeated forward passes."""
    _require_torch()
    model.eval()
    X = torch.tensor(X_global, dtype=torch.float32)
    accumulated = []
    with torch.no_grad():
        for _ in range(passes):
            logits = model((X + torch.randn_like(X) * TTA_NOISE).to(DEVICE))
            accumulated.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    probs = np.mean(accumulated, axis=0)
    return probs, (probs >= threshold).astype(int)


# ─────────────────────────────────────────────
# 8. EXPLAINABILITY
# ─────────────────────────────────────────────

def compute_gradcam(model, global_view, local_view, target_class=1):
    """Grad-CAM over the global branch, hooked at `global_enc.layer3`.

    `global_view` is the 2000-point curve and `local_view` the 200-point zoom;
    both are required because the model is dual-view.
    """
    _require_torch()
    model.eval()
    xg = torch.tensor(global_view[None, None, :], dtype=torch.float32, device=DEVICE)
    xl = torch.tensor(local_view[None, None, :], dtype=torch.float32, device=DEVICE)

    target = model.global_enc.layer3
    captured = {}

    def fwd_hook(_module, _inputs, output):
        # Re-attach the output to the graph so autograd.grad can reach it.
        output.retain_grad()
        captured["acts"] = output

    handle = target.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(xg, xl)
        class_index = int(target_class) if target_class in (0, 1) else 1
        score = logits[0, class_index]

        acts = captured.get("acts")
        grads = None
        if acts is not None:
            grads = torch.autograd.grad(
                score, acts, retain_graph=True, allow_unused=True
            )[0]

        if grads is None:
            return np.zeros(len(global_view), dtype=np.float32)

        weights = grads.mean(dim=2, keepdim=True)                 # (1, C, 1)
        cam = F.relu((weights * acts).sum(dim=1))[0]             # (T,)
    finally:
        handle.remove()

    cam_np = cam.detach().cpu().numpy()
    cam_np = np.interp(
        np.linspace(0, 1, len(global_view)),
        np.linspace(0, 1, len(cam_np)),
        cam_np,
    )
    peak = cam_np.max()
    return (cam_np / peak).astype(np.float32) if peak > 0 else cam_np.astype(np.float32)


def compute_bilstm_attention(model, flux_array):
    """Soft-attention weights over the light curve, normalised to [0, 1]."""
    _require_torch()
    model.eval()
    x = torch.tensor(flux_array[None, :], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        out, _ = model.rnn(x.unsqueeze(-1))
        attn = torch.softmax(model.attn(out).squeeze(-1), dim=1)[0].cpu().numpy()

    att = np.interp(
        np.linspace(0, 1, len(flux_array)),
        np.linspace(0, 1, len(attn)),
        attn,
    )
    return att / att.max() if att.max() > 0 else att


def localization_in_transit(weights, window=TRANSIT_WINDOW):
    """Fraction of total weight falling inside the transit phase window."""
    n = len(weights)
    lo, hi = int(window[0] * n), int(window[1] * n)
    total = float(np.sum(weights))
    return float(np.sum(weights[lo:hi]) / total) if total > 0 else 0.0


def compute_shap(explainer, X, feature_names, top_k=8):
    """Per-sample SHAP contributions sorted by descending absolute value.

    `pos` is derived from the signed value of each contribution, so a feature
    that pushed the prediction toward the negative class is marked False.
    """
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        row = raw[1][0] if len(raw) > 1 else raw[0][0]
    else:
        row = raw[0]
    return sorted(
        [
            {"feature": name, "val": float(v), "pos": bool(float(v) >= 0)}
            for name, v in zip(feature_names, row)
        ],
        key=lambda item: -abs(item["val"]),
    )[:top_k]


def shap_explainability_metrics(explainer, X_feat, y_val, model, val):
    """Faithfulness, sparsity and consistency for the tree model.

    Faithfulness is the accuracy drop on the validation fold when the three
    highest-impact features are replaced by their dataset medians. Sparsity is
    the fraction of near-zero contributions. Consistency is the mean pairwise
    Spearman correlation of the absolute contribution profiles.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import accuracy_score

    raw = explainer.shap_values(X_feat)
    values = raw[1] if isinstance(raw, list) and len(raw) > 1 else raw
    values = np.asarray(values)

    mean_abs = np.abs(values).mean(axis=0)
    top3 = np.argsort(-mean_abs)[:3]

    if len(val) == 0:
        return {
            "shap_faithfulness_drop": None,
            "shap_sparsity":          None,
            "shap_consistency":       None,
        }, values, top3

    X_masked = X_feat[val].copy()
    for idx in top3:
        X_masked[:, idx] = np.median(X_feat[:, idx])

    acc_full = float(accuracy_score(y_val, model.predict(X_feat[val])))
    acc_masked = float(accuracy_score(y_val, model.predict(X_masked)))
    faithfulness = float(acc_full - acc_masked)
    sparsity = float((np.abs(values) < 0.01).mean())

    if len(values) >= 3:
        corrs = [
            spearmanr(np.abs(values[a]), np.abs(values[b]))[0]
            for a in range(len(values))
            for b in range(a + 1, len(values))
        ]
        finite = [c for c in corrs if not np.isnan(c)]
        consistency = float(np.mean(finite)) if finite else 1.0
    else:
        consistency = 1.0

    return {
        "shap_faithfulness_drop": round(faithfulness, 4),
        "shap_sparsity":          round(sparsity, 4),
        "shap_consistency":       round(consistency, 4),
    }, values, top3


# ─────────────────────────────────────────────
# 9. TABULAR MODELS
# ─────────────────────────────────────────────

def tune_model_rs(model, param_dist, X, y, cv, n_iter=15, seed=RANDOM_STATE):
    """RandomizedSearchCV tuning with a graceful degenerate-case fallback."""
    from sklearn.model_selection import RandomizedSearchCV

    if len(np.unique(y)) < 2 or len(y) < 6:
        model.fit(X, y)
        return model
    search = RandomizedSearchCV(
        model, param_dist, n_iter=n_iter, scoring="f1", cv=cv,
        random_state=seed, n_jobs=-1, refit=True, error_score=0.0,
    )
    search.fit(X, y)
    print(f"    RS best F1={search.best_score_:.3f}  params={search.best_params_}")
    return search.best_estimator_


def tune_xgb_optuna(X, y, cv, pos_weight, n_trials=25):
    """Bayesian (TPE) search for XGBoost."""
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBClassifier
    from sklearn.metrics import f1_score

    def objective(trial):
        params = {
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("lr", 0.005, 0.1, log=True),
            "n_estimators":     trial.suggest_int("n_estimators", 200, 600),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("col_bt", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("alpha", 0.0, 1.0),
            "reg_lambda":       trial.suggest_float("lambda", 0.5, 3.0),
            "min_child_weight": trial.suggest_int("mcw", 1, 7),
            "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
        }
        clf = XGBClassifier(
            **params, scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        if len(np.unique(y)) < 2 or len(y) < 6:
            clf.fit(X, y)
            return float(f1_score(y, clf.predict(X), zero_division=0))
        return float(np.mean(cross_val_score(clf, X, y, cv=cv, scoring="f1",
                                             error_score=0.0)))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    model = XGBClassifier(
        max_depth=best["max_depth"], learning_rate=best["lr"],
        n_estimators=best["n_estimators"], subsample=best["subsample"],
        colsample_bytree=best["col_bt"], reg_alpha=best["alpha"],
        reg_lambda=best["lambda"], min_child_weight=best["mcw"],
        gamma=best["gamma"], scale_pos_weight=pos_weight,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X, y)
    print(f"    Optuna XGB best F1={study.best_value:.3f}  params={best}")
    return model


def tune_lgbm_optuna(X, y, cv, n_trials=25):
    """Bayesian (TPE) search for LightGBM."""
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import f1_score

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 7),
            "learning_rate":     trial.suggest_float("lr", 0.005, 0.1, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("mcs", 2, 10),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("col_bt", 0.5, 1.0),
        }
        clf = LGBMClassifier(
            **params, class_weight="balanced", verbose=-1, random_state=RANDOM_STATE
        )
        if len(np.unique(y)) < 2 or len(y) < 6:
            clf.fit(X, y)
            return float(f1_score(y, clf.predict(X), zero_division=0))
        return float(np.mean(cross_val_score(clf, X, y, cv=cv, scoring="f1",
                                             error_score=0.0)))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    model = LGBMClassifier(
        n_estimators=best["n_estimators"], max_depth=best["max_depth"],
        learning_rate=best["lr"], num_leaves=best["num_leaves"],
        min_child_samples=best["mcs"], subsample=best["subsample"],
        colsample_bytree=best["col_bt"], class_weight="balanced",
        verbose=-1, random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    print(f"    Optuna LGBM best F1={study.best_value:.3f}  params={best}")
    return model


# ─────────────────────────────────────────────
# 10. FIGURES
# ─────────────────────────────────────────────

def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_to_b64(fig, dpi=80):
    """Encode a matplotlib figure as a base64 PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt_close = _pyplot()
    plt_close.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _style_axes(ax):
    ax.set_facecolor("#111827")
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_color("#1e2d47")


def make_lightcurve_figure(flux, gradcam=None, star_id=""):
    """Per-star flux plot, optionally overlaid with Grad-CAM shading."""
    plt = _pyplot()
    phase = np.linspace(0, 1, len(flux))
    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor("#0b0f1a")
    _style_axes(ax)
    ax.plot(phase, flux, color="#3b82f6", linewidth=0.9)

    if gradcam is not None:
        for i in range(len(phase) - 1):
            alpha = float(gradcam[i]) * 0.5
            if alpha > 0.05:
                ax.axvspan(phase[i], phase[i + 1], alpha=alpha,
                           color="#ef4444", linewidth=0)

    ax.set_xlabel("Orbital phase", color="#e2e8f0")
    ax.set_ylabel("Normalised flux", color="#e2e8f0")
    ax.set_title(f"Phase-folded light curve — {star_id}", color="#e2e8f0", fontsize=11)
    return fig


def make_gradcam_figure(flux, gradcam, star_id=""):
    """Standalone Grad-CAM activation profile."""
    plt = _pyplot()
    phase = np.linspace(0, 1, len(flux))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#0b0f1a")
    for ax in (ax1, ax2):
        _style_axes(ax)
    ax1.plot(phase, flux, color="#3b82f6", linewidth=0.9)
    ax1.set_ylabel("Normalised flux", color="#e2e8f0")
    ax1.set_title(f"Grad-CAM — {star_id}", color="#e2e8f0", fontsize=11)
    ax2.fill_between(phase, gradcam, color="#ef4444", alpha=0.7)
    ax2.set_xlabel("Orbital phase", color="#e2e8f0")
    ax2.set_ylabel("Activation", color="#e2e8f0")
    return fig


def make_preprocess_figure(curves, labels=None):
    """Grid of preprocessed light curves (global context image)."""
    plt = _pyplot()
    n_plot = min(6, len(curves))
    cols = min(3, n_plot)
    rows = (n_plot + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.4 * rows),
                             squeeze=False)
    fig.patch.set_facecolor("#0b0f1a")
    for idx, ax in enumerate(axes.flat):
        _style_axes(ax)
        if idx < n_plot:
            ax.plot(np.linspace(0, 1, len(curves[idx])), curves[idx],
                    color="#3b82f6", linewidth=0.8)
            if labels is not None:
                ax.set_title(labels[idx], color="#e2e8f0", fontsize=8)
        else:
            ax.axis("off")
    fig.suptitle("Preprocessed phase-folded light curves", color="#e2e8f0",
                 fontsize=11)
    return fig


def make_shap_figure(explainer, X_feat, feature_names):
    """SHAP summary plot for the tree model."""
    if not SHAP_AVAILABLE:
        return None
    plt = _pyplot()
    raw = explainer.shap_values(X_feat)
    values = raw[1] if isinstance(raw, list) and len(raw) > 1 else raw
    fig = plt.figure(figsize=(11, 7))
    fig.patch.set_facecolor("#0b0f1a")
    shap.summary_plot(values, X_feat, feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title(f"SHAP feature importance — XGBoost ({len(feature_names)} features)",
              fontsize=12, color="#e2e8f0")
    return fig


def make_training_figure(history, title):
    """Training/validation curves for a deep model."""
    plt = _pyplot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.patch.set_facecolor("#0b0f1a")
    for ax in (ax1, ax2):
        _style_axes(ax)
    epochs = [h["epoch"] for h in history]
    ax1.plot(epochs, [h["train_loss"] for h in history],
             color="#3b82f6", lw=1.5, label="Train")
    if history and "val_loss" in history[0]:
        ax1.plot(epochs, [h["val_loss"] for h in history],
                 color="#f59e0b", lw=1.5, label="Validation")
    ax1.set_xlabel("Epoch", color="#e2e8f0")
    ax1.set_ylabel("Loss", color="#e2e8f0")
    ax1.set_title("Loss", color="#e2e8f0", fontsize=11)
    ax1.legend(facecolor="#1a2235", labelcolor="#e2e8f0", fontsize=8)
    ax2.plot(epochs, [h["f1"] for h in history], color="#10b981", lw=1.5, label="F1")
    ax2.plot(epochs, [h["acc"] for h in history], color="#8b5cf6", lw=1.5, label="Acc")
    ax2.set_xlabel("Epoch", color="#e2e8f0")
    ax2.set_title("Validation score", color="#e2e8f0", fontsize=11)
    ax2.legend(facecolor="#1a2235", labelcolor="#e2e8f0", fontsize=8)
    fig.suptitle(title, color="#e2e8f0", fontsize=11)
    return fig


# ─────────────────────────────────────────────
# 11. EXPORT
# ─────────────────────────────────────────────

def _sanitize(obj):
    """Replace NaN/Inf with None so the JSON stays strictly valid."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return _sanitize(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


METRIC_KEYS = (
    "accuracy", "precision", "recall", "f1", "auc", "pr_auc", "mcc",
    "bal_acc", "cv_f1_mean", "cv_f1_std", "tp", "fp", "fn", "tn",
)


def safe_metrics(metrics):
    """Normalise a metrics dict to the full prototype contract.

    Missing entries become None rather than 0 so the prototype can render an
    em dash instead of a fabricated zero score.
    """
    if not metrics:
        return {key: None for key in METRIC_KEYS}
    out = {}
    for key in METRIC_KEYS:
        value = metrics.get(key)
        out[key] = None if value is None else float(value)
    return out


def _downsample(values, n=PROTOTYPE_POINTS, decimals=4):
    if values is None or len(values) == 0:
        return [0.0] * n
    resampled = np.interp(np.linspace(0, 1, n),
                          np.linspace(0, 1, len(values)), values)
    return [round(float(v), decimals) for v in resampled]


def build_result_dict(star_id, mission, label, cnn_prob, cnn_pred,
                      cnn_metrics, bilstm_prob, bilstm_pred, bilstm_metrics,
                      cnn_threshold, bilstm_threshold, flux, gradcam,
                      attention, explainability, shap_top8, xgb_metrics,
                      all_metrics, model_preds, feat_names, images,
                      n_train, n_val, star_params):
    """Assemble the full prototype JSON payload.

    `label` is the archive ground truth and is reported only for the
    mismatch banner. The top-level `prediction` is always the CNN's own
    decision, and an unknown archive label stays "unknown" rather than
    being collapsed into a false positive.
    """
    if label is None:
        label_text = "unknown"
    elif int(label) == 1:
        label_text = "planet"
    else:
        label_text = "false_positive"

    xgb_prob_raw = model_preds.get("XGBoost", {}).get("prob")

    return {
        "id":        star_id,
        "mission":   mission,
        "label":     label_text,
        # Model-derived decision. Never derived from the archive label.
        "prediction": "planet" if int(cnn_pred) == 1 else "false",
        "xgb":       safe_metrics(xgb_metrics),
        "cnn":       safe_metrics(cnn_metrics),
        "bilstm":    safe_metrics(bilstm_metrics),
        "models":    {name: safe_metrics(m) for name, m in all_metrics.items()},
        "model_preds": model_preds,
        "shap":      shap_top8,
        "explainability": {
            "shap_faithfulness_drop": explainability.get("shap_faithfulness_drop"),
            "shap_sparsity":          explainability.get("shap_sparsity"),
            "shap_consistency":       explainability.get("shap_consistency"),
            "gradcam_localization":   round(
                explainability.get("gradcam_localization", 0.0) or 0.0, 4
            ),
            "bilstm_localization":    round(
                explainability.get("bilstm_localization", 0.0) or 0.0, 4
            ),
        },
        "gradcam400":         _downsample(gradcam),
        "bilstmAttention400": _downsample(attention),
        "bilstmTransitFocus": round(
            explainability.get("bilstm_localization", 0.0) or 0.0, 4
        ),
        "flux400":            _downsample(flux, decimals=6),
        # null when XGBoost produced no probability - never a 0.5 stand-in.
        "xgbConf": (None if xgb_prob_raw is None
                    else round(float(xgb_prob_raw), 4)),
        "cnnConf": round(float(cnn_prob), 4),
        "bilstmConf": round(float(bilstm_prob), 4),
        "cnnThreshold":       round(float(cnn_threshold), 4),
        "bilstmThreshold":    round(float(bilstm_threshold), 4),
        "nPoints":   int(SEQ_LEN),
        "nFeatures": int(len(feat_names)),
        "featureNames": list(feat_names),
        "n_train":   int(n_train),
        "n_val":     int(n_val),
        "star_params": star_params,
        "img_flux":              images.get("img_flux"),
        "img_gradcam":           images.get("img_gradcam"),
        "img_preprocess":        images.get("img_preprocess"),
        "img_shap":              images.get("img_shap"),
        "img_training":          images.get("img_training"),
        "img_bilstm_training":   images.get("img_bilstm_training"),
    }


def write_result(payload, output_dir="results"):
    """Write one result JSON and re-parse it to confirm validity."""
    os.makedirs(output_dir, exist_ok=True)
    name = str(payload["id"]).replace("/", "_").replace(" ", "_")
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    try:
        json.loads(open(path, encoding="utf-8").read())
        status = "valid JSON"
    except Exception as exc:
        status = f"JSON ERROR: {exc}"
    return path, status


# ─────────────────────────────────────────────
# 12. SPLIT
# ─────────────────────────────────────────────

def host_group(sid):
    """Map a KOI/TIC identifier to its host star.

    K00351.01 and K00351.02 are two candidates on the same star and must land
    in the same fold, otherwise group leakage inflates every score. TIC ids are
    normalised so that "TIC 1234", "tic-1234" and "1234" share one group.
    """
    if not isinstance(sid, str):
        return str(sid)
    cleaned = sid.strip().upper().replace(" ", "").replace("_", "")
    if cleaned.startswith("TIC"):
        cleaned = cleaned[3:].lstrip("-")
        return "TIC" + cleaned if cleaned else "TIC"
    if cleaned.startswith("K") and "." in cleaned:
        return cleaned.split(".")[0]
    return cleaned


def grouped_split(groups, y, n_groups):
    """Return (train, val, test) index arrays split by host star.

    Test is a 15% held-out group-aware fold (seed 99); validation is a further
    20% of the remainder (seed 42). Datasets too small to support three folds
    fall back to a single group-aware train/val split - train and val remain
    disjoint, so reported metrics are still out-of-sample.
    """
    from sklearn.model_selection import GroupShuffleSplit

    idx_all = np.arange(len(y))
    if n_groups < 4 or len(y) < 8:
        print(
            f"  Warning: only {len(y)} samples across {n_groups} groups - "
            "using a single group-aware 80/20 split (no held-out test fold)."
        )
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION,
                                random_state=VAL_SEED)
        tr, val = next(gss.split(idx_all, y, groups))
        return tr, val, np.array([], dtype=int)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION,
                             random_state=TEST_SEED)
    tr_val, test = next(gss1.split(idx_all, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION,
                             random_state=VAL_SEED)
    tr_sub, val_sub = next(gss2.split(tr_val, y[tr_val], groups[tr_val]))
    return tr_val[tr_sub], tr_val[val_sub], test


# ─────────────────────────────────────────────
# 13. TABULAR TRAINING + EVALUATION
# ─────────────────────────────────────────────

def build_tabular_models(X_tr, y_tr, pos_weight, small_batch):
    """Tune the base learners and assemble the Stacking ensemble.

    Returns (model_configs, report) where model_configs maps a display name to
    (estimator, needs_global_scaler).
    """
    from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                                  VotingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    n_pos = max(int((y_tr == 1).sum()), 1)
    n_neg = max(int((y_tr == 0).sum()), 1)
    cv_tune = StratifiedKFold(n_splits=min(3, max(2, min(n_pos, n_neg))),
                              shuffle=True, random_state=RANDOM_STATE)
    report = {}

    xgb_base = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.02, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=3, scale_pos_weight=pos_weight,
        reg_alpha=0.2, reg_lambda=2.0, gamma=0.05, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_base = RandomForestClassifier(
        n_estimators=400, max_depth=None, max_features="sqrt", min_samples_leaf=1,
        class_weight="balanced_subsample", bootstrap=True, oob_score=True,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    et_base = ExtraTreesClassifier(
        n_estimators=400, max_depth=None, max_features="sqrt",
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
    )

    if OPTUNA_AVAILABLE and not small_batch:
        print("  Tuning XGBoost with Optuna (25 trials)...")
        xgb_tuned = tune_xgb_optuna(X_tr, y_tr, cv_tune, pos_weight, n_trials=25)
    else:
        print("  Tuning XGBoost with RandomizedSearchCV...")
        xgb_tuned = tune_model_rs(xgb_base, {
            "max_depth": [4, 5, 6], "learning_rate": [0.01, 0.02, 0.05],
            "n_estimators": [300, 400, 500], "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.7, 0.8], "reg_alpha": [0.1, 0.2, 0.5],
            "min_child_weight": [2, 3, 5],
        }, X_tr, y_tr, cv_tune)

    print("  Tuning Random Forest with RandomizedSearchCV...")
    rf_tuned = tune_model_rs(rf_base, {
        "n_estimators": [300, 400, 600], "max_depth": [None, 8, 12],
        "max_features": ["sqrt", "log2", 0.5], "min_samples_leaf": [1, 2, 3],
    }, X_tr, y_tr, cv_tune)

    print("  Tuning Extra Trees with RandomizedSearchCV...")
    et_tuned = tune_model_rs(et_base, {
        "n_estimators": [300, 400, 600], "max_features": ["sqrt", "log2", 0.5],
        "min_samples_leaf": [1, 2],
    }, X_tr, y_tr, cv_tune)

    lgbm_tuned = None
    if LGBM_AVAILABLE:
        try:
            if OPTUNA_AVAILABLE and not small_batch:
                print("  Tuning LightGBM with Optuna (25 trials)...")
                lgbm_tuned = tune_lgbm_optuna(X_tr, y_tr, cv_tune, n_trials=25)
            else:
                lgbm_base = LGBMClassifier(
                    n_estimators=400, max_depth=5, learning_rate=0.02,
                    class_weight="balanced", min_child_samples=3, verbose=-1,
                    random_state=RANDOM_STATE,
                )
                print("  Tuning LightGBM with RandomizedSearchCV...")
                lgbm_tuned = tune_model_rs(lgbm_base, {
                    "n_estimators": [300, 400, 500], "max_depth": [4, 5, 6],
                    "learning_rate": [0.01, 0.02, 0.05],
                    "num_leaves": [15, 31, 63],
                    "min_child_samples": [2, 3, 5],
                }, X_tr, y_tr, cv_tune)
        except Exception as exc:
            print(f"  LightGBM skipped: {exc}")
            lgbm_tuned = None
    else:
        print("  LightGBM not installed - skipped (pip install lightgbm to enable).")

    k_nn = max(1, min(3, len(y_tr) - 1))
    model_configs = {
        "Logistic Regression": (Pipeline([
            ("scl", RobustScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=0.5,
                                       class_weight="balanced", solver="saga",
                                       random_state=RANDOM_STATE)),
        ]), False),
        "Decision Tree": (DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE), False),
        "KNN": (Pipeline([
            ("scl", RobustScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=k_nn, weights="distance",
                                         metric="manhattan")),
        ]), False),
        "SVM": (Pipeline([
            ("scl", RobustScaler()),
            ("clf", SVC(kernel="rbf", C=3.0, gamma="scale",
                        class_weight="balanced", probability=True,
                        random_state=RANDOM_STATE)),
        ]), False),
        "Random Forest": (rf_tuned, False),
        "Extra Trees":   (et_tuned, False),
        "MLP": (Pipeline([
            ("scl", RobustScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=800,
                                  alpha=1e-3, early_stopping=True,
                                  validation_fraction=0.15,
                                  random_state=RANDOM_STATE)),
        ]), False),
        "XGBoost": (xgb_tuned, False),
    }
    if lgbm_tuned is not None:
        model_configs["LightGBM"] = (lgbm_tuned, False)

    # Stacking: the meta-learner learns how much to trust each base learner,
    # which is strictly more powerful than soft voting.
    print("  Building StackingClassifier meta-ensemble...")
    from sklearn.base import clone
    from sklearn.ensemble import StackingClassifier

    stacking_estimators = [
        ("rf", clone(rf_tuned)),
        ("et", clone(et_tuned)),
        ("xgb", clone(xgb_tuned)),
        ("svm", Pipeline([
            ("scl", RobustScaler()),
            ("clf", SVC(kernel="rbf", C=3.0, gamma="scale",
                        class_weight="balanced", probability=True,
                        random_state=RANDOM_STATE)),
        ])),
        ("lr", Pipeline([
            ("scl", RobustScaler()),
            ("clf", LogisticRegression(C=0.5, class_weight="balanced",
                                       max_iter=2000, random_state=RANDOM_STATE)),
        ])),
    ]
    if lgbm_tuned is not None:
        stacking_estimators.append(("lgbm", clone(lgbm_tuned)))

    stack_cv = StratifiedKFold(
        n_splits=min(5, max(2, min(int((y_tr == 1).sum()), int((y_tr == 0).sum())))),
        shuffle=True, random_state=RANDOM_STATE,
    )
    stacking_clf = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=3000,
                                           class_weight="balanced",
                                           solver="lbfgs",
                                           random_state=RANDOM_STATE),
        cv=stack_cv, stack_method="predict_proba", passthrough=False, n_jobs=-1,
    )
    try:
        stacking_clf.fit(X_tr, y_tr)
        model_configs["Stacking Ensemble"] = (stacking_clf, False)
        report["stacking"] = "StackingClassifier"
    except Exception as exc:
        print(f"  Stacking Ensemble failed ({exc}) - falling back to soft voting.")
        voting_estimators = [
            ("rf", clone(rf_tuned)), ("et", clone(et_tuned)),
            ("xgb", clone(xgb_tuned)),
        ]
        if lgbm_tuned is not None:
            voting_estimators.append(("lgbm", clone(lgbm_tuned)))
        voting = VotingClassifier(estimators=voting_estimators, voting="soft",
                                  n_jobs=-1)
        voting.fit(X_tr, y_tr)
        model_configs["Ensemble (Voting)"] = (voting, False)
        report["stacking"] = "VotingClassifier (fallback)"

    return model_configs, report


def evaluate_tabular_models(model_configs, X_feat, y, tr, val, groups_tr, scaler):
    """Fit every tabular model and collect validation metrics.

    The stored probability vector covers ALL samples, so a per-star export can
    index it by sample index without special-casing whether a star happened to
    land in the validation fold.
    """
    from sklearn.base import clone
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, confusion_matrix,
                                 f1_score, matthews_corrcoef, precision_score,
                                 recall_score, roc_auc_score)
    from sklearn.model_selection import GroupKFold, cross_val_score

    n_groups_tr = len(np.unique(groups_tr))
    cv_folds = min(5, max(2, n_groups_tr))
    cv = GroupKFold(n_splits=cv_folds)

    all_metrics = {}
    model_objects = {}
    model_thresholds = {}

    header = ("  {:<24} {:>8} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7} "
              "{:>9}/{:>7} {:>6}")
    print("\n" + header.format("Model", "Accuracy", "Precision", "Recall", "F1",
                              "AUC", "PR-AUC", "BalAcc", "MCC", "CV F1",
                              "std", "Thr"))
    print("  " + "-" * 126)

    for name, (clf, use_sc) in model_configs.items():
        try:
            if name == "XGBoost":
                clf.fit(X_feat[tr], y[tr],
                        eval_set=[(X_feat[val], y[val])], verbose=False)
            elif name not in ("Stacking Ensemble", "Ensemble (Voting)"):
                clf.fit(X_feat[tr], y[tr])
            # Stacking and Voting refit themselves inside .fit().
        except Exception as exc:
            print(f"  {name} fit failed: {exc}")
            continue

        X_val = scaler.transform(X_feat[val]) if use_sc else X_feat[val]
        X_all = scaler.transform(X_feat) if use_sc else X_feat

        try:
            probs_val = clf.predict_proba(X_val)[:, 1]
        except Exception:
            probs_val = clf.predict(X_val).astype(float)

        thr = find_optimal_threshold(y[val], probs_val)
        model_thresholds[name] = thr
        preds_thr = (probs_val >= thr).astype(int)

        try:
            auc = (0.5 if len(np.unique(y[val])) < 2
                   else float(roc_auc_score(y[val], probs_val)))
        except Exception:
            auc = 0.5
        try:
            pr_auc = float(average_precision_score(y[val], probs_val))
        except Exception:
            pr_auc = float(precision_score(y[val], preds_thr, zero_division=0))

        cm = confusion_matrix(y[val], preds_thr, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp_c, fn_c, tp_c = (int(cm[0, 0]), int(cm[0, 1]),
                                    int(cm[1, 0]), int(cm[1, 1]))
        else:
            tn = fp_c = fn_c = tp_c = None

        try:
            cv_scores = cross_val_score(clone(clf), X_feat[tr], y[tr], cv=cv,
                                        groups=groups_tr, scoring="f1",
                                        error_score=0.0)
            cv_mean, cv_std = float(np.mean(cv_scores)), float(np.std(cv_scores))
        except Exception:
            # An unavailable CV score is reported as null, never silently
            # replaced by the single-split F1.
            cv_mean, cv_std = None, None

        metrics = {
            "accuracy":   float(accuracy_score(y[val], preds_thr)),
            "precision":  float(precision_score(y[val], preds_thr, zero_division=0)),
            "recall":     float(recall_score(y[val], preds_thr, zero_division=0)),
            "f1":         float(f1_score(y[val], preds_thr, zero_division=0)),
            "auc":        auc,
            "pr_auc":     pr_auc,
            "mcc":        float(matthews_corrcoef(y[val], preds_thr)),
            "bal_acc":    float(balanced_accuracy_score(y[val], preds_thr)),
            "cv_f1_mean": cv_mean,
            "cv_f1_std":  cv_std,
            "tp": tp_c, "fp": fp_c, "fn": fn_c, "tn": tn,
        }
        all_metrics[name] = metrics

        try:
            probs_all = clf.predict_proba(X_all)[:, 1]
        except Exception:
            probs_all = clf.predict(X_all).astype(float)
        model_objects[name] = (clf, probs_all, use_sc)

        cv_str = f"{cv_mean:.4f}" if cv_mean is not None else "n/a"
        std_str = f"{cv_std:.4f}" if cv_std is not None else "n/a"
        print("  " + header.format(
            name, f"{metrics['accuracy']:.4f}", f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}", f"{metrics['f1']:.4f}", f"{auc:.4f}",
            f"{pr_auc:.4f}", f"{metrics['bal_acc']:.4f}",
            f"{metrics['mcc']:.3f}", cv_str, std_str, f"{thr:.3f}",
        ))

    return all_metrics, model_objects, model_thresholds


def evaluate_held_out_test(model_objects, X_feat, y, test, scaler,
                           model_thresholds):
    """Evaluate every fitted model on the locked held-out test fold.

    Decisions use the thresholds tuned on validation, so the test fold is
    never used to select a threshold.
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, precision_score, recall_score)

    if len(test) == 0:
        print("\n  No held-out test set - dataset too small for a 3-way split.")
        return {}

    results = {}
    print("\n  Held-out test evaluation (validation thresholds, never re-tuned):")
    for name, (clf, _, use_sc) in model_objects.items():
        try:
            X_test = scaler.transform(X_feat[test]) if use_sc else X_feat[test]
            probs = clf.predict_proba(X_test)[:, 1]
            thr = model_thresholds.get(name, 0.5)
            preds = (probs >= thr).astype(int)
            results[name] = {
                "accuracy":  float(accuracy_score(y[test], preds)),
                "precision": float(precision_score(y[test], preds, zero_division=0)),
                "recall":    float(recall_score(y[test], preds, zero_division=0)),
                "f1":        float(f1_score(y[test], preds, zero_division=0)),
                "bal_acc":   float(balanced_accuracy_score(y[test], preds)),
                "threshold": float(thr),
            }
            r = results[name]
            print(f"    {name:<24} Acc={r['accuracy']:.4f}  F1={r['f1']:.4f}  "
                  f"(thr={thr:.3f})")
        except Exception as exc:
            print(f"    {name}: test evaluation failed - {exc}")
    return results


# ─────────────────────────────────────────────
# 14. FULL PIPELINE
# ─────────────────────────────────────────────

def run_full_pipeline(ids, source="kepler", csv_dir=None,
                      models_dir="saved_models", results_dir="results",
                      epochs=CNN_EPOCHS):
    """Train every model on the given targets and export one JSON per star.

    Returns the list of result payloads written to `results_dir`.
    """
    from sklearn.preprocessing import RobustScaler

    _require_torch()
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    koi_df, toi_df = (None, None)
    if source in ("kepler", "tess"):
        koi_df, toi_df = archive_tables()

    curves, labels, star_ids, missions, skipped = [], [], [], [], []

    print("=== STEP 1: Data collection ===")
    for sid in ids:
        print(f"  [{sid}]")
        try:
            if source == "kepler":
                _, flux, _, label = download_kepler(sid, koi_df)
                mission = "Kepler"
            elif source == "tess":
                _, flux, _, label = download_tess(sid, toi_df)
                mission = "TESS"
            else:
                _, flux, _, label = load_csv(
                    resolve_csv_path(csv_dir or ".", sid)
                )
                mission = "Local"

            print(f"    label={label}")
            if label not in ("planet", "false_positive"):
                # An unlabelled target has no supervised ground truth. Keeping
                # it as class 0 would quietly invent a false positive.
                skipped.append((sid, label))
                print("    SKIPPED: archive label is not determinate - "
                      "cannot be used for supervised training or honest "
                      "accuracy reporting.")
                continue

            curves.append(preprocess(flux))
            labels.append(1 if label == "planet" else 0)
            star_ids.append(sid)
            missions.append(mission)
        except Exception as exc:
            print(f"    ERROR: {exc}")

    for sid, label in skipped:
        print(f"  note: {sid} excluded (label={label})")

    if len(curves) < 2:
        print("\n  Need at least 2 usable samples. Exiting.")
        return []

    X_global = np.array(curves, dtype=np.float32)
    X_local = np.array([extract_local_view(c) for c in curves], dtype=np.float32)
    X_feat, feat_names = feature_matrix(curves)
    y = np.array(labels, dtype=int)
    groups = np.array([host_group(s) for s in star_ids])

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Only class {unique_classes} found - need at least one confirmed "
            "planet AND one false positive."
        )

    n_pos = max(int(y.sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    pos_weight = float(n_neg) / float(n_pos)
    print(f"\n  Samples: {len(y)}  planets: {n_pos}  non-planets: {n_neg}  "
          f"host stars: {len(np.unique(groups))}  pos_weight={pos_weight:.2f}")

    tr, val, test = grouped_split(groups, y, len(np.unique(groups)))
    small_batch = len(y) < 8
    print(f"  Train: {len(tr)}  Val: {len(val)}  "
          f"Test: {len(test) if len(test) else 'none (held out)'}")
    if len(test):
        print(f"  Test host stars: {np.unique(groups[test]).tolist()}")

    scaler = RobustScaler()
    scaler.fit(X_feat[tr])

    X_tr_aug, y_tr_aug = X_feat[tr].copy(), y[tr].copy()
    smote_applied = False
    if SMOTE_AVAILABLE and not small_batch:
        n_min = min(int(y[tr].sum()), int((y[tr] == 0).sum()))
        if n_min >= 2:
            try:
                k_sm = max(1, min(n_min - 1, 3))
                sampler = SMOTE(k_neighbors=k_sm, random_state=RANDOM_STATE)
                X_tr_aug, y_tr_aug = sampler.fit_resample(X_feat[tr], y[tr])
                smote_applied = True
                print(f"  SMOTE: {len(y[tr])} -> {len(y_tr_aug)} training samples")
            except Exception as exc:
                print(f"  SMOTE skipped: {exc}")

    print("\n=== STEP 2: Tabular models ===")
    model_configs, report = build_tabular_models(
        X_tr_aug, y_tr_aug, pos_weight, small_batch
    )
    all_metrics, model_objects, model_thresholds = evaluate_tabular_models(
        model_configs, X_feat, y, tr, val, groups[tr], scaler
    )
    test_results = evaluate_held_out_test(model_objects, X_feat, y, test, scaler,
                                          model_thresholds)

    xgb_model = model_objects["XGBoost"][0]

    print("\n=== STEP 3: DualView CNN ===")
    class_w = class_weights(y)
    cnn, cnn_metrics, cnn_threshold, cnn_history = train_dual_cnn(
        X_global, X_local, y, tr, val, class_w, epochs=epochs
    )
    print("  CNN metrics:", {k: round(v, 4) for k, v in cnn_metrics.items()})

    print("\n=== STEP 4: BiLSTM ===")
    bilstm, bilstm_metrics, bilstm_threshold, bilstm_history = train_bilstm(
        X_global, y, tr, val, class_w, epochs=epochs
    )
    print("  BiLSTM metrics:", {k: round(v, 4) for k, v in bilstm_metrics.items()})

    print(f"\n=== STEP 5: All-star inference with {TTA_PASSES}-pass TTA ===")
    cnn_probs, cnn_preds = tta_predict_cnn(cnn, X_global, X_local, cnn_threshold)
    bilstm_probs, bilstm_preds = tta_predict_bilstm(bilstm, X_global, bilstm_threshold)
    print(f"  CNN predicted {int(cnn_preds.sum())} planets / {len(cnn_preds)} stars")
    print(f"  BiLSTM predicted {int(bilstm_preds.sum())} planets / {len(bilstm_preds)} stars")

    print("\n=== STEP 6: Explainability ===")
    explainer = None
    explainability = {"shap_faithfulness_drop": None, "shap_sparsity": None,
                      "shap_consistency": None}
    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(xgb_model)
        explainability, shap_values, top3 = shap_explainability_metrics(
            explainer, X_feat, y[val], xgb_model, val
        )
        print(f"  Faithfulness={explainability['shap_faithfulness_drop']}  "
              f"Sparsity={explainability['shap_sparsity']}  "
              f"Consistency={explainability['shap_consistency']}")
        print(f"  Top-3 features: {[feat_names[i] for i in top3]}")
    else:
        print("  SHAP not installed - skipping tree explainability.")

    print("\n=== STEP 7: Save artefacts ===")
    torch.save({
        "state_dict": cnn.state_dict(),
        "g_dim": 64 if len(tr) < 30 else 128,
        "l_dim": 32 if len(tr) < 30 else 64,
        "threshold": cnn_threshold,
        "feature_names": feat_names,
    }, os.path.join(models_dir, "cnn_dualview.pt"))
    torch.save({
        "state_dict": bilstm.state_dict(),
        "threshold": bilstm_threshold,
        "feature_names": feat_names,
    }, os.path.join(models_dir, "bilstm.pt"))
    xgb_model.save_model(os.path.join(models_dir, "xgboost.json"))

    manifest = {
        "feature_names": list(feat_names),
        "cnn_threshold": float(cnn_threshold),
        "bilstm_threshold": float(bilstm_threshold),
        "model_thresholds": {k: float(v) for k, v in model_thresholds.items()},
        "metrics": _sanitize(all_metrics),
        "cnn_metrics": _sanitize(cnn_metrics),
        "bilstm_metrics": _sanitize(bilstm_metrics),
        "explainability": _sanitize(explainability),
        "n_train": int(len(tr)), "n_val": int(len(val)),
        "n_test": int(len(test)), "n_samples": int(len(y)),
        "n_features": int(len(feat_names)),
        "seq_len": int(SEQ_LEN), "local_len": int(LOCAL_LEN),
        "smote_applied": bool(smote_applied),
        "test_results": _sanitize(test_results),
        "stacking": report.get("stacking"),
    }
    with open(os.path.join(models_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print("\n=== STEP 8: Figures ===")
    images = {}
    fig = make_preprocess_figure(
        curves[:6],
        [f"{s} ({'planet' if l else 'FP'})"
         for s, l in zip(star_ids[:6], labels[:6])],
    )
    images["img_preprocess"] = fig_to_b64(fig, dpi=70)
    if explainer is not None:
        fig = make_shap_figure(explainer, X_feat, feat_names)
        if fig is not None:
            images["img_shap"] = fig_to_b64(fig, dpi=70)
    images["img_training"] = fig_to_b64(
        make_training_figure(cnn_history, "DualView CNN training"), dpi=70
    )
    images["img_bilstm_training"] = fig_to_b64(
        make_training_figure(bilstm_history, "BiLSTM training"), dpi=70
    )

    print("\n=== STEP 9: Per-star export ===")
    results = []
    for i, sid in enumerate(star_ids):
        flux = curves[i]
        # Grad-CAM is computed for the class the model actually predicted.
        gradcam = compute_gradcam(cnn, flux, X_local[i],
                                  target_class=int(cnn_preds[i]))
        attention = compute_bilstm_attention(bilstm, flux)

        shap_top8 = []
        if explainer is not None:
            shap_top8 = compute_shap(explainer, X_feat[i:i + 1], feat_names)

        model_preds = {}
        for name, (_clf, probs_all, _use_sc) in model_objects.items():
            model_preds[name] = {
                "prob": round(float(probs_all[i]), 4),
                "pred": ("planet"
                         if probs_all[i] >= model_thresholds.get(name, 0.5)
                         else "false"),
            }
        model_preds["DualView CNN"] = {
            "prob": round(float(cnn_probs[i]), 4),
            "pred": "planet" if cnn_preds[i] == 1 else "false",
        }
        model_preds["BiLSTM"] = {
            "prob": round(float(bilstm_probs[i]), 4),
            "pred": "planet" if bilstm_preds[i] == 1 else "false",
        }

        star_explain = dict(explainability)
        star_explain["gradcam_localization"] = localization_in_transit(gradcam)
        star_explain["bilstm_localization"] = localization_in_transit(attention)

        star_images = dict(images)
        star_images["img_flux"] = fig_to_b64(
            make_lightcurve_figure(flux, gradcam, sid), dpi=70
        )
        star_images["img_gradcam"] = fig_to_b64(
            make_gradcam_figure(flux, gradcam, sid), dpi=70
        )

        payload = build_result_dict(
            star_id=sid,
            mission=missions[i],
            label=int(y[i]),
            cnn_prob=float(cnn_probs[i]),
            cnn_pred=int(cnn_preds[i]),
            cnn_metrics=cnn_metrics,
            bilstm_prob=float(bilstm_probs[i]),
            bilstm_pred=int(bilstm_preds[i]),
            bilstm_metrics=bilstm_metrics,
            cnn_threshold=cnn_threshold,
            bilstm_threshold=bilstm_threshold,
            flux=flux,
            gradcam=gradcam,
            attention=attention,
            explainability=star_explain,
            shap_top8=shap_top8,
            xgb_metrics=all_metrics.get("XGBoost"),
            all_metrics=all_metrics,
            model_preds=model_preds,
            feat_names=feat_names,
            images=star_images,
            n_train=len(tr),
            n_val=len(val),
            star_params=star_params_for(sid, missions[i], koi_df, toi_df),
        )
        payload = _sanitize(payload)

        path, status = write_result(payload, results_dir)
        truth = "planet" if y[i] == 1 else "fp"
        model_says = "planet" if cnn_preds[i] == 1 else "fp"
        flag = "  MISMATCH" if truth != model_says else ""
        print(f"  {sid:<14} truth={truth:<7} cnn={model_says:<7} "
              f"conf={cnn_probs[i]:.3f}  {status}{flag}")

        results.append(payload)

    print("\n=== SUMMARY ===")
    for name, metrics in all_metrics.items():
        print(f"  {name:<24} Acc={metrics['accuracy']:.3f}  "
              f"F1={metrics['f1']:.3f}  AUC={metrics['auc']:.3f}")
    print(f"  {'DualView CNN':<24} Acc={cnn_metrics['accuracy']:.3f}  "
          f"F1={cnn_metrics['f1']:.3f}  AUC={cnn_metrics['auc']:.3f}")
    print(f"  {'BiLSTM':<24} Acc={bilstm_metrics['accuracy']:.3f}  "
          f"F1={bilstm_metrics['f1']:.3f}  AUC={bilstm_metrics['auc']:.3f}")
    print(f"\n  Results -> {results_dir}/")
    print(f"  Models  -> {models_dir}/")
    return results


# ─────────────────────────────────────────────
# 15. SINGLE-STAR INFERENCE
# ─────────────────────────────────────────────

def run_inference_single(flux, star_id="star", mission="Local", label="unknown",
                         models_dir="saved_models", output_dir="results"):
    """Score one pre-folded light curve using previously trained artefacts.

    Validation metrics are read from `models_dir/manifest.json` - the same
    numbers reported during training. Nothing is fabricated: if the manifest is
    absent the metrics block is emitted as nulls and the prototype renders an
    em dash.
    """
    _require_torch()

    flux_proc = preprocess(flux)
    local = extract_local_view(flux_proc)
    features = extract_features(flux_proc)
    feat_names = list(features.keys())
    X_feat = np.array([list(features.values())], dtype=float)

    cnn_path = os.path.join(models_dir, "cnn_dualview.pt")
    bilstm_path = os.path.join(models_dir, "bilstm.pt")
    xgb_path = os.path.join(models_dir, "xgboost.json")
    manifest_path = os.path.join(models_dir, "manifest.json")

    for path in (cnn_path, bilstm_path, xgb_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {os.path.basename(path)} in '{models_dir}'. "
                "Run run_full_pipeline() (or --mode batch) first."
            )

    cnn_blob = torch.load(cnn_path, map_location=DEVICE, weights_only=False)
    if list(feat_names) != list(cnn_blob.get("feature_names", feat_names)):
        raise ValueError(
            "Feature set of the loaded models does not match extract_features(). "
            "Retrain so the 30-feature contract stays aligned."
        )

    cnn = DualViewCNN(g_dim=cnn_blob.get("g_dim", 128),
                      l_dim=cnn_blob.get("l_dim", 64))
    cnn.load_state_dict(cnn_blob["state_dict"])
    cnn.to(DEVICE).eval()
    cnn_threshold = float(cnn_blob.get("threshold", 0.5))

    bilstm_blob = torch.load(bilstm_path, map_location=DEVICE, weights_only=False)
    bilstm = BiLSTMNet()
    bilstm.load_state_dict(bilstm_blob["state_dict"])
    bilstm.to(DEVICE).eval()
    bilstm_threshold = float(bilstm_blob.get("threshold", 0.5))

    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.load_model(xgb_path)

    def _tta_probs(model, views):
        """Average the class-1 probability over the same TTA passes as training."""
        accumulated = []
        with torch.no_grad():
            for _ in range(TTA_PASSES):
                noisy = [v + torch.randn_like(v) * TTA_NOISE for v in views]
                logits = model(*[v.to(DEVICE) for v in noisy])
                accumulated.append(
                    torch.softmax(logits, dim=1)[0, 1].cpu().item()
                )
        return float(np.mean(accumulated))

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)

    # Reuse the thresholds tuned on validation during training rather than
    # re-deciding at 0.5 for a single star.
    saved_thresholds = manifest.get("model_thresholds") or {}
    xgb_threshold = float(saved_thresholds.get("XGBoost", 0.5))
    cnn_threshold = float(manifest.get("cnn_threshold", cnn_threshold))
    bilstm_threshold = float(manifest.get("bilstm_threshold", bilstm_threshold))

    xg = torch.tensor(flux_proc[None, None, :], dtype=torch.float32)
    xl = torch.tensor(local[None, None, :], dtype=torch.float32)
    cnn_prob = _tta_probs(cnn, (xg, xl))
    cnn_pred = int(cnn_prob >= cnn_threshold)

    xb = torch.tensor(flux_proc[None, :], dtype=torch.float32)
    bilstm_prob = _tta_probs(bilstm, (xb,))
    bilstm_pred = int(bilstm_prob >= bilstm_threshold)

    gradcam = compute_gradcam(cnn, flux_proc, local, target_class=cnn_pred)
    attention = compute_bilstm_attention(bilstm, flux_proc)

    try:
        xgb_prob = float(xgb.predict_proba(X_feat)[0, 1])
    except Exception:
        xgb_prob = None

    shap_top8 = []
    if SHAP_AVAILABLE:
        shap_top8 = compute_shap(shap.TreeExplainer(xgb), X_feat, feat_names)

    model_preds = {
        "XGBoost": {
            "prob": None if xgb_prob is None else round(xgb_prob, 4),
            "pred": ("unknown" if xgb_prob is None
                     else ("planet" if xgb_prob >= xgb_threshold else "false")),
        },
        "DualView CNN": {"prob": round(cnn_prob, 4),
                         "pred": "planet" if cnn_pred else "false"},
        "BiLSTM": {"prob": round(bilstm_prob, 4),
                   "pred": "planet" if bilstm_pred else "false"},
    }

    cnn_metrics = manifest.get("cnn_metrics")
    bilstm_metrics = manifest.get("bilstm_metrics")
    all_metrics = manifest.get("metrics") or {}
    xgb_metrics = all_metrics.get("XGBoost")
    explainability = dict(manifest.get("explainability") or {})
    explainability["gradcam_localization"] = localization_in_transit(gradcam)
    explainability["bilstm_localization"] = localization_in_transit(attention)

    payload = build_result_dict(
        star_id=star_id, mission=mission,
        label=1 if label == "planet" else (0 if label == "false_positive" else None),
        cnn_prob=cnn_prob, cnn_pred=cnn_pred, cnn_metrics=cnn_metrics,
        bilstm_prob=bilstm_prob, bilstm_pred=bilstm_pred,
        bilstm_metrics=bilstm_metrics,
        cnn_threshold=cnn_threshold, bilstm_threshold=bilstm_threshold,
        flux=flux_proc, gradcam=gradcam, attention=attention,
        explainability=explainability, shap_top8=shap_top8,
        xgb_metrics=xgb_metrics, all_metrics=all_metrics,
        model_preds=model_preds, feat_names=feat_names,
        images={
            "img_flux": fig_to_b64(
                make_lightcurve_figure(flux_proc, gradcam, star_id), dpi=70),
            "img_gradcam": fig_to_b64(
                make_gradcam_figure(flux_proc, gradcam, star_id), dpi=70),
        },
        n_train=manifest.get("n_train", 0),
        n_val=manifest.get("n_val", 0),
        star_params={},
    )
    payload = _sanitize(payload)
    path, status = write_result(payload, output_dir)
    print(f"  {star_id}: CNN={cnn_prob:.3f} (thr {cnn_threshold:.3f})  "
          f"BiLSTM={bilstm_prob:.3f}  -> {path}  {status}")
    if not cnn_metrics:
        print("  Note: no manifest.json found, so validation metrics are null.")
    return payload


# ─────────────────────────────────────────────
# 16. CLI
# ─────────────────────────────────────────────

DEMO_IDS = ["K00001.01", "K00007.01", "K00017.01", "K00022.01",
            "K00041.01", "K00069.01", "K00070.01", "K00072.01"]


def main():
    parser = argparse.ArgumentParser(
        description="Exoplanet transit detection pipeline "
                    "(mirrors exoplanet_colab.ipynb)"
    )
    parser.add_argument("--mode",
                        choices=["kepler", "tess", "csv", "batch", "infer"],
                        default="batch")
    parser.add_argument("--id", help="Single KOI or TIC ID")
    parser.add_argument("--ids", help="Comma-separated list of IDs")
    parser.add_argument("--file", help="Path to a CSV file")
    parser.add_argument("--csv-dir",
                        help="Directory containing CSV files for batch mode")
    parser.add_argument("--models", default="saved_models",
                        help="Model directory")
    parser.add_argument("--out", default="results", help="Results directory")
    parser.add_argument("--epochs", type=int, default=CNN_EPOCHS,
                        help="Epochs for the CNN and BiLSTM")
    args = parser.parse_args()

    if args.mode in ("kepler", "tess"):
        if not args.id:
            parser.error("--id is required for kepler/tess mode")
        run_full_pipeline([args.id], source=args.mode, models_dir=args.models,
                          results_dir=args.out, epochs=args.epochs)

    elif args.mode == "batch":
        if args.ids:
            ids = [i.strip() for i in args.ids.split(",") if i.strip()]
            source = "tess" if ids[0].replace("TIC-", "").isdigit() else "kepler"
        else:
            print(f"No --ids given. Using demo batch: {DEMO_IDS}")
            ids, source = DEMO_IDS, "kepler"
        run_full_pipeline(ids, source=source, csv_dir=args.csv_dir,
                          models_dir=args.models, results_dir=args.out,
                          epochs=args.epochs)

    elif args.mode == "csv":
        if not args.file:
            parser.error("--file is required for csv mode")
        run_full_pipeline([os.path.basename(args.file)], source="csv",
                          csv_dir=os.path.dirname(os.path.abspath(args.file)),
                          models_dir=args.models, results_dir=args.out,
                          epochs=args.epochs)

    elif args.mode == "infer":
        if not args.file:
            parser.error("--file is required for infer mode")
        _, flux, _, _ = load_csv(args.file)
        name = os.path.splitext(os.path.basename(args.file))[0]
        run_inference_single(flux, star_id=name, models_dir=args.models,
                             output_dir=args.out)


if __name__ == "__main__":
    main()
