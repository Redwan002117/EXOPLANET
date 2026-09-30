# EXOPLANET — Exoplanet Transit Detection

Machine-learning pipeline and interactive web prototype for detecting transiting
exoplanets in Kepler, TESS, and arbitrary light-curve data.

The reference implementation is `exoplanet_colab.ipynb`. `pipeline.py` is a
standalone, production-shaped port of that notebook, and `exoplanet_prototype.html`
is a browser UI that renders the JSON produced by `pipeline.py`.

## Layout

| Path | Purpose |
| --- | --- |
| `exoplanet_colab.ipynb` | Authoritative methodology. Kept unmodified. |
| `pipeline.py` | Standalone training, evaluation, export, and inference CLI. |
| `exoplanet_prototype.html` | Single-file prototype UI. Reads result JSON. |
| `server.py` | Local static server + NASA TAP proxy (avoids browser CORS). |
| `requirements.txt` | Python dependencies. |
| `results/` | Per-star result JSON. **Tracked in git.** |
| `saved_models/` | Trained checkpoints and manifest. **Git-ignored.** |

Thesis write-ups are the `.docx` and `.pdf` files in the repository root.

## Model decision logic

A star's displayed prediction is derived from model output, never from its archive
label. The resolution order is:

1. `model_preds["DualView CNN"]` — the CNN's own argmax
2. `cnnConf` compared against `cnnThreshold` (validation-tuned Youden threshold,
   capped at 0.5)
3. otherwise `unknown`

The legacy top-level `prediction` field is retained in exports for compatibility
but is deliberately **not** used by the UI. Stars whose archive label is
`unknown` are excluded from supervised training and are never scored TP/TN/FP/FN.
Missing metrics and confidences are exported as `null` and render as `—`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`imbalanced-learn`, `lightgbm`, and `optuna` are optional — the pipeline degrades
gracefully and falls back when they are absent. With all three installed the
following branches are live and have been exercised end to end:

| Branch | Behaviour |
| --- | --- |
| SMOTE (`imbalanced-learn`) | Oversamples the minority class of the training fold. Verified to rebalance a 3/13 fold to 13/13 (16 → 26 rows). A no-op when the fold is already balanced, which is correct — SMOTE's default strategy only lifts the minority *to* the majority. |
| Optuna (XGBoost) | 25-trial TPE search replaces `RandomizedSearchCV`. |
| Optuna (LightGBM) | 25-trial TPE search for the LightGBM model. |
| LightGBM | Added to the model pool and to the stacking/voting ensemble. |

Optuna and SMOTE are skipped when fewer than 8 labelled samples are available
(`small_batch`), because cross-validation folds become degenerate.

## Usage

```powershell
# single star by identifier
python pipeline.py --mode kepler --id KOI-1234
python pipeline.py --mode tess  --id 1234

# a directory of light curves
python pipeline.py --mode csv --csv-dir .\data

# train on the notebook's bundled dataset and export per-star JSON
python pipeline.py --mode batch --csv-dir .\data --out .\results

# re-score a star using previously saved artefacts
python pipeline.py --mode infer --file .\data\koi-1234.csv --models .\saved_models
```

`--mode csv` and `--mode infer` perform **no** NASA archive network calls.

## Running the prototype

```powershell
python server.py
```

Then open the printed local URL. The prototype looks for star JSON in `results/`.
Use the live NASA lookup only for inspecting an archive record — that path runs no
model and reports its prediction and confidence as unknown.

## Known caveats

- `Thesis_Report.pdf` and `Technical_Documentation.pdf` are **stale**: they predate
  the corrections that made predictions model-derived and are out of sync with the
  current `.docx` files. Regenerate them before submission. The `.docx` files are current.
- No trained models are committed. `results/` and `saved_models/` are empty in a fresh
  clone; populate them by running a real training cycle.
- Any metrics in a tracked `results/` JSON that came from a short smoke-test run
  (very few stars, few epochs) are plumbing checks only and are not an accuracy claim.
  A very small validation fold can legitimately produce AUC 0.0000 and MCC -1.000;
  an *undefined* AUC is reported as 0.5 (chance), never as 0.0.
