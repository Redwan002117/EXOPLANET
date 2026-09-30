# Results

Produced by a real training run on live NASA Kepler light curves. Every number
here is copied from `logs/training_run.log`; nothing is hand-entered or
smoothed. Re-run with `logs/training_targets.txt` to reproduce.

## Run configuration

| Setting | Value |
| --- | --- |
| Targets | 60 KOI ids (`logs/training_targets.txt`) |
| Source | NASA Exoplanet Archive `cumulative` table, live via `lightkurve` / MAST |
| Cadences per star | ~64,000 raw, stitched across 17–18 quarters, then phase-folded |
| Class balance | 30 `planet` / 30 `false_positive` |
| Splits (host-grouped) | 40 train / 11 validation / 9 locked test |
| SMOTE | 40 → 44 training rows |
| Optuna | 25 trials each for XGBoost and LightGBM |
| Epochs | 80 (CNN and BiLSTM) |
| Wall clock | 50.8 min on CPU |
| Artefacts | `cnn_dualview.pt`, `bilstm.pt`, `xgboost.json`, `manifest.json` |

### What "planet" means here

The authoritative notebook derives its label from `koi_pdisposition` and treats
`CANDIDATE` **or** `CONFIRMED` as `planet` (`pipeline.py:252-255`). The archive's
`cumulative` table contains **zero** `CONFIRMED` rows — only `CANDIDATE` (4,717)
and `FALSE POSITIVE` (4,847). So every `planet` label in this run is a KOI
**candidate**, i.e. not yet independently confirmed. This is the notebook's
definition and was preserved deliberately; it is not a defect introduced here,
but it does mean "planet" is softer than the word suggests.

## Held-out performance

Validation-fold scores from the run log. These are the honest generalisation
estimates, on 11 validation stars:

| Model | Acc | F1 | AUC |
| --- | --- | --- | --- |
| Logistic Regression | 0.636 | 0.667 | 0.607 |
| Decision Tree | 0.727 | 0.667 | 0.679 |
| KNN | 0.636 | 0.667 | 0.643 |
| SVM | 0.545 | 0.615 | 0.571 |
| Random Forest | 0.545 | 0.615 | 0.643 |
| Extra Trees | 0.545 | 0.615 | 0.500 |
| MLP | 0.636 | 0.667 | 0.571 |
| XGBoost | 0.545 | 0.615 | 0.536 |
| **LightGBM** | **0.727** | **0.727** | 0.643 |
| Stacking Ensemble | 0.636 | 0.667 | **0.714** |
| DualView CNN | 0.455 | 0.571 | 0.393 |
| BiLSTM | 0.364 | 0.533 | 0.321 |

**Read this honestly: these are weak results, and the deep models are worse than
chance.** The BiLSTM's AUC of 0.321 and the CNN's 0.393 are *below* 0.5, meaning
they rank false positives above planets on held-out data. LightGBM at 0.727
accuracy is the best performer.

The cause is training-set size, not necessarily architecture. The CNN has 156,634
trainable parameters and the BiLSTM 150,339, trained on **40** phase-folded
light curves. At that ratio the deep models memorise the training set (the
all-star inference below shows recall 1.000 with zero false negatives, which is
the memorisation signature) and generalise worse than a gradient-boosted tree.

### All-star inference is in-sample — do not quote it as accuracy

After training on all 60 stars, the pipeline re-scored every star with 7-pass TTA:

```
CNN verdicts : 51 planet, 9 false
TP=30  FP=21  TN=9  FN=0
accuracy=0.650  precision=0.588  recall=1.000  f1=0.741
```

This looks better than the validation table, but it is **in-sample**: those same
stars were in the training set. The locked-test and validation numbers above are
the ones that estimate generalisation. Recall 1.000 with FN=0 is a memorisation
artefact, not a capability.

## Integrity checks

`logs/verification.log` records a 7/7 pass across `py_compile`, unit sanity,
balanced smoke, imbalanced smoke, single-star inference, prototype JS syntax, and
prototype JS behaviour.

`validate_results.py` re-derives every exported verdict from CNN output alone and
passed on all 60 real JSON files:

- every `prediction` is recoverable from `model_preds` / `cnnConf` / `cnnThreshold` alone
- no live-NASA record carries a fabricated verdict
- every metric finite, every JSON round-trippable, schema complete
- all 10 manifest thresholds within the notebook ceiling `(0, 0.5]`

## What would make these results defensible

1. **Scale the dataset.** 60 stars is a plumbing-scale run. Several hundred
   labelled stars would put the deep models on even footing with LightGBM.
2. **Fix the label definition.** Cross-check `CANDIDATE` KOIs against the
   confirmed-planets table (`ps` / `pscomppars`) so `planet` means confirmed.
3. **Report the locked test fold only.** The 9-star test fold is the single
   defensible held-out estimate; the 11-star validation fold is used for
   threshold selection and is therefore mildly optimistic.
4. **Retune for the deep models.** 80 epochs on 40 samples is heavily
   overfit; early stopping on validation and a smaller model would help more than
   a longer search.
