# Predicting Coronary Heart Disease from BRFSS Survey Data

CS-433 Machine Learning (EPFL), Project 1. Team: Timo Fischer, Herman Kristiansen,
Alexandre Pestel.

We predict whether a person has coronary heart disease (MICHD) from their answers
to the BRFSS health survey, with models implemented from scratch in NumPy. The
final model averages a regularized logistic regression on non-linear features with
gradient boosted decision trees, and its decision threshold is tuned for F1 by
5-fold cross-validation.

| | F1 | Accuracy |
|---|---|---|
| 5-fold cross-validation | 0.4364 ± 0.0010 | 0.874 |
| AIcrowd test set | **0.447** | 0.877 |

## Quick start

**Requirements.** Python 3 and NumPy. Matplotlib is only needed to draw the
learning-curve figure.

**Data.** Download `dataset_to_release.zip` from the course repository
(`projects/project1/data`) and unzip it into `data/dataset/`:

```
data/dataset/x_train.csv
data/dataset/y_train.csv
data/dataset/x_test.csv
```

**Run.**

```
python run.py
```

This writes `submission.csv`, identical to our best AIcrowd submission.

- **Time:** about 50 minutes, because the cross-validation that picks the
  threshold trains the boosted trees five times.
- **Memory:** up to about 5 GB of RAM.
- **Determinism:** every random choice is seeded.
- **Cache:** parsing the CSV files takes about 3 minutes, so the first run saves
  the arrays to `data/dataset_cache.npz` (about 1 GB) and later runs load it.

## How `run.py` works

All statistics (medians, means, bins) are computed on training data only.

1. **Cleaning.** Drop the columns that describe the interview (dates, record ids,
   sampling weights) and the raw height and weight (mixed units). Turn the BRFSS
   "don't know / refused" codes (7/9, 77/99, ...) into missing values. Exclude
   `HAREHAB1` (rehabilitation after a heart attack, only asked to people who had
   one) as target leakage.
2. **Logistic regression.**
   - Base features: drop the columns with more than 90% missing values, fill the
     missing values with the median and standardize, which leaves 198 features.
   - Added features: the squares of the 117 non-binary features (clipped to ±5)
     and the 190 pairwise products of 20 risk factors, for 505 features in total.
   - Training: L2 penalty λ = 1e-4, 10 Newton steps.
3. **Gradient boosted trees.**
   - Input: all remaining columns, including the mostly empty ones (questions of
     optional modules asked only to sub-groups, e.g. diabetics).
   - Binning: each column is cut into 64 quantile bins, plus one bin for missing
     values.
   - Training: 250 trees of depth 4, learning rate 0.1, 70% of the samples per
     tree.
4. **Average and threshold.** The prediction is 0.5 × logistic + 0.5 × trees. The
   threshold (0.21) maximizes F1 on the out-of-fold predictions of the 5-fold
   cross-validation. Both models are then retrained on all the data.

## Files

| File | Content |
|---|---|
| `run.py` | Full pipeline that produces the submission |
| `implementations.py` | The 6 required methods and their helpers, plus Newton's method for regularized logistic regression and a linear SVM |
| `boosting.py` | Histogram-based gradient boosted decision trees |
| `mlp.py` | Small neural network (one hidden layer, Adam), used in the experiments only |
| `cross_validation.py` | Experiments of the report |
| `helpers.py` | Course helpers to load the data and write the submission |
| `ml_methods/` | First version of the 6 methods, one file per method |

## Experiments

All experiments use the same 5 folds as `run.py` (seed 42). For each variant they
report the mean ± std of F1 over folds and the paired F1 difference with a
reference variant.

| Command | Content | Time |
|---|---|---|
| `python cross_validation.py models` | The 6 methods and the SVM on the baseline features | ~20 min |
| `python cross_validation.py features` | Logistic regression (gradient descent): one-hot, squares, interactions | ~1 h |
| `python cross_validation.py newton` | Logistic regression (Newton): age × all, cubes, 20 risk factors | ~30 min |
| `python cross_validation.py tuning` | Regularization strength and class weights | ~25 min |
| `python cross_validation.py ensemble` | Averages with small neural networks | ~20 min |
| `python cross_validation.py boosting` | Boosted trees, alone and averaged with the logistic regression | ~1 h |
| `python cross_validation.py indicators` | Module indicators for the logistic regression | ~1 h |
| `python cross_validation.py boosting_variants` | Lower learning rate and column subsampling for the trees | ~1 h |
| `python cross_validation.py counts` | Counts of chronic conditions and limitations | ~1 h |
| `python cross_validation.py learning_curve` | Learning curve, saved to `report/learning_curve.pdf` | ~10 min |

- **Dependencies:** `boosting_variants` and `counts` reuse the predictions saved
  by `indicators` (in `data/oof_predictions.npz`), so run `indicators` first.
- **Selecting variants:** pass their names, e.g.
  `python cross_validation.py features squares` or
  `python cross_validation.py models "SVM (balanced)"`. The reference row always
  runs.
- **Times** were measured on a laptop with 8 GB of RAM.
