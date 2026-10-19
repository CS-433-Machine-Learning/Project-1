"""Generate the AIcrowd submission for Project 1 (heart disease prediction).

Pipeline: load data -> clean features -> two models, averaged:
  - regularized logistic regression (Newton's method) on standardized features with
    squared and interaction terms
  - gradient boosted trees (boosting.py) on the cleaned raw features
-> pick the decision threshold by 5-fold cross-validation -> predict test set.

Usage: python run.py  (expects the dataset in data/dataset/, writes submission.csv)
"""

import itertools
import os

import numpy as np

from boosting import apply_bins, fit_bins, predict_boosting, train_boosting
from helpers import create_csv_submission, load_csv_data
from implementations import (
    reg_logistic_regression,
    reg_logistic_regression_newton,
    sigmoid,
)

DATA_DIR = "data/dataset"
CACHE_FILE = "data/dataset_cache.npz"
OUTPUT_FILE = "submission.csv"

SEED = 42
K_FOLDS = 5
NAN_THRESHOLD = 0.9  # drop columns with more than 90% missing values
LAMBDA = 1e-4
MAX_ITERS = 500  # gradient descent (train)
NEWTON_ITERS = 10  # Newton's method (train_newton), used for the final model

# Gradient boosted trees, averaged 50/50 with the logistic regression: the trees
# alone reach a cross-validated F1 of 0.4347, the average 0.4364 (logistic: 0.4326)
BOOSTING_SETTINGS = dict(n_trees=250, learning_rate=0.1, max_depth=4)
BOOSTING_WEIGHT = 0.5

# Survey design columns (dates, record ids, sampling weights): they describe the
# interview, not the person. WEIGHT2/HEIGHT3 mix pounds and kilograms in one column
# their cleaned versions (WTKG3, HTM4, _BMI5) are kept instead
DROPPED_COLUMNS = [
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "QSTVER",
    "QSTLANG",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CLLCPWT",
    "_DUALCOR",
    "_LLCPWT",
    "WEIGHT2",
    "HEIGHT3",
]
# Target leakage: "outpatient rehabilitation after your heart attack" is only asked
# to people who had a heart attack (100% of the respondents are positive). It would
# give the label away, so it is never used, even by the models that keep sparse
# columns.
LEAKY_COLUMNS = ["HAREHAB1"]


# Data loading


def load_data():
    """Load the dataset, using a local cache since parsing the CSVs takes ~3 minutes.

    Returns:
        x_train, x_test, y_train, test_ids, column names
    """
    if os.path.exists(CACHE_FILE):
        d = np.load(CACHE_FILE)
        return d["x_train"], d["x_test"], d["y_train"], d["test_ids"], list(d["names"])

    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_DIR)
    with open(os.path.join(DATA_DIR, "x_train.csv")) as f:
        names = f.readline().strip().split(",")[1:]  # first column is the id
    np.savez(
        CACHE_FILE,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        test_ids=test_ids,
        names=np.array(names),
    )
    return x_train, x_test, y_train, test_ids, names


# Feature processing


def replace_special_codes(x):
    """Turn the BRFSS "don't know / refused" codes into NaN and "none" codes into 0.

    The codes depend on the answer width, which we read from the column maximum:
    1 digit -> 7, 9; 2 digits -> 77, 99 (88 = none); 3 digits -> 777, 999 (888 = none);
    4 digits -> 7777, 9999. For 1-digit columns that also use 8 (e.g. EMPLOY1),
    7 is a real answer and is kept.

    Args:
        x: shape=(N, D)

    Returns:
        a copy of x with the codes replaced
    """
    x = x.copy()
    for j in range(x.shape[1]):
        col = x[:, j]  # view: edits apply to x
        if np.all(np.isnan(col)):
            continue
        top = np.nanmax(col)
        if top == 9:
            col[col == 9] = np.nan
            if not np.any(col == 8):
                col[col == 7] = np.nan
        elif top == 99:
            col[(col == 77) | (col == 99)] = np.nan
            col[col == 88] = 0
        elif top == 999:
            col[(col == 777) | (col == 999)] = np.nan
            col[col == 888] = 0
        elif top == 9999:
            col[(col == 7777) | (col == 9999)] = np.nan
    return x


def fit_preprocessing(x, names):
    """Learn the preprocessing parameters on the training data only.

    Args:
        x: shape=(N, D), raw features
        names: list of the D column names

    Returns:
        dict with the kept columns, the medians used to fill NaN, and the mean/std
    """
    keep = np.array([name not in DROPPED_COLUMNS for name in names])
    x = replace_special_codes(x)
    keep &= np.isnan(x).mean(axis=0) <= NAN_THRESHOLD
    x = x[:, keep]
    median = np.nanmedian(x, axis=0)
    x = np.where(np.isnan(x), median, x)
    mean, std = x.mean(axis=0), x.std(axis=0)
    nonconstant = std > 0
    return {
        "keep": keep,
        "median": median,
        "mean": mean,
        "std": std,
        "nonconstant": nonconstant,
    }


def transform(x, params):
    """Apply the preprocessing learned by fit_preprocessing and add a bias column.

    Args:
        x: shape=(N, D), raw features
        params: output of fit_preprocessing

    Returns:
        tx: shape=(N, D' + 1), standardized features with a leading column of ones
    """
    x = replace_special_codes(x)[:, params["keep"]]
    x = np.where(np.isnan(x), params["median"], x)
    nc = params["nonconstant"]
    x = (x[:, nc] - params["mean"][nc]) / params["std"][nc]
    return np.c_[np.ones(len(x)), x]


# Non-linear features: squares of the non-binary features, and pairwise products of
# classic cardiovascular risk factors (logistic regression alone is linear in x)
RISK_FACTORS = [
    "_AGE80",  # age
    "SEX",
    "_BMI5",  # body mass index
    "GENHLTH",  # self-reported general health
    "_RFHYPE5",  # high blood pressure
    "_RFCHOL",  # high cholesterol
    "DIABETE3",
    "_SMOKER3",
    "CVDSTRK3",  # ever had a stroke
    "DIFFWALK",  # difficulty walking
]
# 10 more health and lifestyle factors: with all 20, cross-validated F1 goes from
# 0.4312 to 0.4326 (190 products instead of 45)
EXTRA_RISK_FACTORS = [
    "CHCKIDNY",  # kidney disease
    "CHCCOPD1",  # chronic obstructive pulmonary disease
    "HAVARTH3",  # arthritis
    "ADDEPEV2",  # depression
    "_TOTINDA",  # no physical activity
    "DRNKANY5",  # drinks alcohol
    "INCOME2",
    "EDUCA",
    "PHYSHLTH",  # days of poor physical health
    "QLACTLM2",  # activity limited by health problems
]
CLIP = 5  # standardized values are clipped to +-5 before squaring, against outliers


def _expand(base_tx, params):
    """Append the squared (and cubed) and interaction columns to the baseline features."""
    columns = [base_tx]
    clipped = np.clip(base_tx[:, params["non_binary"]], -CLIP, CLIP)
    if params["squares"]:
        columns.append(clipped**2)
    if params["cubes"]:
        columns.append(clipped**3)
    if params["pairs"]:
        columns.append(
            np.stack([base_tx[:, i] * base_tx[:, j] for i, j in params["pairs"]], 1)
        )
    return np.concatenate(columns, axis=1)


def fit_poly(
    x,
    names,
    squares=True,
    interactions=True,
    factors=RISK_FACTORS,
    age_with_all=False,
    cubes=False,
):
    """Learn the baseline preprocessing, then the stats of the added columns.

    Args:
        x: shape=(N, D), raw features
        names: list of the D column names
        squares, interactions, cubes: which kinds of columns to add
        factors: column names whose pairwise products are added
        age_with_all: also add age x every other feature

    Returns:
        dict with the baseline parameters, the added columns and their mean/std
    """
    base = fit_preprocessing(x, names)
    base_tx = transform(x, base)[:, 1:]  # standardized, without the bias
    kept = list(np.array(names)[base["keep"]][base["nonconstant"]])
    pairs = []
    if interactions:
        pairs = list(itertools.combinations([kept.index(n) for n in factors], 2))
    if age_with_all:
        age = kept.index("_AGE80")
        done = {frozenset(p) for p in pairs}
        pairs += [
            (age, j)
            for j in range(len(kept))
            if j != age and frozenset((age, j)) not in done
        ]
    params = {
        "base": base,
        "squares": squares,
        "cubes": cubes,
        "non_binary": np.array([len(np.unique(c)) > 2 for c in base_tx.T]),
        "pairs": pairs,
    }
    expanded = _expand(base_tx, params)
    params["mean"], params["std"] = expanded.mean(axis=0), expanded.std(axis=0)
    return params


def transform_poly(x, params):
    """Apply fit_poly: baseline features + added columns, standardized, with bias.

    Returns float32 to halve the memory (~500 columns, the Mac has 8 GB of RAM).
    """
    expanded = _expand(transform(x, params["base"])[:, 1:], params)
    nc = params["std"] > 0
    expanded = (expanded[:, nc] - params["mean"][nc]) / params["std"][nc]
    return np.c_[np.ones(len(x)), expanded].astype(np.float32)


def fit_final(x, names):
    """Features of the final model: squares + products of all 20 risk factors."""
    return fit_poly(x, names, factors=RISK_FACTORS + EXTRA_RISK_FACTORS)


# Training and evaluation


def train(y01, tx):
    """Regularized logistic regression with step size 1/L.

    L = 0.25 * largest eigenvalue of tx.T tx / N + 2 * lambda is the Lipschitz
    constant of the gradient, so gamma = 1/L guarantees a decreasing loss.

    Args:
        y01: shape=(N, ), labels in {0, 1}
        tx: shape=(N, D), float64 or float32

    Returns:
        w: shape=(D, ), same dtype as tx
    """
    # y and w use the dtype of tx, so float32 inputs are never copied to float64
    y01 = y01.astype(tx.dtype)
    gram = (tx.T @ tx / len(y01)).astype(np.float64)
    lipschitz = 0.25 * np.linalg.eigvalsh(gram).max() + 2 * LAMBDA
    w, loss = reg_logistic_regression(
        y01, tx, LAMBDA, np.zeros(tx.shape[1], dtype=tx.dtype), MAX_ITERS, 1 / lipschitz
    )
    print(f"  training loss: {loss:.4f}")
    return w


def train_newton(y01, tx, lambda_=LAMBDA, balanced=False):
    """Regularized logistic regression with Newton's method.

    Same objective as train(). With the ~360 correlated features of fit_poly,
    500 gradient steps do not reach the minimum, while 10 Newton steps do
    (cross-validated F1 0.4295 -> 0.4312).

    Args:
        y01: shape=(N, ), labels in {0, 1}
        tx: shape=(N, D)
        lambda_: regularization parameter
        balanced: weight the samples so that each class counts for half of the loss

    Returns:
        w: shape=(D, )
    """
    tx = tx.astype(np.float64)  # the Newton step is solved in float64
    weights = None
    if balanced:
        pos = np.mean(y01 == 1)
        weights = np.where(y01 == 1, 0.5 / pos, 0.5 / (1 - pos))
    w, loss = reg_logistic_regression_newton(
        y01, tx, lambda_, np.zeros(tx.shape[1]), NEWTON_ITERS, weights=weights
    )
    print(f"  training loss: {loss:.4f}")
    return w


def f1_score(y01, pred01):
    """F1 score of binary predictions, with the positive class 1."""
    tp = np.sum((pred01 == 1) & (y01 == 1))
    precision = tp / max(np.sum(pred01 == 1), 1)
    recall = tp / max(np.sum(y01 == 1), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def best_threshold(y01, proba):
    """Pick the probability threshold that maximizes F1.

    Only ~9% of people are sick, so the usual 0.5 threshold almost never predicts 1.

    Returns:
        (threshold, F1 at that threshold)
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y01, (proba >= t).astype(int)) for t in thresholds]
    i = int(np.argmax(scores))
    return thresholds[i], scores[i]


def kfold_indices(n, k, seed):
    """Split the indices 0..n-1 into k random folds of (almost) equal size."""
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, k)


def cross_validate(x, y01, names, fit_fn, transform_fn, folds, train_fn=train):
    """Out-of-fold scores of one preprocessing variant.

    Each fold is predicted by a model trained on the other folds. The threshold is
    tuned once on all these out-of-fold predictions, then scored fold by fold.

    Args:
        x: shape=(N, D), raw features
        y01: shape=(N, ), labels in {0, 1}
        names: list of the D column names
        fit_fn, transform_fn: preprocessing functions (e.g. fit_preprocessing, transform)
        folds: list of index arrays, from kfold_indices
        train_fn: train (gradient descent) or train_newton

    Returns:
        threshold: tuned on all out-of-fold predictions
        f1s, accuracies: one value per fold, at that threshold
    """
    proba = np.empty(len(y01))
    for k, valid in enumerate(folds):
        tr = np.concatenate([f for i, f in enumerate(folds) if i != k])
        params = fit_fn(x[tr], names)
        w = train_fn(y01[tr], transform_fn(x[tr], params))
        proba[valid] = sigmoid(transform_fn(x[valid], params) @ w)
    return score_folds(y01, proba, folds)


def score_folds(y01, proba, folds):
    """Tune the threshold on all out-of-fold predictions, then score each fold.

    Args:
        y01: shape=(N, ), labels in {0, 1}
        proba: shape=(N, ), out-of-fold predicted scores
        folds: list of index arrays, from kfold_indices

    Returns:
        threshold, and the per-fold F1 scores and accuracies at that threshold
    """
    threshold, _ = best_threshold(y01, proba)
    f1s = [f1_score(y01[v], (proba[v] >= threshold).astype(int)) for v in folds]
    accuracies = [np.mean((proba[v] >= threshold) == y01[v]) for v in folds]
    return threshold, np.array(f1s), np.array(accuracies)


# Main


def fit_tree_features(x, names):
    """Columns and quantile bins of the boosted trees, learned on training data.

    Unlike the linear model, the trees keep the mostly empty columns: many are
    questions of optional modules asked only to a sub-group (e.g. diabetics), so
    whether they were answered is informative. Missing values get their own bin,
    so no imputation is needed.
    """
    clean = replace_special_codes(x)
    dropped = DROPPED_COLUMNS + LEAKY_COLUMNS
    keep = np.array([name not in dropped for name in names])
    keep &= ~np.all(np.isnan(clean), axis=0)
    return {"keep": keep, "edges": fit_bins(clean[:, keep])}


def transform_tree_features(x, params):
    """Binned features of fit_tree_features, shape (D, N)."""
    return apply_bins(replace_special_codes(x)[:, params["keep"]], params["edges"])


def fit_model(x, y01, names):
    """Train the final model: logistic regression (Newton) on the 505 features of
    fit_final, and gradient boosted trees on the binned raw features.

    Args:
        x: shape=(N, D), raw features
        y01: shape=(N, ), labels in {0, 1}
        names: list of the D column names

    Returns:
        dict with everything predict_model needs
    """
    poly = fit_final(x, names)
    w = train_newton(y01, transform_poly(x, poly))
    trees = fit_tree_features(x, names)
    booster = train_boosting(
        y01, transform_tree_features(x, trees), seed=SEED, **BOOSTING_SETTINGS
    )
    return {"poly": poly, "w": w, "trees": trees, "booster": booster}


def predict_model(model, x):
    """Probability of class 1: average of the logistic regression and the trees."""
    tx = transform_poly(x, model["poly"]).astype(np.float64)
    p_logistic = sigmoid(tx @ model["w"])
    p_trees = predict_boosting(
        model["booster"], transform_tree_features(x, model["trees"])
    )
    return BOOSTING_WEIGHT * p_trees + (1 - BOOSTING_WEIGHT) * p_logistic


def main():
    print("Loading data...")
    x_train, x_test, y_train, test_ids, names = load_data()
    y01 = (y_train + 1) / 2  # {-1, 1} -> {0, 1} for logistic regression

    # 1) Cross-validation: choose the threshold and estimate the score on held-out data
    print(f"{K_FOLDS}-fold cross-validation...")
    folds = kfold_indices(len(y01), K_FOLDS, SEED)
    proba = np.empty(len(y01))
    for k, valid in enumerate(folds):
        print(f"  fold {k + 1}/{K_FOLDS}")
        tr = np.concatenate([f for i, f in enumerate(folds) if i != k])
        model = fit_model(x_train[tr], y01[tr], names)
        proba[valid] = predict_model(model, x_train[valid])
    threshold, f1s, accs = score_folds(y01, proba, folds)
    print(f"  F1 = {f1s.mean():.4f} +- {f1s.std():.4f}")
    print(f"  accuracy = {accs.mean():.4f} +- {accs.std():.4f}")
    print(f"  threshold = {threshold:.2f}")

    # 2) Final model on all training data, same hyperparameters and threshold
    print("Final training on all data...")
    write_submission(x_train, y01, x_test, test_ids, names, threshold)


def write_submission(x_train, y01, x_test, test_ids, names, threshold):
    """Train on all the training data and write the predictions for the test set."""
    model = fit_model(x_train, y01, names)
    pred01 = predict_model(model, x_test) >= threshold
    y_pred = np.where(pred01, 1, -1)  # back to {-1, 1} for the submission
    print(f"  {np.mean(y_pred == 1):.1%} of the test set predicted sick")
    create_csv_submission(test_ids, y_pred, OUTPUT_FILE)
    print(f"Submission written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
