"""Generate the AIcrowd submission for Project 1 (heart disease prediction).

Pipeline: load data -> clean features -> standardize -> regularized logistic
regression -> pick the decision threshold on a validation split -> predict test set.

Usage: python run.py  (expects the dataset in data/dataset/, writes submission.csv)
"""

import os

import numpy as np

from helpers import create_csv_submission, load_csv_data
from implementations import reg_logistic_regression, sigmoid

DATA_DIR = "data/dataset"
CACHE_FILE = "data/dataset_cache.npz"
OUTPUT_FILE = "submission.csv"

SEED = 42
VALID_FRACTION = 0.2
NAN_THRESHOLD = 0.9  # drop columns with more than 90% missing values
LAMBDA = 1e-4
MAX_ITERS = 500

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



# Training and evaluation



def train(y01, tx):
    """Regularized logistic regression with step size 1/L.

    L = 0.25 * largest eigenvalue of tx.T tx / N + 2 * lambda is the Lipschitz
    constant of the gradient, so gamma = 1/L guarantees a decreasing loss.

    Args:
        y01: shape=(N, ), labels in {0, 1}
        tx: shape=(N, D)

    Returns:
        w: shape=(D, )
    """
    lipschitz = 0.25 * np.linalg.eigvalsh(tx.T @ tx / len(y01)).max() + 2 * LAMBDA
    w, loss = reg_logistic_regression(
        y01, tx, LAMBDA, np.zeros(tx.shape[1]), MAX_ITERS, 1 / lipschitz
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



# Main


def main():
    print("Loading data...")
    x_train, x_test, y_train, test_ids, names = load_data()
    y01 = (y_train + 1) / 2  # {-1, 1} -> {0, 1} for logistic regression

    # 1) Validation: choose the threshold and estimate the score on held-out data
    print("Validation run...")
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y01))
    n_valid = int(VALID_FRACTION * len(y01))
    valid, tr = idx[:n_valid], idx[n_valid:]

    params = fit_preprocessing(x_train[tr], names)
    w = train(y01[tr], transform(x_train[tr], params))
    proba = sigmoid(transform(x_train[valid], params) @ w)
    threshold, f1 = best_threshold(y01[valid], proba)
    accuracy = np.mean((proba >= threshold) == y01[valid])
    print(f"  validation: F1 = {f1:.4f}, accuracy = {accuracy:.4f}")
    print(f"  threshold = {threshold:.2f}")

    # 2) Final model on all training data, same hyperparameters and threshold
    print("Final training on all data...")
    params = fit_preprocessing(x_train, names)
    print(f"  {params['nonconstant'].sum()} features kept out of {len(names)}")
    w = train(y01, transform(x_train, params))
    pred01 = sigmoid(transform(x_test, params) @ w) >= threshold
    y_pred = np.where(pred01, 1, -1)  # back to {-1, 1} for the submission
    print(f"  {np.mean(y_pred == 1):.1%} of the test set predicted sick")

    create_csv_submission(test_ids, y_pred, OUTPUT_FILE)
    print(f"Submission written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
