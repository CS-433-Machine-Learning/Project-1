"""5-fold cross-validation experiments for the report (baselines and ablation study).

Experiments, all on the same 5 folds as run.py:
- models: the 6 methods of implementations.py, plus a linear SVM, on the baseline
  features
- features: preprocessing variants (baseline, one-hot, squares, interactions) with
  regularized logistic regression trained by gradient descent
- newton: richer interactions and cubes on top of the 10-factor model, trained with
  Newton's method
- tuning: regularization strength and class weights of the final model (Newton)
- ensemble: averages of the final logistic regression and small neural networks
- boosting: gradient boosted trees, alone and averaged with the logistic regression
- indicators: sub-group module indicators for the logistic regression
- boosting_variants: smaller step / column subsampling for the trees (run after
  indicators, whose saved predictions it reuses)
- counts: global health features (numbers of conditions and limitations)
- learning_curve: training/validation loss vs training set size (bias-variance),
  saved as a figure for the report

The decision threshold is tuned once on the pooled out-of-fold predictions, then F1
and accuracy are reported per fold as mean +- std. Since the folds are shared, each
row is also compared to a reference fold by fold, which tells whether a difference
is real or just noise.

Usage: python cross_validation.py [models|features|newton|tuning [variant ...]|ensemble|boosting|indicators|boosting_variants|counts|learning_curve]
(models takes ~10 minutes; one-hot alone ~25 minutes, the other variants ~10 each)
"""

import os
import sys
import time
from functools import partial

import numpy as np

import boosting as gb
import implementations as imp
import mlp
import run

# One-hot variant: columns with at most MAX_CATEGORIES distinct values are treated
# as categorical; answers given by less than MIN_FREQUENCY of people are ignored
MAX_CATEGORIES = 10
MIN_FREQUENCY = 0.001


def fit_onehot(x, names):
    """Learn the one-hot encoding on the training data only.

    Categorical columns get one 0/1 column per frequent answer ("don't know" and
    "refused" codes included, so they keep their own weight). Continuous columns
    are cleaned as in the baseline. Columns with missing values also get a 0/1
    "is missing" indicator.

    Args:
        x: shape=(N, D), raw features
        names: list of the D column names

    Returns:
        dict with the list of output columns and their mean/std
    """
    keep = np.array([name not in run.DROPPED_COLUMNS for name in names])
    cleaned = run.replace_special_codes(x)
    keep &= np.isnan(cleaned).mean(axis=0) <= run.NAN_THRESHOLD

    specs = []  # (kind, column index, value)
    for j in np.where(keep)[0]:
        col = x[:, j]
        values, counts = np.unique(col[~np.isnan(col)], return_counts=True)
        if len(values) <= MAX_CATEGORIES:
            specs += [
                ("equals", j, v)
                for v, c in zip(values, counts)
                if c >= MIN_FREQUENCY * len(col)
            ]
        else:
            specs.append(("number", j, np.nanmedian(cleaned[:, j])))
        if np.isnan(col).mean() >= MIN_FREQUENCY:
            specs.append(("missing", j, None))

    params = {"specs": specs}
    encoded = _encode(x, specs)
    params["mean"], params["std"] = encoded.mean(axis=0), encoded.std(axis=0)
    return params


def _encode(x, specs):
    """Build the one-hot matrix (float32 to fit in memory: ~700 columns)."""
    cleaned = run.replace_special_codes(x)
    out = np.empty((len(x), len(specs)), dtype=np.float32)
    for k, (kind, j, value) in enumerate(specs):
        if kind == "equals":
            out[:, k] = x[:, j] == value
        elif kind == "missing":
            out[:, k] = np.isnan(x[:, j])
        else:  # "number": median-filled cleaned value
            out[:, k] = np.where(np.isnan(cleaned[:, j]), value, cleaned[:, j])
    return out


def transform_onehot(x, params):
    """Apply the encoding learned by fit_onehot, standardize, and add a bias column."""
    nc = params["std"] > 0
    encoded = _encode(x, params["specs"])[:, nc]
    encoded = (encoded - params["mean"][nc]) / params["std"][nc]
    return np.c_[np.ones(len(x), dtype=np.float32), encoded]


# "squares+inter" is the variant used by run.py
VARIANTS = {
    "baseline": (run.fit_preprocessing, run.transform),
    "one-hot": (fit_onehot, transform_onehot),
    "squares": (partial(run.fit_poly, interactions=False), run.transform_poly),
    "interactions": (partial(run.fit_poly, squares=False), run.transform_poly),
    "squares+inter": (run.fit_poly, run.transform_poly),
}

# Richer interactions and cubes, on top of the 10-factor model (trained with Newton)
NEWTON_VARIANTS = {
    "10 factors": (run.fit_poly, run.transform_poly),
    "age x all": (partial(run.fit_poly, age_with_all=True), run.transform_poly),
    "20 factors": (run.fit_final, run.transform_poly),  # what run.py uses
    # polynomial degree 3 (lecture 4a: the degree is a hyperparameter)
    "cubes": (partial(run.fit_poly, cubes=True), run.transform_poly),
    "20 factors + cubes": (
        partial(
            run.fit_poly, factors=run.RISK_FACTORS + run.EXTRA_RISK_FACTORS, cubes=True
        ),
        run.transform_poly,
    ),
}


# The 6 methods of implementations.py, on the baseline features

# Some people have ||x||^2 ~ 8e4 after standardization: a larger SGD step diverges
SGD_GAMMA = 1e-5


def _largest_eigenvalue(tx):
    """Largest eigenvalue of tx.T tx / N, the curvature of the MSE loss."""
    return np.linalg.eigvalsh(tx.T @ tx / len(tx)).max()


def _least_squares(y01, tx):
    return imp.least_squares(y01, tx)[0]


def _ridge(y01, tx):
    return imp.ridge_regression(y01, tx, run.LAMBDA)[0]


def _mse_gd(y01, tx):
    gamma = 1 / _largest_eigenvalue(tx)
    return imp.mean_squared_error_gd(
        y01, tx, np.zeros(tx.shape[1]), run.MAX_ITERS, gamma
    )[0]


def _mse_sgd(y01, tx):
    np.random.seed(run.SEED)  # batch_iter draws the samples with np.random
    # one pass over the data: as many steps as training samples
    return imp.mean_squared_error_sgd(
        y01, tx, np.zeros(tx.shape[1]), len(y01), SGD_GAMMA
    )[0]


def _logistic(y01, tx):
    gamma = 1 / (0.25 * _largest_eigenvalue(tx))
    return imp.logistic_regression(
        y01, tx, np.zeros(tx.shape[1]), run.MAX_ITERS, gamma
    )[0]


# Initial step of the subgradient method (decays as 1/sqrt(t)); best of 0.1, 1, 10
# on one fold, for both SVM variants
SVM_GAMMA = 1.0


def _svm(y01, tx, balanced=False):
    y_pm = 2 * y01 - 1  # the SVM uses labels in {-1, 1}
    return imp.svm_subgradient_descent(
        y_pm,
        tx,
        run.LAMBDA,
        np.zeros(tx.shape[1]),
        run.MAX_ITERS,
        SVM_GAMMA,
        balanced,
    )[0]


def _identity(z):
    return z


# name -> (train(y01, tx) -> w, score function applied to tx @ w).
# The regression models fit y in {0, 1} directly, so tx @ w is already a score
# that can be thresholded like a probability. The SVM margin x^T w is mapped to
# (0, 1) by the sigmoid, which keeps the order, so the same threshold grid applies.
MODELS = {
    "least squares": (_least_squares, _identity),
    "ridge": (_ridge, _identity),
    "MSE GD": (_mse_gd, _identity),
    "MSE SGD": (_mse_sgd, _identity),
    "logistic": (_logistic, imp.sigmoid),
    "reg. logistic": (run.train, imp.sigmoid),
    "SVM (hinge)": (_svm, imp.sigmoid),
    "SVM (balanced)": (partial(_svm, balanced=True), imp.sigmoid),
}


def compare_models(x, y01, names, folds, selected):
    """Cross-validate the methods. The preprocessing is fitted once per fold
    and shared by all models, so they are compared on exactly the same data.

    Args:
        selected: names of the MODELS to run (all of them if empty); the
            reference "reg. logistic" always runs
    """
    models = {
        name: fns
        for name, fns in MODELS.items()
        if not selected or name in selected or name == "reg. logistic"
    }
    scores = {name: np.empty(len(y01)) for name in models}
    for k, valid in enumerate(folds):
        print(f"fold {k + 1}/{len(folds)}...")
        tr = np.concatenate([f for i, f in enumerate(folds) if i != k])
        params = run.fit_preprocessing(x[tr], names)
        tx_tr, tx_valid = run.transform(x[tr], params), run.transform(x[valid], params)
        for name, (train_fn, score_fn) in models.items():
            w = train_fn(y01[tr], tx_tr)
            scores[name][valid] = score_fn(tx_valid @ w)
    return {name: run.score_folds(y01, s, folds) for name, s in scores.items()}


# Training variants of the final model (20 factors, Newton): regularization strength
# and class weights. The features are built once per fold and shared.
TRAINING_VARIANTS = {
    "lambda 1e-4": run.train_newton,  # what run.py uses
    "lambda 1e-3": partial(run.train_newton, lambda_=1e-3),
    "lambda 1e-2": partial(run.train_newton, lambda_=1e-2),
    "balanced": partial(run.train_newton, balanced=True),
}


def compare_training(x, y01, names, folds, selected):
    """Cross-validate TRAINING_VARIANTS on the features of run.py.

    Args:
        selected: names of the variants to run (all of them if empty); the
            reference "lambda 1e-4" always runs
    """
    variants = {
        name: fn
        for name, fn in TRAINING_VARIANTS.items()
        if not selected or name in selected or name == "lambda 1e-4"
    }
    scores = {name: np.empty(len(y01)) for name in variants}
    for k, valid in enumerate(folds):
        print(f"fold {k + 1}/{len(folds)}...")
        tr = np.concatenate([f for i, f in enumerate(folds) if i != k])
        params = run.fit_final(x[tr], names)
        tx_tr = run.transform_poly(x[tr], params)
        tx_valid = run.transform_poly(x[valid], params)
        for name, train_fn in variants.items():
            w = train_fn(y01[tr], tx_tr)
            scores[name][valid] = imp.sigmoid(tx_valid @ w)
    return {name: run.score_folds(y01, s, folds) for name, s in scores.items()}


# Ensembles of the final logistic regression with small neural networks (mlp.py).
# The MLP is trained on the 198 baseline features; the probabilities are averaged.
MLP_SETTINGS = dict(n_hidden=128, epochs=6, learning_rate=1e-3, weight_decay=1e-4)
MLP_SEEDS = [42, 43, 44, 45, 46]
ENSEMBLE_WEIGHTS = [0.2, 0.3, 0.4]  # weight of the MLP in the average


def compare_ensembles(x, y01, names, folds):
    """Cross-validate logistic regression (run.py), MLPs and their averages.

    Returns:
        dict name -> (threshold, per-fold F1, per-fold accuracy)
    """
    oof = {name: np.empty(len(y01)) for name in ["logistic", "MLP", "5 MLPs"]}
    for k, valid in enumerate(folds):
        print(f"fold {k + 1}/{len(folds)}...")
        tr = np.concatenate([f for i, f in enumerate(folds) if i != k])
        params = run.fit_final(x[tr], names)
        w = run.train_newton(y01[tr], run.transform_poly(x[tr], params))
        tx_valid = run.transform_poly(x[valid], params).astype(np.float64)
        oof["logistic"][valid] = imp.sigmoid(tx_valid @ w)

        base = run.fit_preprocessing(x[tr], names)
        x_tr = run.transform(x[tr], base)[:, 1:]  # the MLP has its own bias
        x_valid = run.transform(x[valid], base)[:, 1:]
        probas = [
            mlp.mlp_predict_proba(
                mlp.train_mlp(y01[tr], x_tr, seed=seed, **MLP_SETTINGS), x_valid
            )
            for seed in MLP_SEEDS
        ]
        oof["MLP"][valid] = probas[0]
        oof["5 MLPs"][valid] = np.mean(probas, axis=0)

    for name in ["MLP", "5 MLPs"]:
        for a in ENSEMBLE_WEIGHTS:
            oof[f"{a:.1f} {name}"] = a * oof[name] + (1 - a) * oof["logistic"]
    return {name: run.score_folds(y01, p, folds) for name, p in oof.items()}


# Gradient boosted trees (boosting.py) on the cleaned raw features, sparse columns
# included (no imputation: missing values get their own bin), alone and averaged
# with the logistic regression
BOOSTING_SETTINGS = dict(learning_rate=0.1, max_depth=4)
BOOSTING_TREES = [250, 300, 350, 400]  # evaluated along the way, in one training
BOOSTING_WEIGHTS = [0.5, 0.6, 0.7, 0.8]  # weight of the boosting in the average


def boosting_features(x_tr, x_valid, names):
    """Binned features for the trees, as in run.py."""
    params = run.fit_tree_features(x_tr, names)
    return (
        run.transform_tree_features(x_tr, params),
        run.transform_tree_features(x_valid, params),
    )


def compare_boosting(x, y01, names, folds):
    """Cross-validate the boosted trees, alone and averaged with run.py's model.

    Returns:
        dict name -> (threshold, per-fold F1, per-fold accuracy)
    """
    oof = {"logistic": np.empty(len(y01))}
    oof.update({f"GB {n}": np.empty(len(y01)) for n in BOOSTING_TREES})
    for k, valid in enumerate(folds):
        print(f"fold {k + 1}/{len(folds)}...")
        tr = np.concatenate([f for i, f in enumerate(folds) if i != k])
        params = run.fit_final(x[tr], names)
        w = run.train_newton(y01[tr], run.transform_poly(x[tr], params))
        tx_valid = run.transform_poly(x[valid], params).astype(np.float64)
        oof["logistic"][valid] = imp.sigmoid(tx_valid @ w)

        xb_tr, xb_valid = boosting_features(x[tr], x[valid], names)
        score = {"valid": None}

        def staged(n, model):  # running score of the validation fold
            s = model["lr"] * gb.predict_tree(model["trees"][-1], xb_valid)
            score["valid"] = (
                s + model["base"] if score["valid"] is None else score["valid"] + s
            )
            if n in BOOSTING_TREES:
                oof[f"GB {n}"][valid] = imp.sigmoid(score["valid"])

        gb.train_boosting(
            y01[tr],
            xb_tr,
            n_trees=max(BOOSTING_TREES),
            callback=staged,
            **BOOSTING_SETTINGS,
        )

    for n in BOOSTING_TREES:
        for a in BOOSTING_WEIGHTS:
            oof[f"{a:.1f} GB {n}"] = a * oof[f"GB {n}"] + (1 - a) * oof["logistic"]
    return {name: run.score_folds(y01, p, folds) for name, p in oof.items()}


# Sub-group module indicators for the logistic regression: whether a person
# answered the questions of an optional module (asked e.g. only to diabetics). The
# questions of a module share the same pattern, so one indicator per distinct
# pattern is kept. These are the columns that made the boosted trees better.
OOF_FILE = "data/oof_predictions.npz"


def fit_indicators(x, names):
    """run.fit_final, plus one "answered" indicator per distinct module pattern."""
    params = run.fit_final(x, names)
    base_keep = params["base"]["keep"]
    dropped = run.DROPPED_COLUMNS + run.LEAKY_COLUMNS
    candidates = [
        j
        for j, name in enumerate(names)
        if not base_keep[j] and name not in dropped and not np.all(np.isnan(x[:, j]))
    ]
    seen, columns = set(), []
    for j in candidates:
        answered = ~np.isnan(x[:, j])
        pattern = np.packbits(answered).tobytes()
        if answered.mean() >= MIN_FREQUENCY and pattern not in seen:
            seen.add(pattern)
            columns.append(j)
    answered = ~np.isnan(x[:, columns])
    params["indicators"] = columns
    params["ind_mean"], params["ind_std"] = answered.mean(axis=0), answered.std(axis=0)
    return params


def transform_indicators(x, params):
    """run.transform_poly features followed by the standardized indicators."""
    answered = ~np.isnan(x[:, params["indicators"]])
    indicators = (answered - params["ind_mean"]) / params["ind_std"]
    return np.c_[run.transform_poly(x, params), indicators.astype(np.float32)]


def compare_indicators(x, y01, names, folds):
    """Logistic regression with and without indicators, the boosted trees of
    run.py, and their 50/50 averages. The out-of-fold predictions are saved to
    OOF_FILE so that later ensemble experiments do not need to retrain."""
    oof = {name: np.empty(len(y01)) for name in ["logistic", "logistic+ind", "GB"]}
    for k, valid in enumerate(folds):
        print(f"fold {k + 1}/{len(folds)}...")
        tr = np.concatenate([f for i, f in enumerate(folds) if i != k])
        for name, fit_fn, transform_fn in [
            ("logistic", run.fit_final, run.transform_poly),
            ("logistic+ind", fit_indicators, transform_indicators),
        ]:
            params = fit_fn(x[tr], names)
            w = run.train_newton(y01[tr], transform_fn(x[tr], params))
            tx_valid = transform_fn(x[valid], params).astype(np.float64)
            oof[name][valid] = imp.sigmoid(tx_valid @ w)
        if k == 0:
            print(f"  {len(params['indicators'])} module indicators")
        xb_tr, xb_valid = boosting_features(x[tr], x[valid], names)
        booster = gb.train_boosting(
            y01[tr],
            xb_tr,
            n_trees=run.BOOSTING_SETTINGS["n_trees"],
            learning_rate=run.BOOSTING_SETTINGS["learning_rate"],
            max_depth=run.BOOSTING_SETTINGS["max_depth"],
        )
        oof["GB"][valid] = gb.predict_boosting(booster, xb_valid)
    np.savez(OOF_FILE, **{k.replace("+", "_"): v for k, v in oof.items()})
    oof["0.5 GB + log."] = 0.5 * oof["GB"] + 0.5 * oof["logistic"]
    oof["0.5 GB + log+ind"] = 0.5 * oof["GB"] + 0.5 * oof["logistic+ind"]
    return {name: run.score_folds(y01, p, folds) for name, p in oof.items()}


# Boosting variants, averaged with the logistic regression whose out-of-fold
# predictions were saved by the indicators experiment (same folds), so that only
# the trees are retrained
BOOSTING_VARIANTS = {
    "lr 0.05": (dict(learning_rate=0.05, max_depth=4), [400, 500]),
    "col 0.5": (dict(learning_rate=0.1, max_depth=4, colsample=0.5), [300, 350]),
}
VARIANT_WEIGHTS = [0.5, 0.6, 0.7]


def compare_boosting_variants(x, y01, names, folds):
    """Boosting variants alone and averaged with the saved logistic predictions."""
    saved = dict(np.load(OOF_FILE))
    oof = {"logistic": saved["logistic"], "current": None}
    oof["current"] = 0.5 * saved["GB"] + 0.5 * saved["logistic"]  # run.py's model
    for name, (_, trees) in BOOSTING_VARIANTS.items():
        oof.update({f"{name} {n}": np.empty(len(y01)) for n in trees})
    for k, valid in enumerate(folds):
        print(f"fold {k + 1}/{len(folds)}...")
        tr = np.concatenate([f for i, f in enumerate(folds) if i != k])
        xb_tr, xb_valid = boosting_features(x[tr], x[valid], names)
        for name, (settings, trees) in BOOSTING_VARIANTS.items():
            score = {"valid": None}

            def staged(n, model, name=name, trees=trees, score=score, valid=valid):
                s = model["lr"] * gb.predict_tree(model["trees"][-1], xb_valid)
                score["valid"] = (
                    s + model["base"] if score["valid"] is None else score["valid"] + s
                )
                if n in trees:
                    oof[f"{name} {n}"][valid] = imp.sigmoid(score["valid"])

            gb.train_boosting(
                y01[tr], xb_tr, n_trees=max(trees), callback=staged, **settings
            )
    for name, (_, trees) in BOOSTING_VARIANTS.items():
        for n in trees:
            for a in VARIANT_WEIGHTS:
                oof[f"{a:.1f} {name} {n}"] = (
                    a * oof[f"{name} {n}"] + (1 - a) * oof["logistic"]
                )
    np.savez(
        OOF_FILE,
        **saved,
        **{
            k.replace(" ", "_").replace(".", "p"): v
            for k, v in oof.items()
            if k.startswith(("lr", "col"))
        },
    )
    return {name: run.score_folds(y01, p, folds) for name, p in oof.items()}


# Global health features: models do not easily "count" over many columns (a tree
# of depth 4 sees at most 4 of them), so we give them the totals directly.
CONDITIONS = [  # "ever told you had ...": 1 = yes in BRFSS
    "CVDSTRK3",
    "ASTHMA3",
    "CHCSCNCR",
    "CHCOCNCR",
    "CHCCOPD1",
    "HAVARTH3",
    "ADDEPEV2",
    "CHCKIDNY",
    "DIABETE3",
    "BPHIGH4",
    "TOLDHI2",
]
LIMITATIONS = [  # 1 = yes
    "QLACTLM2",
    "USEEQUIP",
    "BLIND",
    "DECIDE",
    "DIFFWALK",
    "DIFFDRES",
    "DIFFALON",
]


def health_counts(x, names):
    """Number of chronic conditions, number of limitations, and days of poor
    physical + mental health in the last month (capped at 30).

    Returns:
        shape=(N, 3)
    """
    col = {name: j for j, name in enumerate(names)}
    clean = run.replace_special_codes(x)
    n_conditions = sum(clean[:, col[c]] == 1 for c in CONDITIONS)
    n_limitations = sum(clean[:, col[c]] == 1 for c in LIMITATIONS)
    days = np.nan_to_num(clean[:, col["PHYSHLTH"]]) + np.nan_to_num(
        clean[:, col["MENTHLTH"]]
    )
    return np.c_[n_conditions, n_limitations, np.minimum(days, 30)].astype(float)


def fit_counts(x, names):
    """run.fit_final, plus the mean/std of the health counts."""
    params = run.fit_final(x, names)
    counts = health_counts(x, names)
    params["count_mean"], params["count_std"] = counts.mean(axis=0), counts.std(axis=0)
    params["names"] = names
    return params


def transform_counts(x, params):
    """run.transform_poly features followed by the standardized health counts."""
    counts = (health_counts(x, params["names"]) - params["count_mean"]) / params[
        "count_std"
    ]
    return np.c_[run.transform_poly(x, params), counts.astype(np.float32)]


def compare_counts(x, y01, names, folds):
    """Health counts added to the logistic regression and to the trees, compared
    with run.py's model (saved out-of-fold predictions, same folds)."""
    saved = dict(np.load(OOF_FILE))
    oof = {"current": 0.5 * saved["GB"] + 0.5 * saved["logistic"]}
    oof.update({k: np.empty(len(y01)) for k in ["logistic+counts", "GB+counts"]})
    for k, valid in enumerate(folds):
        print(f"fold {k + 1}/{len(folds)}...")
        tr = np.concatenate([f for i, f in enumerate(folds) if i != k])
        params = fit_counts(x[tr], names)
        w = run.train_newton(y01[tr], transform_counts(x[tr], params))
        tx_valid = transform_counts(x[valid], params).astype(np.float64)
        oof["logistic+counts"][valid] = imp.sigmoid(tx_valid @ w)
        trees = run.fit_tree_features(x[tr], names)
        edges = gb.fit_bins(health_counts(x[tr], names))
        xb_tr = np.r_[
            run.transform_tree_features(x[tr], trees),
            gb.apply_bins(health_counts(x[tr], names), edges),
        ]
        xb_valid = np.r_[
            run.transform_tree_features(x[valid], trees),
            gb.apply_bins(health_counts(x[valid], names), edges),
        ]
        booster = gb.train_boosting(
            y01[tr],
            xb_tr,
            n_trees=run.BOOSTING_SETTINGS["n_trees"],
            learning_rate=run.BOOSTING_SETTINGS["learning_rate"],
            max_depth=run.BOOSTING_SETTINGS["max_depth"],
        )
        oof["GB+counts"][valid] = gb.predict_boosting(booster, xb_valid)
    np.savez(
        OOF_FILE,
        **saved,
        logistic_counts=oof["logistic+counts"],
        GB_counts=oof["GB+counts"],
    )
    oof["0.5 both+counts"] = 0.5 * oof["GB+counts"] + 0.5 * oof["logistic+counts"]
    oof["0.5 GB+counts"] = 0.5 * oof["GB+counts"] + 0.5 * saved["logistic"]
    oof["0.5 log+counts"] = 0.5 * saved["GB"] + 0.5 * oof["logistic+counts"]
    return {name: run.score_folds(y01, p, folds) for name, p in oof.items()}


# Learning curve (lecture 4b): training and validation loss for growing training
# sets. Close curves mean high bias (the model is too simple), a gap means high
# variance (overfitting).
LC_FRACTIONS = [0.01, 0.03, 0.1, 0.3, 1.0]  # of the 4 training folds
LC_VARIANTS = {
    "baseline (198 features)": (run.fit_preprocessing, run.transform),
    "squares + interactions (360 features)": (run.fit_poly, run.transform_poly),
}
LC_FIGURE = "report/learning_curve.pdf"


def learning_curve(x, y01, names, folds):
    """Train on growing random subsets of folds 2-5, evaluate on fold 1.

    Returns:
        list of (variant, training size, training loss, validation loss, validation F1)
    """
    valid = folds[0]
    pool = np.random.default_rng(run.SEED).permutation(np.concatenate(folds[1:]))
    rows = []
    for name, (fit_fn, transform_fn) in LC_VARIANTS.items():
        for fraction in LC_FRACTIONS:
            tr = pool[: int(fraction * len(pool))]
            print(f"{name}, {len(tr)} training points...")
            params = fit_fn(x[tr], names)
            tx_tr, tx_valid = transform_fn(x[tr], params), transform_fn(
                x[valid], params
            )
            w = run.train_newton(y01[tr], tx_tr).astype(tx_tr.dtype)
            _, f1 = run.best_threshold(y01[valid], imp.sigmoid(tx_valid @ w))
            train_loss = imp.logistic_loss(y01[tr], tx_tr, w)
            valid_loss = imp.logistic_loss(y01[valid], tx_valid, w)
            rows.append((name, len(tr), train_loss, valid_loss, f1))
    return rows


def plot_learning_curve(rows, path):
    """Save the learning curve as a vector figure for the report (one IEEE column)."""
    import matplotlib

    matplotlib.use("Agg")  # no window, just the file
    import matplotlib.pyplot as plt

    ink, muted, grid = "#2b2b2b", "#6b6b6b", "#e3e3e3"
    styles = {  # fixed categorical order: slot 1 blue, slot 2 orange
        "baseline (198 features)": ("#2a78d6", "o"),
        "squares + interactions (360 features)": ("#eb6834", "s"),
    }
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for name, (color, marker) in styles.items():
        r = [row for row in rows if row[0] == name]
        sizes = [row[1] for row in r]
        for idx, dash, label in [(3, "-", "validation"), (2, "--", "training")]:
            ax.plot(
                sizes,
                [row[idx] for row in r],
                dash,
                color=color,
                marker=marker,
                markersize=4,
                linewidth=1.5,
                label=f"{name.split(' (')[0]}, {label}",
                markeredgecolor="white",
                markeredgewidth=0.8,
            )
    ax.set_xscale("log")
    ax.set_xlabel("training set size", color=ink, fontsize=8)
    ax.set_ylabel("logistic loss", color=ink, fontsize=8)
    ax.tick_params(colors=muted, labelsize=7)
    ax.grid(True, color=grid, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.legend(
        fontsize=6.5, frameon=False, labelcolor=ink, loc="upper right", handlelength=3.5
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    print(f"figure saved to {path}")


def compare_features(x, y01, names, folds, selected, variants, train_fn, reference):
    """Cross-validate preprocessing variants with regularized logistic regression.

    Args:
        selected: names of the variants to run (all of them if empty)
        variants: dict name -> (fit_fn, transform_fn)
        train_fn: run.train (gradient descent) or run.train_newton
        reference: variant that always runs, as the reference row
    """
    results = {}
    for name, (fit_fn, transform_fn) in variants.items():
        if selected and name not in selected and name != reference:
            continue
        print(f"{name}...")
        start = time.time()
        results[name] = run.cross_validate(
            x, y01, names, fit_fn, transform_fn, folds, train_fn
        )
        print(f"  done in {time.time() - start:.0f}s")
    return results


def print_table(results, reference):
    """Print mean +- std over folds, and the paired F1 difference to `reference`."""
    ref_f1 = results[reference][1]
    print(f"\n{run.K_FOLDS}-fold cross-validation (mean +- std over folds)")
    print(f"{'':<15}{'F1':<18}{'accuracy':<18}{'threshold':<11}F1 vs {reference}")
    for name, (threshold, f1s, accs) in results.items():
        diff = f1s - ref_f1
        print(
            f"{name:<15}{f1s.mean():.4f} +- {f1s.std():.4f}   "
            f"{accs.mean():.4f} +- {accs.std():.4f}   {threshold:<11.2f}"
            f"{diff.mean():+.4f} +- {diff.std():.4f}"
        )


def main():
    experiment = sys.argv[1] if len(sys.argv) > 1 else "all"
    x_train, _, y_train, _, names = run.load_data()
    y01 = (y_train + 1) / 2
    # Same folds as run.py, so the reference rows match its cross-validation score
    folds = run.kfold_indices(len(y01), run.K_FOLDS, run.SEED)

    selected = sys.argv[2:]  # e.g. python cross_validation.py features squares
    if experiment in ("models", "all"):
        results = compare_models(x_train, y01, names, folds, selected)
        print_table(results, "reg. logistic")
    if experiment in ("features", "all"):
        results = compare_features(
            x_train, y01, names, folds, selected, VARIANTS, run.train, "baseline"
        )
        print_table(results, "baseline")
    if experiment in ("newton", "all"):
        results = compare_features(
            x_train,
            y01,
            names,
            folds,
            selected,
            NEWTON_VARIANTS,
            run.train_newton,
            "10 factors",
        )
        print_table(results, "10 factors")
    if experiment in ("tuning", "all"):
        results = compare_training(x_train, y01, names, folds, selected)
        print_table(results, "lambda 1e-4")
    if experiment in ("ensemble", "all"):
        print_table(compare_ensembles(x_train, y01, names, folds), "logistic")
    if experiment in ("boosting", "all"):
        print_table(compare_boosting(x_train, y01, names, folds), "logistic")
    if experiment in ("indicators", "all"):
        print_table(compare_indicators(x_train, y01, names, folds), "logistic")
    if experiment == "boosting_variants":  # needs the indicators experiment first
        print_table(compare_boosting_variants(x_train, y01, names, folds), "current")
    if experiment == "counts":  # needs the indicators experiment first
        print_table(compare_counts(x_train, y01, names, folds), "current")
    if experiment in ("learning_curve", "all"):
        rows = learning_curve(x_train, y01, names, folds)
        print(
            f"\n{'':<26}{'size':>8}{'train loss':>12}{'valid loss':>12}{'valid F1':>10}"
        )
        for name, size, train_loss, valid_loss, f1 in rows:
            print(
                f"{name:<26}{size:>8}{train_loss:>12.4f}{valid_loss:>12.4f}{f1:>10.4f}"
            )
        plot_learning_curve(rows, LC_FIGURE)


if __name__ == "__main__":
    main()
