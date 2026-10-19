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
- learning_curve: training/validation loss vs training set size (bias-variance),
  saved as a figure for the report

The decision threshold is tuned once on the pooled out-of-fold predictions, then F1
and accuracy are reported per fold as mean +- std. Since the folds are shared, each
row is also compared to a reference fold by fold, which tells whether a difference
is real or just noise.

Usage: python cross_validation.py [models|features|newton|tuning [variant ...]|ensemble|boosting|learning_curve]
(models takes ~10 minutes; one-hot alone ~25 minutes, the other variants ~10 each)
"""

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
