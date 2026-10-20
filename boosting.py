"""Gradient boosted decision trees for binary classification, in NumPy.

Each tree asks yes/no questions on the features ("is age above this value?") and
the trees are added one after the other, each one correcting the errors of the
previous ones (boosting). As in XGBoost, each tree is fitted with the first and
second derivatives of the logistic loss (a Newton step in function space).

For speed, every feature is first cut into at most N_BINS quantile bins, with a
separate bin 0 for missing values, so that the best split of a node is found from
per-bin histograms instead of sorting the data.
"""

import numpy as np

from implementations import sigmoid

N_BINS = 64  # bins per feature, bin 0 = missing value (fits in uint8)


def fit_bins(x, n_bins=N_BINS):
    """Quantile bin edges of each feature, learned on the training data.

    Args:
        x: shape=(N, D), may contain NaN

    Returns:
        list of D arrays of increasing edges
    """
    quantiles = np.linspace(0, 1, n_bins)[1:-1]
    edges = []
    for col in x.T:
        values = col[~np.isnan(col)]
        edges.append(np.unique(np.quantile(values, quantiles)) if len(values) else [])
    return edges


def apply_bins(x, edges):
    """Bin index of every value, feature-major for fast per-feature histograms.

    Returns:
        shape=(D, N), uint8: 0 for missing values, then 1, 2, ... by increasing value
    """
    out = np.zeros((x.shape[1], len(x)), dtype=np.uint8)
    for j, e in enumerate(edges):
        col = x[:, j]
        b = np.searchsorted(e, col, side="right") + 1
        b[np.isnan(col)] = 0
        out[j] = b
    return out


def _histograms(xb, g, h, idx):
    """Per-feature, per-bin sums of g and h over the samples idx, shape (D, N_BINS)."""
    gs, hs = g[idx], h[idx]
    sub = xb[:, idx]
    hist_g = np.stack([np.bincount(row, gs, N_BINS) for row in sub])
    hist_h = np.stack([np.bincount(row, hs, N_BINS) for row in sub])
    return hist_g, hist_h


def _best_split(hist_g, hist_h, lambda_, min_child_weight):
    """Best (feature, bin) split of a node, from its histograms.

    Samples with bin <= b go left (missing values, bin 0, always go left).

    Returns:
        (gain, feature, bin), gain <= 0 if no valid split
    """
    g_left = np.cumsum(hist_g, axis=1)[:, :-1]
    h_left = np.cumsum(hist_h, axis=1)[:, :-1]
    g_tot, h_tot = hist_g[0].sum(), hist_h[0].sum()  # any feature sums to the total
    g_right, h_right = g_tot - g_left, h_tot - h_left
    gain = (
        g_left**2 / (h_left + lambda_)
        + g_right**2 / (h_right + lambda_)
        - g_tot**2 / (h_tot + lambda_)
    )
    gain[(h_left < min_child_weight) | (h_right < min_child_weight)] = -np.inf
    feature, b = np.unravel_index(np.argmax(gain), gain.shape)
    return gain[feature, b], feature, b


def build_tree(xb, g, h, idx, max_depth, lambda_, min_child_weight):
    """Grow one regression tree on the gradients g and Hessians h of samples idx.

    Histogram subtraction: only the smaller child's histograms are computed from
    the data, the larger child's are the parent's minus the smaller child's.

    Returns:
        dict of arrays: feature (-1 for a leaf), bin, left, right, value
    """
    tree = {"feature": [], "bin": [], "left": [], "right": [], "value": []}

    def grow(node_idx, depth, hists):
        node = len(tree["feature"])
        for key in tree:
            tree[key].append(0)
        gain = 0
        if depth < max_depth and len(node_idx) > 1:
            gain, feature, b = _best_split(*hists, lambda_, min_child_weight)
        if gain <= 0:  # leaf: Newton step -G / (H + lambda)
            tree["feature"][node] = -1
            tree["value"][node] = -g[node_idx].sum() / (h[node_idx].sum() + lambda_)
            return node
        goes_left = xb[feature, node_idx] <= b
        left_idx, right_idx = node_idx[goes_left], node_idx[~goes_left]
        left_hists = right_hists = None
        if depth + 1 < max_depth:  # the children may split: they need histograms
            small_is_left = len(left_idx) <= len(right_idx)
            small = _histograms(xb, g, h, left_idx if small_is_left else right_idx)
            large = (hists[0] - small[0], hists[1] - small[1])
            left_hists, right_hists = (
                (small, large) if small_is_left else (large, small)
            )
        tree["feature"][node], tree["bin"][node] = feature, b
        tree["left"][node] = grow(left_idx, depth + 1, left_hists)
        tree["right"][node] = grow(right_idx, depth + 1, right_hists)
        return node

    grow(idx, 0, _histograms(xb, g, h, idx) if max_depth > 0 else None)
    return {key: np.array(values) for key, values in tree.items()}


def predict_tree(tree, xb):
    """Output of one tree for binned data xb of shape (D, N)."""
    node = np.zeros(xb.shape[1], dtype=int)
    rows = np.arange(xb.shape[1])
    while True:
        internal = tree["feature"][node] >= 0
        if not internal.any():
            return tree["value"][node]
        n, r = node[internal], rows[internal]
        goes_left = xb[tree["feature"][n], r] <= tree["bin"][n]
        node[internal] = np.where(goes_left, tree["left"][n], tree["right"][n])


def train_boosting(
    y,
    xb,
    n_trees=300,
    learning_rate=0.1,
    max_depth=4,
    lambda_=1.0,
    min_child_weight=1.0,
    subsample=0.7,
    colsample=1.0,
    seed=42,
    callback=None,
):
    """Gradient boosting with the logistic loss (labels in {0, 1}).

    Args:
        y: shape=(N, ), labels in {0, 1}
        xb: shape=(D, N), binned features from apply_bins
        n_trees: number of trees
        learning_rate: shrinkage of each tree's contribution
        max_depth: depth of each tree
        lambda_: L2 penalty on the leaf values
        min_child_weight: minimum sum of Hessians in a child
        subsample: fraction of the samples used to grow each tree
        colsample: fraction of the features each tree may split on (as in random
            forests: more diverse trees)
        seed: seed of the subsampling (reproducibility)
        callback: optional function(n_trees_so_far, model), called after each tree

    Returns:
        model: dict with the initial score and the list of trees
    """
    rng = np.random.default_rng(seed)
    p0 = y.mean()
    model = {"base": np.log(p0 / (1 - p0)), "trees": [], "lr": learning_rate}
    score = np.full(len(y), model["base"])
    for t in range(n_trees):
        p = sigmoid(score)
        g, h = p - y, p * (1 - p)
        idx = np.sort(rng.choice(len(y), int(subsample * len(y)), replace=False))
        if colsample < 1:
            features = np.sort(
                rng.choice(len(xb), int(colsample * len(xb)), replace=False)
            )
            tree = build_tree(
                xb[features], g, h, idx, max_depth, lambda_, min_child_weight
            )
            internal = tree["feature"] >= 0
            tree["feature"][internal] = features[tree["feature"][internal]]
        else:
            tree = build_tree(xb, g, h, idx, max_depth, lambda_, min_child_weight)
        model["trees"].append(tree)
        score += learning_rate * predict_tree(tree, xb)
        if callback is not None:
            callback(t + 1, model)
    return model


def predict_boosting(model, xb, n_trees=None):
    """Probability of class 1, using the first n_trees trees (all by default)."""
    trees = model["trees"][:n_trees]
    score = model["base"] + model["lr"] * sum(predict_tree(t, xb) for t in trees)
    return sigmoid(score)
