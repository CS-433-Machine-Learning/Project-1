"""A small neural network (multi-layer perceptron) for binary classification, in NumPy.

One hidden layer with ReLU activations and a sigmoid output, trained on the
logistic loss with mini-batches and the Adam optimizer. Unlike logistic regression
on hand-made features, the hidden layer learns its own non-linear combinations of
the inputs.
"""

import numpy as np

from implementations import sigmoid


def init_mlp(n_inputs, n_hidden, rng):
    """Random initial weights (He initialization for the ReLU layer).

    Returns:
        dict with W1 (D, H), b1 (H, ), w2 (H, ), b2 (scalar), all float32
    """
    return {
        "W1": (rng.normal(size=(n_inputs, n_hidden)) * np.sqrt(2 / n_inputs)).astype(
            np.float32
        ),
        "b1": np.zeros(n_hidden, dtype=np.float32),
        "w2": (rng.normal(size=n_hidden) * np.sqrt(1 / n_hidden)).astype(np.float32),
        "b2": np.float32(0),
    }


def mlp_forward(params, x):
    """Hidden activations and output logits.

    Args:
        x: shape=(N, D)

    Returns:
        h: shape=(N, H), z: shape=(N, )
    """
    h = np.maximum(x @ params["W1"] + params["b1"], 0)
    z = h @ params["w2"] + params["b2"]
    return h, z


def mlp_predict_proba(params, x, batch_size=65536):
    """Predicted probability of class 1, computed by batches to limit memory."""
    out = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        _, z = mlp_forward(params, x[start : start + batch_size])
        out[start : start + batch_size] = sigmoid(z)
    return out


def mlp_gradients(params, x, y, weight_decay):
    """Gradients of the mean logistic loss (+ weight_decay/2 ||W||^2) on a batch."""
    h, z = mlp_forward(params, x)
    dz = (sigmoid(z) - y) / len(y)  # derivative of the logistic loss w.r.t. z
    dh = np.outer(dz, params["w2"]) * (h > 0)  # back through the ReLU
    return {
        "W1": x.T @ dh + weight_decay * params["W1"],
        "b1": dh.sum(axis=0),
        "w2": h.T @ dz + weight_decay * params["w2"],
        "b2": dz.sum(),
    }


def train_mlp(
    y,
    x,
    n_hidden=64,
    epochs=10,
    batch_size=512,
    learning_rate=1e-3,
    weight_decay=1e-4,
    seed=42,
    callback=None,
):
    """Train the MLP with Adam on mini-batches (labels in {0, 1}).

    Args:
        y: shape=(N, ), labels in {0, 1}
        x: shape=(N, D), standardized features
        n_hidden: number of hidden units
        epochs: number of passes over the data
        batch_size, learning_rate, weight_decay: optimization settings
        seed: seed of the initialization and of the batch order (reproducibility)
        callback: optional function(epoch, params), called after each epoch

    Returns:
        params: the trained weights
    """
    rng = np.random.default_rng(seed)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    params = init_mlp(x.shape[1], n_hidden, rng)
    # Adam state: running means of the gradient and of its square
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}
    beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            idx = order[start : start + batch_size]
            grads = mlp_gradients(params, x[idx], y[idx], weight_decay)
            step += 1
            for k in params:
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                m_hat = m[k] / (1 - beta1**step)
                v_hat = v[k] / (1 - beta2**step)
                params[k] = (
                    params[k] - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
                ).astype(np.float32)
        if callback is not None:
            callback(epoch, params)
    return params
