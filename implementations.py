import numpy as np

# ---------------------------------------------------------------------------
# Linear regression helpers
# ---------------------------------------------------------------------------


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # Number of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def compute_mse(y, tx, w):
    """Compute the MSE loss 1/(2N) * sum(e^2).

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, ). The vector of model parameters.

    Returns:
        the value of the loss (a numpy scalar), corresponding to the input parameters w.
    """
    e = y - tx @ w
    return e @ e / (2 * len(y))


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the MSE loss at w.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, ). The vector of model parameters.

    Returns:
        an array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx @ w
    return -tx.T @ e / len(y)


# ---------------------------------------------------------------------------
# Logistic regression helpers (labels must be in {0, 1})
# ---------------------------------------------------------------------------


def sigmoid(t):
    """Apply the sigmoid function on t, without overflow for large |t|.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    # 1 / (1 + exp(-t)) = exp(-log(1 + exp(-t)))
    return np.exp(-np.logaddexp(0, -t))


def logistic_loss(y, tx, w):
    """Compute the negative log-likelihood loss, without regularization term.

    Args:
        y:  shape=(N, ), labels in {0, 1}
        tx: shape=(N, D)
        w:  shape=(D, )

    Returns:
        a non-negative loss (numpy scalar)
    """
    z = tx @ w
    # log(1 + exp(z)) computed as logaddexp(0, z) to avoid overflow
    return np.mean(np.logaddexp(0, z) - y * z)


def logistic_gradient(y, tx, w, lambda_=0):
    """Compute the gradient of the (regularized) logistic loss.

    The penalty is lambda_ * ||w||^2, hence the 2 * lambda_ * w term.

    Args:
        y:  shape=(N, ), labels in {0, 1}
        tx: shape=(N, D)
        w:  shape=(D, )
        lambda_: scalar, regularization parameter

    Returns:
        a vector of shape (D, )
    """
    return tx.T @ (sigmoid(tx @ w) - y) / len(y) + 2 * lambda_ * w


def logistic_hessian(tx, w, lambda_=0):
    """Compute the Hessian of the (regularized) logistic loss.

    Uses tx.T @ diag(s) @ tx = (tx.T * s) @ tx, so the N x N diagonal matrix is never built.

    Args:
        tx: shape=(N, D)
        w:  shape=(D, )
        lambda_: scalar, regularization parameter

    Returns:
        a hessian matrix of shape=(D, D)
    """
    pred = sigmoid(tx @ w)
    s = pred * (1 - pred)
    return (tx.T * s) @ tx / tx.shape[0] + 2 * lambda_ * np.identity(tx.shape[1])


# ---------------------------------------------------------------------------
# Required methods
# ---------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent on the MSE loss.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        initial_w: shape=(D, ). The initial guess for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the last weight vector of the method, shape=(D, )
        loss: the corresponding MSE loss (scalar)
    """
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1) on the MSE loss.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        initial_w: shape=(D, ). The initial guess for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the last weight vector of the method, shape=(D, )
        loss: the corresponding MSE loss on the full dataset (scalar)
    """
    w = initial_w
    for _ in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1):
            w = w - gamma * compute_mse_gradient(minibatch_y, minibatch_tx, w)
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)

    Returns:
        w: optimal weights, shape=(D, )
        loss: the corresponding MSE loss (scalar)
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Minimizes MSE(w) + lambda_ * ||w||^2, i.e. solves (tx.T tx + 2N lambda_ I) w = tx.T y.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        lambda_: scalar, regularization parameter

    Returns:
        w: optimal weights, shape=(D, )
        loss: the corresponding MSE loss, without the penalty term (scalar)
    """
    N, D = tx.shape
    w = np.linalg.solve(tx.T @ tx + 2 * N * lambda_ * np.identity(D), tx.T @ y)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (labels in {0, 1}).

    Args:
        y: shape=(N, ), labels in {0, 1}
        tx: shape=(N, D)
        initial_w: shape=(D, ). The initial guess for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the last weight vector of the method, shape=(D, )
        loss: the corresponding logistic loss (scalar)
    """
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (labels in {0, 1}).

    Minimizes logistic_loss(w) + lambda_ * ||w||^2.

    Args:
        y: shape=(N, ), labels in {0, 1}
        tx: shape=(N, D)
        lambda_: scalar, regularization parameter
        initial_w: shape=(D, ). The initial guess for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the last weight vector of the method, shape=(D, )
        loss: the corresponding logistic loss, without the penalty term (scalar)
    """
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * logistic_gradient(y, tx, w, lambda_)
    return w, logistic_loss(y, tx, w)


# ---------------------------------------------------------------------------
# Extra (not graded): Newton's method, for the competition part
# ---------------------------------------------------------------------------


def reg_logistic_regression_newton(y, tx, lambda_, initial_w, max_iters, gamma=1.0):
    """Regularized logistic regression using Newton's method (labels in {0, 1}).

    Each step costs O(N D^2 + D^3), but needs far fewer iterations than GD.

    Args:
        y: shape=(N, ), labels in {0, 1}
        tx: shape=(N, D)
        lambda_: scalar, regularization parameter (> 0 keeps the Hessian invertible)
        initial_w: shape=(D, ). The initial guess for the model parameters
        max_iters: a scalar denoting the total number of Newton steps
        gamma: a scalar denoting the stepsize (1.0 is the pure Newton step)

    Returns:
        w: the last weight vector of the method, shape=(D, )
        loss: the corresponding logistic loss, without the penalty term (scalar)
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = logistic_gradient(y, tx, w, lambda_)
        hessian = logistic_hessian(tx, w, lambda_)
        w = w - gamma * np.linalg.solve(hessian, gradient)
    return w, logistic_loss(y, tx, w)
