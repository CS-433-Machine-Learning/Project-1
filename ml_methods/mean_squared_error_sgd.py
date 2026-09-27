from ml_methods_helpers import batch_iter, compute_loss, compute_gradient

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1, mae=False):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for m_y, m_tx in batch_iter(y, tx, batch_size):
            loss = compute_loss(m_y, m_tx, w, mae=mae)
            g = compute_gradient(m_y, m_tx, w, mae=mae)
            w = w - gamma * g
            ws.append(w)
            losses.append(loss)
        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, ws