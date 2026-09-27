import numpy as np
from ml_methods_helpers import compute_loss

def least_squares(y,tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    
    w = np.linalg.solve(np.matmul(np.transpose(tx), tx), np.matmul(np.transpose(tx), y))
    mse = compute_loss(y, tx, w, mae=False)
    return [w, mse]