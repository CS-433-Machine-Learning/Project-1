import numpy as np
from ml_methods.logistic_helpers import logistic_loss, logistic_gradient, logistic_hessian

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, newton = 0):
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    w = initial_w
    for _ in range(max_iters):
        
        # Calculate loss, gradient and hessian
        loss = logistic_loss(y, tx, w)
        gradient = logistic_gradient(y, tx, w, lambda_)
        hessian = logistic_hessian(y, tx, w, lambda_)
        
        #Do one step by Newton's method
        if newton:
            w = w - gamma * np.matmul(np.linalg.solve(hessian, np.identity(hessian.shape[0])), gradient)
            
        #else use regular gradient descent
        else:
            w = w - gamma * gradient
        
    return (loss, w)