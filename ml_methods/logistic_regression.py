import numpy as np
from ml_methods.reg_logistic_regression import reg_logistic_regression

def logistic_regression(y, tx, initial_w, max_iters, gamma, newton = 0):
    
    # Call regularized logistic regression with punishment factor 0
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma, newton)