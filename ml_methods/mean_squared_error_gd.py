from mean_squared_error_sgd import *

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma, mae=False):
    
    #Call mean_squared_error_sgd with a full batch
    return mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=tx.shape[0], mae=mae)