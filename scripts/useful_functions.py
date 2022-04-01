# -*- coding: utf-8 -*-
"""All functions that we use in different methods"""
import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e=y - tx.dot(w)
    length=len(e)
    grad = -1/length*np.transpose(tx).dot(e)
    return grad

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
    
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def split_data(x, y, ratio, seed=1):
    """ split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest."""
    # set seed to be able to find same result multiple times
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

def accuracy(y_pred, y_test):
    #Check the accuracy
    accuracy = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            accuracy += 1
    accuracy = accuracy / len(y_pred)
    return accuracy
