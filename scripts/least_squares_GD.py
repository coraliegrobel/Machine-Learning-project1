# -*- coding: utf-8 -*-
"""Gradient Descent"""

import numpy as np

from useful_functions import accuracy, compute_mse, compute_gradient
from proj1_helpers import predict_labels
from implementations import least_squares_GD


###### Cross validation ######

from useful_functions import compute_mse, build_k_indices
from plots_cross_validation import cross_validation_visualization

import matplotlib.pyplot as plt


def cross_validation(y, x, k_indices, k, lambda_, max_iters):
    """return the loss of ridge regression."""
    
    # get k'th subgroup in test, others in train
    k_val = np.reshape(np.vstack((k_indices[:k, :], k_indices[k+1:,])), -1)
    x_train = x[k_val]
    x_test = x[k_indices[k]]
    y_train = y[k_val]
    y_test = y[k_indices[k]]
    
    gamma = lambda_
    w_initial = np.zeros(x_train.shape[1])
    # gradient descent
    weight, loss_tr = least_squares_GD(y_train, x_train, w_initial, max_iters, gamma)
    
    # calculate the loss for test data
    loss_te = compute_mse(y_test, x_test, weight)
    return loss_tr, loss_te

def cross_validation_demo_GD(x, y, lambdas, max_iters):
    x_min=lambdas[0]
    x_max=lambdas[-1]
    y_min=0.5
    y_max=0.6
    seed = 1
    k_fold = 4
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    mse_tr = []
    mse_te = []
    full_mse_tr = []
    full_mse_te = []
    
    size_lambda = len(lambdas)
    
    # cross validation
    for k in range(k_fold) :
        mse_tr = []
        mse_te = []
        for ind, lambda_ in enumerate(lambdas):
            print("Current Step k=({k}/{k_f}) for lambda ({ind}/{tot}).   ".format(
                k=k+1, k_f=k_fold, ind=ind+1, tot=size_lambda), end="\r")
            cur_mse_tr, cur_mse_te = cross_validation(y, x, k_indices, k, lambda_, max_iters)
            mse_tr.append(cur_mse_tr)
            mse_te.append(cur_mse_te)
            
        full_mse_tr.append(mse_tr)
        full_mse_te.append(mse_te)
        if np.min(np.min(full_mse_tr)) < y_min:
            y_min = np.amin(full_mse_tr)
            y_max = 1.2*y_min
        cross_validation_visualization(lambdas, mse_tr, mse_te, x_min, x_max, 0.99*y_min, y_max)
    full_mse_tr_mean = np.mean(full_mse_tr, 0)
    full_mse_te_mean = np.mean(full_mse_te, 0)
    
    plt.figure()
    cross_validation_visualization(lambdas, np.mean(full_mse_tr, 0), np.mean(full_mse_te, 0), x_min, x_max, 0.99*y_min, y_max)
    plt.show()
    return full_mse_tr_mean, full_mse_te_mean