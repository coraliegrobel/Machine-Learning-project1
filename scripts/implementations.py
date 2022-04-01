# -*- coding: utf-8 -*-
"""All implementations asked for."""

import numpy as np

from useful_functions import accuracy, compute_mse, compute_gradient
from proj1_helpers import predict_labels
from batch_iter import batch_iter
from logistic_regression import *



### Least Square Gradient Descent ###

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y,tx,w)
        w = w - gamma*grad
        
    loss = compute_mse(y, tx, w)
    return w, loss


### Least Square Stochastic Gradient Descent ###

def compute_stoch_gradient(y, tx, w, batch_size):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        grad = compute_gradient(minibatch_y, minibatch_tx, w)
        
    return grad

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    batch_size = 1
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_stoch_gradient(y,tx,w,batch_size)
        w = w - gamma*grad
    
    loss = compute_mse(y, tx, w)
    return w, loss

###  ------------------------------------ Least Square Normal equation  --------------------------------------###
def least_squares(y, tx): 
    "Least squares normal equation"
    """calculate the least squares solution."""
    
    w=np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y))
    
    mse = compute_mse(y,tx,w)
    rmse = np.sqrt(2*mse)
    return w,rmse

### ---------------------------------------- Ridge Regression ----------------------------------------------------###
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    e = y - tx.dot(w)
    loss = 0.5*np.mean(e**2) + lambda_*(np.linalg.norm(w)**2)

    return w,loss
    

### ---------------------------------------- Logistic regression ----------------------------------------------------###

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    losses = []
    threshold = 1e-8
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, weights = gradient_descent(y, tx, initial_w, gamma)

        losses.append(loss)
        if iter > 1:
            if np.abs(losses[-2]-losses[-1])<threshold:break
        
    return weights, losses[-1] 



### ---------------------------------------- Regularized logistic regression ----------------------------------------------------###
def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    losses = []
    threshold = 1e-8
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, weights = penalized_gradient_descent(y, tx, initial_w, gamma, lambda_)

        losses.append(loss)
        if iter > 1:
            if np.abs(losses[-2]-losses[-1])<threshold:break
        
    return weights, losses[-1] 