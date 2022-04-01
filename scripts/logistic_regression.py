

import numpy as np
import matplotlib.pyplot as plt
from useful_functions import *
from proj1_helpers import *


# ------------------------------------------ Logistic  -----------------------------------------------------#


#Logistic regression
def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """calculate loss by negative log likelihood."""
    sig = sigmoid(tx.dot(w))
    epsilon = 1e-16 #We use epsilon to avoid division by zero

    loss = y.T.dot(np.log(sig + epsilon)) + (1 - y).T.dot(np.log(1 - sig + epsilon))
    return loss/(tx.shape[0])

def calculate_gradient(y, tx, w):
    """compute the gradient of the loss"""
    sig = sigmoid(tx.dot(w))
    grad = tx.T.dot(sig - y)
    return grad

def gradient_descent(y, tx, w, learning_rate):
    """Compute the gradient of the loss"""
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w -= learning_rate * grad
    return loss, w

def run_logistic_regression(x, y, X_test, y_test, learning_rate):
    # init parameters
    max_iter = 300
    loss_tr = []
    loss_te = []
    y[np.where(y == -1)] = 0 #change (-1,1) to (0,1)
    y_test[np.where(y == -1)] = 0 #change (-1,1) to (0,1)
    threshold = 1e-8

    w = np.zeros((x.shape[1], ))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = gradient_descent(y, x, w, learning_rate)
        
        loss_tr.append(loss)
        curr_loss_te  = calculate_loss(y_test, X_test, w)
        loss_te.append(curr_loss_te)
        if iter > 1:
            if np.abs(loss_te[-2] - loss_te[-1]) < threshold: break
        
    y_pred = predict_labels(w, X_test)
    accuracy(y_pred, y_test)
    return w, loss_tr, loss_te, y_pred


# ------------------------------------------ Regularized -----------------------------------------------------#

def penalized_gradient_descent(y, tx, w, gamma, lambda_):
    """Compute the gradient and the loss for regularized logistic regression"""
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    w -= gamma * gradient
    return loss, w

def run_logistic_regression_penalized(x, y, X_test, y_test, learning_rate, lambda_):
    # init parameters
    max_iter = 20
    losses_tr = []
    losses_te = []
    y[np.where(y == -1)] = 0 #change (-1,1) to (0,1)
    y_test[np.where(y == -1)] = 0 #change (-1,1) to (0,1)
    threshold = 1e-8

    w = np.zeros((x.shape[1], ))
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = penalized_gradient_descent(y, x, w, learning_rate, lambda_)
        
        losses_tr.append(loss)
        losses_te.append(calculate_loss(y_test, X_test, w))
        if iter > 1:
            if np.abs(losses_te[-2]-losses_te[-1])<threshold:break

    
    y_pred = predict_labels(w ,X_test)

    return w, losses_tr, losses_te, y_pred


    #------------------------------------------ Cross validation Log ---------------------------------------------#
def cross_validation_visualization_log(hyperparameter, loss_tr, loss_te, type):
    """visualization of the loss accross hyperparameters."""
   
    plt.semilogx(hyperparameter, np.abs(loss_tr), marker=".", color='b', label='train error')
    plt.semilogx(hyperparameter, np.abs(loss_te), marker=".", color='r', label='test error')
    if type == "learning_rate":
        plt.xlabel("learning rates")
    else: plt.xlabel("lambdas")
    plt.ylabel("negative log likelyhood loss")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    
def cross_validation_log_reg(y, x, k_indices, k, learning_rate, lambda_):
    """return the loss of regularized logistic regression."""
    
    # get k'th subgroup in test, others in train
    k_val = np.reshape(np.vstack((k_indices[:k, :], k_indices[k+1:,])), -1)
    x_train = x[k_val]
    x_test = x[k_indices[k]]
    y_train = y[k_val]
    y_test = y[k_indices[k]]
    
    # compute the loss
    _, loss_tr, loss_te, _ = run_logistic_regression_penalized(x_train, y_train, x_test, y_test,learning_rate, lambda_)
    
    return loss_tr[-1], loss_te[-1]

def cross_validation_log(y, x, k_indices, k, learning_rate):
    """return the loss of logistic regression."""
    
    # get k'th subgroup in test, others in train
    k_val = np.reshape(np.vstack((k_indices[:k, :], k_indices[k+1:,])), -1)
    x_train = x[k_val]
    x_test = x[k_indices[k]]
    y_train = y[k_val]
    y_test = y[k_indices[k]]
    
    # compute the loss
    _, loss_tr, loss_te, _ = run_logistic_regression(x_train, y_train, x_test, y_test, learning_rate)
    return loss_tr[-1], loss_te[-1]



def cross_validation_demo_log_lr(x, y,show_box_plot = False):
    seed = 1
    k_fold = 4
    learning_rates = np.logspace(-10, 0, 5)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    loss_tr = []
    loss_te = []
    full_loss_tr = []
    full_loss_te = []
    
    # cross validation
    for ind, lr in enumerate(learning_rates):
        loss_tr = []
        loss_te = []
        
        for k in range(k_fold):
            cur_loss_tr, cur_loss_te = cross_validation_log(y, x, k_indices, k, lr)
            loss_tr.append(cur_loss_tr)
            loss_te.append(cur_loss_te)
            
        full_loss_tr.append(np.mean(loss_tr))
        full_loss_te.append(np.mean(loss_te))

    cross_validation_visualization_log(learning_rates, full_loss_tr, full_loss_te, "learning_rate")
    
    return learning_rates[np.argmin(np.abs(full_loss_te))]
    

def cross_validation_demo_log_lambda(x, y, learning_rate, show_box_plot = False):
    seed = 1
    k_fold = 4
    lambdas = np.logspace(-20, 0, 5)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    loss_tr = []
    loss_te = []
    full_loss_tr = []
    full_loss_te = []
    
    # cross validation
    for ind, lambda_ in enumerate(lambdas):
        loss_tr = []
        loss_te = []
        
        for k in range(k_fold):
            cur_loss_tr, cur_loss_te = cross_validation_log_reg(y, x, k_indices, k,learning_rate, lambda_)
            loss_tr.append(cur_loss_tr)
            loss_te.append(cur_loss_te)
            
        full_loss_tr.append(np.mean(loss_tr))
        full_loss_te.append(np.mean(loss_te))

    cross_validation_visualization_log(lambdas, full_loss_tr, full_loss_te, "lambda")
    
    return lambdas[np.argmin(np.abs(full_loss_te))]


