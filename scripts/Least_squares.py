# -*- coding: utf-8 -*-
"""All functions that we use in cross-validation"""

from useful_functions import *
import matplotlib.pyplot as plt
from proj1_helpers import *


def least_squares_normal(y, tx):
    """Return the weights and loss for Least squares"""
    w=np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y))
    
    # Computation of the loss
    mse = compute_mse(y,tx,w)
    rmse = np.sqrt(2*mse)
    return w,rmse

#------------------------------------------ Cross validation Least squares ---------------------------------------------#
def cross_validation_visualization_least(degree, accuracy_te, accuracy_tr):
    """visualization of the accuracy in function of the degree for train and test """
    plt.plot(degree, accuracy_te, marker=".", color='r', label='test accuracy')
    plt.plot(degree, accuracy_tr, marker=".", color='b', label='train accuracy')
    plt.xlabel("Degrees")
    plt.ylabel("Accuracy")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    
def cross_validation_least_squares(y, x, k_indices, k, degree):
    """return the accuracy of the train and test data set using least squares."""
    
    # get k'th subgroup in test, others in train
    k_val = np.reshape(np.vstack((k_indices[:k, :], k_indices[k+1:,])), -1)
    x_train = x[k_val]
    x_test = x[k_indices[k]]
    y_train = y[k_val]
    y_test = y[k_indices[k]]
    
    
    # form data with polynomial degree
    tx_train = build_poly(x_train, degree)
    tx_test = build_poly(x_test, degree)
    
    # least squares
    w_train,loss = least_squares_normal(y_train, tx_train)
    y_pred_te = predict_labels(w_train, tx_test)
    y_pred_tr = predict_labels(w_train, tx_train)
    
    #accuracy
    accuracy_tr = accuracy(y_pred_tr,y_train)
    accuracy_te = accuracy(y_pred_te,y_test)
    
    return accuracy_tr,accuracy_te

def cross_validation_demo_least_squares(x, y):
    """Find the best hyperparameters for Least Squares using accuracy method"""
    #initialisation 
    seed = 1
    degrees = range(1,15)
    k_fold = 4
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    accuracy_tr_temp = []
    accuracy_te_temp = []
    all_accuracy_tr = []
    all_accuracy_te = []
    
    # cross validation
    #Iteration across degrees
    for degree in degrees :
        accuracy_tr_temp = []
        accuracy_te_temp = []
        
        # Iteration across the k data set 
        for k in range(k_fold):
            accuracy_tr,accuracy_te = cross_validation_least_squares(y, x, k_indices, k, degree)
            accuracy_tr_temp.append(accuracy_tr)
            accuracy_te_temp.append(accuracy_te)
            
        all_accuracy_tr.append(np.mean(accuracy_tr_temp))
        all_accuracy_te.append(np.mean(accuracy_te_temp))
        
    degree_optimal = degrees[np.argmax(all_accuracy_te)]
    print("The optimal degree for Least squares method is %d with a mean accuracy accross the k-fold of %f."%(degree_optimal, all_accuracy_te[np.argmax(all_accuracy_te)]))
    cross_validation_visualization_least(degrees,all_accuracy_te,all_accuracy_tr)

    return degree_optimal, all_accuracy_te
