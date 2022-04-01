# -*- coding: utf-8 -*-
"""All functions that we use in cross-validation"""

from useful_functions import *
import matplotlib.pyplot as plt
from proj1_helpers import *
from useful_functions import *

def ridge_regression_function(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    
    # Computation of the loss taking into account the lambda_
    e = y - tx.dot(w)
    loss = 0.5*np.mean(e**2) + lambda_*(np.linalg.norm(w)**2)

    return w,loss

#------------------------------------------ Cross validation Visualization Ridge regression ------------------------------------------#
def cross_validation_visualization_ridge_regression(degrees, accuracy_te, accuracy_tr):
    """Visualisation of the Accuracy in function of the degree according to the degree"""
    plt.plot(degrees, accuracy_tr, marker=".", color='b', label='train error')
    plt.plot(degrees, accuracy_te, marker=".", color='r', label='test accuracy')
    plt.xlabel("degrees")
    plt.ylabel("Accuracy")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    
# -------------------------------------- Cross validation ridge regression by accuracy -------------------------------------- #
    
def cross_validation_accuracy_ridge(y, x, k_indices, k, lambda_, degree):
    """Return the accuracy of the test and train data set using ridge regression"""
    
    # get k'th subgroup in test, others in train
    k_val = np.reshape(np.vstack((k_indices[:k, :], k_indices[k+1:,])), -1)
    x_train = x[k_val]
    x_test = x[k_indices[k]]
    y_train = y[k_val]
    y_test = y[k_indices[k]]
    
    # form data with polynomial degree
    tx_train = build_poly(x_train, degree)
    tx_test = build_poly(x_test, degree)
    
    # ridge regression
    w_train,loss = ridge_regression_function(y_train, tx_train, lambda_)
    
    # Prediction of the y according to the weights that we find 
    y_pred_te = predict_labels(w_train, tx_test)
    y_pred_tr = predict_labels(w_train, tx_train)

    #accuracy
    accuracy_te = accuracy(y_pred_te,y_test)
    accuracy_tr = accuracy(y_pred_tr,y_train)
    return accuracy_te, accuracy_tr



# -------------------------------------- Cross validation using accuracy  ----------------------#
def cross_validation_demo_ridge_regression(x, y) :
    """Cfind the best hyperparameters for Ridge regression according to their accuracy"""
    #initialisation 
    seed = 1
    degrees = range(1,15)
    k_fold = 4
    lambdas = np.logspace(-5, 0, 10)
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the the accuracy
    accuracy_temp_te=[]
    accuracy_temp_tr=[]
    all_accuracy_lambda_te = []
    all_accuracy_lambda_tr = []
    best_accuracy_degree_te = []
    best_accuracy_degree_tr = []    
    best_accuracy = 0
    
    #Iteration on degrees 
    for degree in degrees:
        all_accuracy_lambda_te=[]
        all_accuracy_lambda_tr=[]
        
        #Iteration on lambdas
        for lambda_ in lambdas:
            accuracy_temp_te=[]
            accuracy_temp_tr=[]
            
            for k in range (k_fold):
                accuracy_te, accuracy_tr = cross_validation_accuracy_ridge(y, x, k_indices, k, lambda_, degree)
                accuracy_temp_te.append(accuracy_te)
                accuracy_temp_tr.append(accuracy_tr)
                
            #If it is a better accuracy than the previous one we take the best lambda with the according degree 
            if(np.mean(accuracy_temp_te) > best_accuracy):
                best_accuracy = np.mean(accuracy_temp_te)
                degree_optimal = degree
                lambda_optimal = lambda_
                
            #We store the mean across the k-fold for this lambda and this degree    
            all_accuracy_lambda_te.append(np.mean(accuracy_temp_te))
            all_accuracy_lambda_tr.append(np.mean(accuracy_temp_tr))
        
        # to be able to plot the accuracy across the degree we take the best acuracy by degree
        best_accuracy_degree_te.append(all_accuracy_lambda_te[np.argmax(all_accuracy_lambda_te)])
        best_accuracy_degree_tr.append(all_accuracy_lambda_tr[np.argmax(all_accuracy_lambda_te)])

    print("The optimal parameters for the ridge regression are lambda = %f and degree = %d for a mean accuracy across the k-fold of %f"%(lambda_optimal, degree_optimal, best_accuracy))
    cross_validation_visualization_ridge_regression(degrees, best_accuracy_degree_tr,best_accuracy_degree_te)
                                              
    return lambda_optimal, degree_optimal, best_accuracy_degree_te