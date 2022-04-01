# -*- coding: utf-8 -*-
"""Functions that we use to get our best model."""

from implementations import *
from proj1_helpers import *
from useful_functions import accuracy
from preprocessing import full_preprocessing

def extract_data():
    DATA_TRAIN_PATH = '../data/train.csv/train.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    DATA_TEST_PATH = '../data/test.csv/test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    return y, tX, ids, tX_test, ids_test

def compute_GD(y, tX, tX_test):
    w_initial = np.zeros(tX.shape[1])
    gamma = 0.39810717
    max_iters = 1000
    w_GD, _ = least_squares_GD(y, tX, w_initial, max_iters, gamma)
    y_pred_GD = predict_labels(w_GD, tX_test)
    return y_pred_GD

def compute_SGD(y, tX, tX_test):
    w_initial = np.zeros(tX.shape[1])
    gamma = 0.00316228
    max_iters = 2000
    w_SGD, _ = least_squares_SGD(y, tX, w_initial, max_iters, gamma)
    y_pred_SGD = predict_labels(w_SGD, tX_test)
    return y_pred_SGD

def compute_LS(y, tX, tX_test):
    degree = 12
    
    # form data with polynomial degree
    tX_train_poly = build_poly(tX, degree)
    tX_test_poly = build_poly(tX_test, degree)

    w_LS, _ = least_squares(y, tX_train_poly)
    y_pred_LS = predict_labels(w_LS, tX_test_poly)
    return y_pred_LS

def compute_RR(y, tX, tX_test):
    degree = 13
    lambda_ = 1e-5
    
    # form data with polynomial degree
    tX_train_poly = build_poly(tX, degree)
    tX_test_poly = build_poly(tX_test, degree)

    w_RR, _ = ridge_regression(y, tX_train_poly, lambda_)
    y_pred_RR = predict_labels(w_RR, tX_test_poly)
    return y_pred_RR

def compute_LR(y, tX, tX_test):
    max_iters = 1000
    gamma = 1e-5
    w_initial = np.zeros(tX.shape[1])

    w_LR, _ = logistic_regression(y, tX, w_initial, max_iters, gamma)
    y_pred_LR = predict_labels(w_LR, tX_test)
    return y_pred_LR

def compute_RLR(y, tX, tX_test):
    max_iters = 500
    gamma = 1e-2
    lambda_ = 1e-5
    w_initial = np.zeros(tX.shape[1])

    w_LR, _ = reg_logistic_regression(y, tX, lambda_ , w_initial, max_iters, gamma)
    y_pred_LR = predict_labels(w_LR, tX_test)
    return y_pred_LR


def vote(y_pred_GD, y_pred_SGD, y_pred_LS, y_pred_RR, y_pred_LR, y_pred_LRP):
    y_vote = y_pred_GD + y_pred_SGD + y_pred_LS + y_pred_RR + y_pred_LR + y_pred_LRP
    # y_pred_RR is the more accurate, so if the result of the vote is 0, we want to take its value
    y_pred = y_pred_RR.copy()
    y_pred[y_vote > 0] = 1
    y_pred[y_vote < 0] = -1
    return y_pred

def result(y_pred, ids_test):
    OUTPUT_PATH = '../data/votation_pred.csv'
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

if __name__ == "__main__":
    print("Start extracting and preprocessing the datas.")
    y, tX, ids, tX_test, ids_test = extract_data()

    tX_preprocessed, tX_test_preprocessed = full_preprocessing(tX, tX_test)

    #TODO : Delete those lines to pred the online datas.
    from useful_functions import split_data, build_poly
    tX_train_preprocessed, tX_test_preprocessed, y, y_te = split_data(tX_preprocessed, y, 0.8)

    print("Computing gradient descent.")
    y_pred_GD = compute_GD(y, tX_train_preprocessed, tX_test_preprocessed)
    print("Computing stochastic gradient descent.")
    y_pred_SGD = compute_SGD(y, tX_train_preprocessed, tX_test_preprocessed)
    print("Computing least square.")
    y_pred_LS = compute_LS(y, tX_train_preprocessed, tX_test_preprocessed)
    print("Computing ridge regression.")
    y_pred_RR = compute_RR(y, tX_train_preprocessed, tX_test_preprocessed)
    print("Computing logistic regression.")
    y_pred_LR = compute_LR(y, tX_train_preprocessed, tX_test_preprocessed)
    print("Computing regularized logistic regression.")
    y_pred_RLR = compute_RLR(y, tX_train_preprocessed, tX_test_preprocessed)

    acc_GD = accuracy(y_pred_GD, y_te)
    acc_SGD = accuracy(y_pred_SGD, y_te)
    acc_LS = accuracy(y_pred_LS, y_te)
    acc_RR = accuracy(y_pred_RR, y_te)
    acc_LR = accuracy(y_pred_LR, y_te)
    acc_RLR = accuracy(y_pred_RLR, y_te)

    print("Computing votations.")
    y_pred = vote(y_pred_GD, y_pred_SGD, y_pred_LS, y_pred_RR, y_pred_LR, y_pred_RLR)

    acc = accuracy(y_pred, y_te)
    
    print("acc_GD = {a}".format(a=acc_GD))
    print("acc_SGD = {a}".format(a=acc_SGD))
    print("acc_LS = {a}".format(a=acc_LS))
    print("acc_RR = {a}".format(a=acc_RR))
    print("acc_LR = {a}".format(a=acc_LR))
    print("acc_RLR = {a}".format(a=acc_RLR))
    print("accuracy = {a}".format(a=acc))

    print("Write result in a csv file.")
    result(y_pred, ids_test)