# -*- coding: utf-8 -*-
"""Functions that we use to get our best model."""

from implementations import *
from proj1_helpers import *
from preprocessing import full_preprocessing
from useful_functions import build_poly
    
def extract_data():
    DATA_TRAIN_PATH = '../data/train.csv/train.csv' # for mac '../data/train.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    DATA_TEST_PATH = '../data/test.csv/test.csv'# for mac '../data/test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    return y, tX, ids, tX_test, ids_test


def compute_RR(y, tX, tX_test):
    degree = 13
    lambda_ = 0.00001
    
    # form data with polynomial degree
    tX_train_poly = build_poly(tX, degree)
    tX_test_poly = build_poly(tX_test, degree)

    w_RR, _ = ridge_regression(y, tX_train_poly, lambda_)
    y_pred_RR = predict_labels(w_RR, tX_test_poly)
    return y_pred_RR


def result(y_pred, ids_test):
    OUTPUT_PATH = '../data/sample-submission.csv'
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

if __name__ == "__main__":
    print("Start extracting and preprocessing the datas.")
    y, tX, ids, tX_test, ids_test = extract_data()
    
    # Pre processing the data
    tX_preprocessed, tX_test_preprocessed = full_preprocessing(tX, tX_test)

    print("Computing ridge regression.")
    y_pred_RR = compute_RR(y, tX_preprocessed, tX_test_preprocessed)

    print("Write result in a csv file.")
    result(y_pred_RR, ids_test)
    
    