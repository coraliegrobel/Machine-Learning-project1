# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt
import time


def cross_validation_visualization(lambds, mse_tr, mse_te, x_min, x_max, y_min, y_max):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    