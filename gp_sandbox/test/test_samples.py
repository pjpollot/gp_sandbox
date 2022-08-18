import pytest
import numpy as np

from gaussian_processes.kernels import RBF
from gaussian_processes import GPBinaryClassifier
from optimization.gradient_based import GradientDescentOptimizer

import sklearn.gaussian_process as skgp

def test_optimization():
    eps = 1e-2
    print("Optimization testing:")
    def f(x):
        val = x['x']**2 + x['y']**2
        grad = {'x':2*x['x'], 'y':2*x['y']}
        return val, grad
    
    x0 = {'x':1.2, 'y':-2.}

    gd = GradientDescentOptimizer(f, learning_rate=.1)
    gd.minimize(x0)

    x_min = gd.get_result()
    assert f(x_min)[0] < eps

def test_RBF_kernel():
    eps = 1e-5

    n = 100
    p = 50
    X = np.random.uniform(size=(n,2))
    Y = np.random.uniform(size=(p,2))

    corr_length, sigma  = np.exp(np.random.normal(size=2))
    ker1 = RBF(l=corr_length, sigma=sigma)
    ker2 = skgp.kernels.ConstantKernel(constant_value=sigma**2) * skgp.kernels.RBF(length_scale=corr_length)

    # Variance test
    K1 = ker1(X)
    K2 = ker2(X)
    assert (np.abs(K1-K2) < eps).all()

    # Covariance test
    cov1 = ker1(X, Y)
    cov2 = ker2(X, Y)
    (np.abs(cov1 - cov2) < eps).all()


def generate_dataset(including_testing_set=False):
    def separator(x):
        return 2*(x[1] > x[0]**2) - 1

    n_train = 100
    X_train = np.random.uniform(size=(n_train, 2))
    y_train = np.array([
        separator(x) for x in X_train
    ])

    if including_testing_set:
        n_test = 500
        X_test = np.random.uniform(size=(n_test,2))
        y_test = np.array([
            separator(x) for x in X_test
        ])
        return X_train, X_test, y_train, y_test


    return X_train, y_train

"""
def test_marginal_loglik():
    X, y = generate_dataset()
    eps = 1e-2

    corr_length, sigma  = np.exp(np.random.normal(size=2))

    rbf1 = RBF(l=corr_length, sigma=1)
    gpc1 = GPBinaryClassifier(kernel_function=rbf1, minimizer=None).fit(X, y)

    rbf2 = skgp.kernels.ConstantKernel(constant_value=sigma**2) * skgp.kernels.RBF(length_scale=corr_length)
    gpc2 = skgp.GaussianProcessClassifier(optimizer=None).fit(X, y)

    assert abs(gpc1.log_marginal_likelihood() - gpc2.log_marginal_likelihood()) < eps
"""

def test_performance():
    X_train, X_test, y_train, y_test = generate_dataset(including_testing_set=True)
    threshold = 1e-1

    gpc = GPBinaryClassifier().fit(X_train, y_train)
    proba = gpc.predict(X_test)
    y_pred = 2*(proba>.5)-1
    
    n_error = 0
    n_test = len(y_test)
    for i in range(n_test):
        if y_test[i] != y_pred[i]:
            n_error += 1
    error_rate = (1.*n_error)/n_test
    assert error_rate <= threshold