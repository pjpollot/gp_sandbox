import pytest
import numpy as np

from gp.kernels import RBF
from gp.models import GPBinaryClassifier
from gp.optimization import GradientDescentOptimizer

import sklearn.gaussian_process as skgp

def test_optimization():
    eps = 1e-2
    print("Optimization testing:")
    def f(x):
        val = x['x']**2 + x['y']**2
        grad = {'x':2*x['x'], 'y':2*x['y']}
        return val, grad
    
    x0 = {'x':1.2, 'y':-2.}

    gd = GradientDescentOptimizer(f)
    gd.minimize(x0)

    x_min = gd.get_result()
    assert abs(x_min['x']) < eps
    assert abs(x_min['y']) < eps

def test_RBF_kernel():
    eps = 1e-5

    n = 100
    p = 50
    X = np.random.uniform(size=(n,2))
    Y = np.random.uniform(size=(p,2))

    ker1 = RBF(l=1., sigma=1.)
    ker2 = skgp.kernels.ConstantKernel(constant_value=1.) * skgp.kernels.RBF(length_scale=1.)

    # Variance test
    K1 = ker1(X)
    K2 = ker2(X)
    assert (np.abs(K1-K2) < eps).all()

    # Covariance test
    cov1 = ker1(X, Y)
    cov2 = ker2(X, Y)
    (np.abs(cov1 - cov2) < eps).all()


def test_marginal_loglik():
    n_train = 100
    X_train = np.random.uniform(size=(n_train, 2))
    def separator(x):
        return 2*(x[1] > x[0]**2) - 1
    y_train = np.array([
        separator(x) for x in X_train
    ])

    eps = 1e-2

    gpc1 = GPBinaryClassifier(minimizer=None).fit(X_train, y_train)
    gpc2 = skgp.GaussianProcessClassifier(optimizer=None).fit(X_train, y_train)

    assert abs(gpc1.log_marginal_likelihood() - gpc2.log_marginal_likelihood()) < eps
