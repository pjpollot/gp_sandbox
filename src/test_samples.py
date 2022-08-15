import pytest
import numpy as np

from gp.kernels import RBF
from gp.models import GPBinaryClassifier
from gp.optimization import GradientDescentOptimizer
from gp.sigmoids import Logistic

def test_compute_covariance_matrix():
    dim = 2
    X = np.random.uniform(size=(2,dim))
    Y = np.random.uniform(size=(3,dim))
    ker = RBF(dim)

    K1, grad =  ker.evaluate_matrix(X, Y, return_grad=True)
    K2 = ker.evaluate_matrix(X, Y, return_grad=False)

    assert K1.all()
    assert (K1 == K2).all()
    assert grad

    K1, grad =  ker.evaluate_matrix(X, return_grad=True)
    K2 = ker.evaluate_matrix(X, return_grad=False)

    assert K1.all()
    assert (K1 == K2).all()
    assert grad


def test_optimization():
    print("Optimization testing:")
    def f(x):
        val = x['x']**2 + x['y']**2
        grad = {'x':2*x['x'], 'y':2*x['y']}
        return val, grad
    
    x0 = {'x':1.2, 'y':-2.}
    gd = GradientDescentOptimizer(f)
    gd.minimize(x0)
    x_min = gd.get_result()
    assert abs(x_min['x']) < .01
    assert abs(x_min['y']) < .01
    

def test_GPC_fit_and_loglik():
    dim = 2
    X = np.random.uniform(size=(5,dim))
    y = np.array([1, -1, 1, 1, -1])
    ker = RBF(dim)
    logistic = Logistic()
    minimizer = GradientDescentOptimizer()
    gpc = GPBinaryClassifier(kernel_function=ker, sigmoid_function=logistic, minimizer=minimizer).fit(X, y)
    logZ, gradLogZ = gpc.log_marginal_likelihood()
    assert logZ is not None
    assert len(gradLogZ) > 0