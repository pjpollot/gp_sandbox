import pytest
import numpy as np

from gp.kernels import RBF
from gp.optimization import GradientDescentMinimizer

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
    gd = GradientDescentMinimizer(f)
    gd.optimize(x0)
    x_min = gd.get_result()
    assert abs(x_min['x']) < .01
    assert abs(x_min['y']) < .01
    