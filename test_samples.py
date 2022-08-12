import pytest
import numpy as np

from gp.kernels import RBF

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
