import numpy as np

from abc import ABCMeta, abstractmethod
from numpy import exp, log
from scipy.spatial.distance import pdist, cdist, squareform

"""
Kernel abstract class
"""
class Kernel(metaclass=ABCMeta):
    def __init__(self, parameters):
        self._param = parameters

    @abstractmethod
    def __call__(self, X, Y=None, return_grad=False):
        pass
    
    def set_param(self, param):
        for key, value in param.items():
            if key in self._param:
                self._param[key] = value

    def get_param(self):
        return self._param


"""
A case in point, the Radial Basis Function
"""
class RBF(Kernel):
    def __init__(self, l=1., sigma=1.):
        parameters = {
            "log_l": log(l),
            "log_sigma": log(sigma) 
        }
        super().__init__(parameters)
    
    def __call__(self, X, Y=None, return_grad=False):
        if Y is None:
            dists = squareform(pdist(X, metric='sqeuclidean'))
        else:
            dists = cdist(X, Y, metric='sqeuclidean')
        K = exp(2*self._param['log_sigma'] - exp(-2*self._param['log_l']) * dists/2)
        if return_grad:
            grad_K = dict()
            grad_K['log_l'] = exp(-2*self._param['log_l'])*np.multiply(dists, K)
            grad_K['log_sigma'] = 2*K
            return K, grad_K
        return K