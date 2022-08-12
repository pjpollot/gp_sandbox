from abc import ABCMeta, abstractmethod
from math import exp, log
import numpy as np

"""
Kernel abstract class
"""
class Kernel(metaclass=ABCMeta):
    def __init__(self, input_dim, parameters):
        self._d = input_dim
        self._param = parameters

    @abstractmethod
    def evaluate(self, x, y, return_grad):
        pass
    
    def evaluate_matrix(self, X1, X2=None, return_grad=False):
        n = len(X1)
        if X2 is not None:
            p = len(X2)
            K = np.zeros((n,p))

            if return_grad:
                grad_K = dict()
                for key in self._param:
                    grad_K[key] = np.zeros((n,p))

            for i in range(n):
                for j in range(p):
                    if return_grad:
                        K[i,j], grad = self.evaluate(X1[i,:], X2[j,:], True)
                        for key, value in grad.items():
                            grad_K[key] = value
                    else:
                        K[i,j] = self.evaluate(X1[i,:], X2[j,:], False)
        else:
            # We only compute the covariance of X1
            K = np.zeros((n,n))

            if return_grad:
                grad_K = dict()
                for key in self._param:
                    grad_K[key] = np.zeros((n,n))

            for i in range(n):
                for j in range(i+1):
                    if return_grad:
                        K[i,j], grad = self.evaluate(X1[i, :], X1[j, :], True)
                        K[j,i] = K[i,j]
                        for key, value in grad.items():
                            grad_K[key][i,j] = value
                            grad_K[key][j,i] = value
                    else:
                        K[i,j] = self.evaluate(X1[i, :], X1[j, :], False)
                        K[j,i] = K[i,j]
            
        if return_grad:
            return K, grad_K
        return K
            


"""
A case in point, the Radial Basis Function
"""
class RBF(Kernel):
    def __init__(self, input_dim, l=1., sigma=1.):
        parameters = {
            "log_l": log(l),
            "log_sigma": log(sigma) 
        }
        super().__init__(input_dim, parameters)
    
    def evaluate(self, x, y, return_grad=False):
        sqr_sum = 0
        for i in range(self._d):
            sqr_sum = (x[i]-y[i])**2
        
        ker = exp(2*self._param["log_sigma"]) *exp( - sqr_sum*exp(-2*self._param["log_l"])/2 )

        if return_grad:
            grad = {
                "log_l": sqr_sum*exp(-2*self._param["log_l"])*ker,
                "log_sigma":2*ker
            }
            return ker, grad
        return ker