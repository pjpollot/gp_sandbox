from abc import abstractmethod
import numpy as np

from .kernels import Kernel

from .utils import extract_diagonal_matrix, hermite_quadrature

# Abstract GP class
class Abstract_GP:
    def __init__(self, kernel_function:Kernel):
        self._d = kernel_function._d
        self._kernel = kernel_function

        self._n = 0
        self._X = None
        self._y = None
        self._loglik = None

    @abstractmethod
    def set_param(self, param, verbose):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass