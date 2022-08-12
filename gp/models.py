from abc import ABCMeta, abstractmethod
import numpy as np

from .sigmoids import Sigmoid
from .kernels import Kernel
from .utils import extract_diagonal_matrix, hermite_quadrature

# Abstract GP class
class GP(metaclass=ABCMeta):
    def __init__(self, kernel_function: Kernel):
        self._d = kernel_function._d
        self._kernel = kernel_function
        
        self._X = None
        self._y = None

    def set_param(self, param):
        for key, value in param.items():
            if key in self._kernel._param:
                self._kernel._param[key] = value

    def get_param(self):
        return self._kernel._param

    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def log_marginal_likelihood(self, param):
        pass

    @abstractmethod
    def predict(self, X):
        pass

# GP Binary Classifier
class GPBinaryClassifier(GP):
    def __init__(self, kernel_function: Kernel, sigmoid_function: Sigmoid):
        super().__init__(kernel_function)
        self._sigmoid = sigmoid_function
        self._sqrt_W = None
        self._L = None

    def fit(self, X, y, laplace_n_iter=10, optim_n_iter=100):
        self._X = X.copy()
        self._y = y.copy()
        return self
        

