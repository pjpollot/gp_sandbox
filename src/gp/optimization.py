from abc import ABCMeta, abstractmethod
import numpy as np

# ---------- ABSTRACT CLASSES ---------------

class Optimizer(metaclass=ABCMeta):
    def __init__(self):
        self._x_min = None
        self._f_min = None
    
    def get_result(self, return_objective=False):
        if return_objective:
            return self._f_min, self._x_min
        else:
            return self._x_min
    
    @abstractmethod
    def minimize(self, x0: dict, n_iter, verbose: bool):
        pass

class GradientBasedOptimizer(Optimizer):
    def __init__(self, obj_func_and_grad=None):
        super().__init__()
        self._obj_and_grad =obj_func_and_grad
        self._grad_min = None

    def set_objective_and_gradient(self, obj_func_and_grad):
        self._obj_and_grad = obj_func_and_grad
        return self

# ----------------------------------------------

class GradientDescentOptimizer(GradientBasedOptimizer):
    def __init__(self, obj_func_and_grad=None, learning_rate=.01):
        super().__init__(obj_func_and_grad)
        self._lr = learning_rate 
    
    def minimize(self, x0: dict, n_iter=100, verbose=False):
        self._x_min = x0.copy()
        for i in range(n_iter):
            self._f_min, self._grad_min = self._obj_and_grad(self._x_min)
            if verbose:
                print("objective function= %.4f" %(self._f_min))
            for key in self._x_min:
                self._x_min[key] -= self._lr * self._grad_min[key]