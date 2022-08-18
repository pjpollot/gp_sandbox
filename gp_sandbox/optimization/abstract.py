from abc import ABCMeta, abstractmethod

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