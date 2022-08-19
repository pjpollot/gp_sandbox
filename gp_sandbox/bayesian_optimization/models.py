from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
import scipy.optimize as opt
from sklearn.gaussian_process import GaussianProcessRegressor

from .acquisition_functions import expected_improvement

def constraints_all_satisfied(constraints_values) -> bool:
    for c in constraints_values:
        if c < 0:
            # ie == -1
            return False
    # end loop: everything is == 1
    return True

class BOModule(metaclass=ABCMeta):
    def __init__(self, black_box_objective_function, bounds: scipy.optimize.Bounds, black_box_constraints=None):
        self._x_min = None
        self._f_min = None

        self._X = None

        self._objective = black_box_objective_function
        self._objective_dataset = None

        self._n_constraints = 0
        if black_box_constraints is not None:
            self._n_constraints = len(black_box_constraints)
        self._constraints = black_box_constraints
        self._constraints_dataset = None
    
    def get_result(self, return_objective=False):
        if return_objective:
            return self._x_min, self._f_min
        else:
            return self._x_min
    
    @abstractmethod
    def acquisition_maximization(self):
        # The user must implement this part
        pass

    def minimize(self, X_init, objective_init_dataset, constraints_init_dataset=None, n_iter=100, verbose=True):
        self._X = X_init.copy()
        self._objective_dataset = objective_init_dataset.copy()
        if self._n_constraints > 0 and constraints_init_dataset is not None:
            self._constraints_dataset = constraints_init_dataset.copy()

        # update x_min and f_min
        self.__find_minimum_from_dataset()

        # BO iteration
        for it in range(n_iter):
            x_new = self.acquisition_maximization()
            f_new = self._objective(x_new)

            self._X = np.concatenate((self._X, [x_new]))
            self._objective_dataset = np.concatenate((self._objective_dataset, [f_new]))

            if self._n_constraints > 0:
                # if there are constraints
                constraints_new = np.array([constraint(x_new) for constraint in self._constraints])
                self._constraints_dataset = np.concatenate((self._constraints_dataset, [constraints_new]))

            if verbose:
                print('iteration %d: new objective evaluation=%.4f' % (it+1, f_new))

        self.__find_minimum_from_dataset()
    
    def __find_minimum_from_dataset(self):
        if self._objective_dataset is None:
            return None

        self._x_min = None
        self._f_min = None
        
        n = len(self._objective_dataset)
        
        # if there are constraints
        if self._n_constraints > 0:
            for k in range(n):
                if constraints_all_satisfied(self._constraints_dataset[k,:]):
                    if self._x_min == None or (self._objective_dataset[k] < self._f_min):
                        self._x_min = self._X[k,:].copy()
                        self._f_min = self._objective_dataset[k]
        else:
            # if there is no constraint
            self._x_min = self._X[0,:].copy()
            self._f_min = self._objective_dataset[0]
            for k in range(1,n):
                if self._objective_dataset[k] < self._f_min:
                    self._x_min = self._X[k,:].copy()
                    self._f_min = self._objective_dataset[k]

class EIAlgorithm(BOModule):
    def __init__(self, black_box_objective_function, bounds: scipy.optimize.Bounds):
        super().__init__(black_box_objective_function, bounds, None)
    
    def acquisition_maximization(self):
        gp = GaussianProcessRegressor().fit(self._X, self._objective_dataset)
        
        def function_to_minimize(x):
            X_to_pred = np.array([x])
            mean, std = gp.predict(X_to_pred, return_std=True)
            return -expected_improvement(self._f_min, mean, std)

        res = opt.minimize(fun=function_to_minimize, x0=self._x_min)
        return res.x
    
    def minimize(self, X_init, objective_init_dataset, n_iter=100, verbose=True):
        return super().minimize(X_init, objective_init_dataset, None, n_iter, verbose)