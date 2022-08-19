from distutils.log import warn
from logging import warning
import unittest
import numpy as np
import warnings

from scipy.optimize import Bounds

from ..bayesian_optimization import EIAlgorithm

# the minimum is 0 at [3, 0.5]
def norm2_function(x):
    return (x**2).sum()

def noisy_norm2_function(x):
    return norm2_function(x) + 1e-2 * np.random.normal()

class EITest(unittest.TestCase):
    def setUp(self):
        dim = 2
        n_init = 10

        self.X = np.random.uniform(low=-4, high=4, size=(n_init, dim))
        self.y = np.array([
            noisy_norm2_function(x) for x in self.X
        ])

        bounds = Bounds([-2 for k in range(dim)],[2 for k in range(dim)])
        self.bo = EIAlgorithm(noisy_norm2_function, bounds)
    
    def test_performance(self):
        warnings.filterwarnings('ignore')
        self.bo.minimize(self.X, self.y, verbose=False)
        x_min = self.bo.get_result()
        self.assertAlmostEqual(norm2_function(x_min), 0, places=2)