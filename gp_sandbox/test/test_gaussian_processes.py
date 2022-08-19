from cmath import log
from math import sqrt
import unittest
import numpy as np
import sklearn.gaussian_process as sgp

from scipy.special import expit
from scipy.linalg import cho_solve, cholesky
from numpy import log, sqrt, identity
from numpy.linalg import solve

from ..gaussian_processes import GPBinaryClassifier
from ..gaussian_processes.kernels import RBF
from ..gaussian_processes.sigmoids import Logistic

def inside_ball(x):
            norm = (x**2).sum()
            if norm < 1:
                return 1
            else:
                return -1

class GPBCTest(unittest.TestCase):
    def setUp(self):
        l, sigma = np.exp(np.random.uniform(low=-1e-5, high=1, size=2))

        self.kernel = RBF(l, sigma)
        self.sigmoid = Logistic()

        self.kernel_check = sgp.kernels.ConstantKernel(constant_value=sigma**2) * sgp.kernels.RBF(length_scale=l)

        self.dim = 2
        # build the training set
        self.n_train = 50

        self.X_train = np.random.uniform(size=(self.n_train, self.dim))
        self.y_train = np.array([
            inside_ball(x) for x in self.X_train
        ])

        # build the testing set
        self.n_test = 200

        self.X_test = np.random.uniform(size=(self.n_test, self.dim))
        self.y_test = np.array([
            inside_ball(x) for x in self.X_test
        ])
    
    def test_kernel(self):
        K = self.kernel(self.X_train)
        K_check = self.kernel_check(self.X_train)
        self.__assertMatrixAlmostEqual(K, K_check, n_decimals=6)
        
        cov = self.kernel(self.X_train, self.X_test)
        cov_check = self.kernel_check(self.X_train, self.X_test)
        self.__assertMatrixAlmostEqual(cov, cov_check, n_decimals=6)

    def test_likelihood(self):
        gpc = GPBinaryClassifier(kernel_function=self.kernel).fit(self.X_train, self.y_train, optimize_parameters=False)
        gpc_check = sgp.GaussianProcessClassifier(kernel=self.kernel_check, optimizer=None).fit(self.X_train, self.y_train)

        self.assertAlmostEqual(gpc.log_marginal_likelihood(), gpc_check.log_marginal_likelihood(), places=6, msg='log-marginal-likelihood are not equal')


    def __assertMatrixAlmostEqual(self, M1, M2, n_decimals):
        n, p = M1.shape
        for i in range(n):
            for j in range(p):
                self.assertAlmostEqual(M1[i,j], M2[i,j], n_decimals, 'Matrixes at index (%d,%d) are not equal' % (i, j))
    """
    def __assertArrayAlmostEqual(self, a1, a2, n_decimals):
        n = len(a1)
        for i in range(n):
            self.assertAlmostEqual(a1[i], a2[i], n_decimals, 'Arrays at index %d are not equal' % (i))
    """
