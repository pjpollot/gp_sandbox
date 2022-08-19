from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import identity
from numpy.linalg import solve, cholesky
from numpy import sqrt, log
from scipy.stats import multivariate_normal
from scipy.optimize import minimize

from .sigmoids import Logistic, Sigmoid
from .kernels import RBF, Kernel
from ..utils.misc import extract_diagonal_matrix, hermite_quadrature

# Abstract GP class
class GP(metaclass=ABCMeta):
    def __init__(self, kernel_function: Kernel):
        self._kernel = kernel_function
        # the training set information
        self._n = 0
        self._X = None
        self._y = None
        # additional useful information
        self._L = None
    
    def get_param(self):
        return self._kernel.get_param()

    def fit(self, X, y, optimize_parameters=True):
        self._n = len(y)
        self._X = X.copy()
        self._y = y.copy()
        if optimize_parameters:
            # define the objective function and its gradient
            def negative_loglik(param_vector):
                param = self.__vector_to_param(param_vector)
                logZ, gradLogZ = self.log_marginal_likelihood(param, True)
                m_gradLogZ = dict()
                for key, value in gradLogZ.items():
                    m_gradLogZ[key] = -value
                return -logZ, self.__param_to_vector(m_gradLogZ)
            # then minimize it
            param_vector0 = self.__param_to_vector(self._kernel.get_param())
            minimize(fun=negative_loglik, x0=param_vector0, method='BFGS', jac=True)
        return self
    
    @abstractmethod
    def log_marginal_likelihood(self, param, return_grad=False):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def sample(self, X, size=1):
        pass

    def __param_to_vector(self, param: dict):
        d = len(param)
        li = list(param.items())
        param_vector = np.zeros(d)
        for k in range(d):
            param_vector[k] = li[k][1]
        return param_vector

    def __vector_to_param(self, param_vector):
        param = dict()
        k = 0
        for key in self._kernel.get_param():
            param[key] = param_vector[k]
            k += 1
        return param

# GP Binary Classifier
class GPBinaryClassifier(GP):
    def __init__(self, kernel_function=RBF(), sigmoid_function=Logistic(), laplace_n_iter=10):
        super().__init__(kernel_function)
        self._sigmoid = sigmoid_function
        self._sqrt_W = None
        self._grad_loglik = None
        self._laplace_n_iter = laplace_n_iter

    
    def predict(self, X, return_mean_var=False, hermite_quad_deg=10):
        p = len(X)
        k = self._kernel(self._X, X)
        mean = k.T @ self._grad_loglik
        v = solve(self._L, self._sqrt_W @ k)
        var = self._kernel(X) - v.T @ v
        proba = np.zeros(p)
        for i in range(p):
            proba[i] = hermite_quadrature(self._sigmoid, deg=hermite_quad_deg, mean=mean[i], var=var[i,i])
        if return_mean_var:
            return proba, mean, var
        return proba

    def sample(self, X, size=1, return_f=False, epsilon=1e-4):
        _, mean, var = self.predict(X, return_mean_var=True)
        sigma = var + epsilon*identity(len(X))
        f_samples = multivariate_normal(mean=mean, cov=sigma).rvs(size=size)

        proba = self._sigmoid(f_samples)
        if return_f:
            return proba, f_samples
        return proba
    
    def log_marginal_likelihood(self, param=None, return_grad=False):
        if param is not None:
            # change the parameters of the GP
            self._kernel.set_param(param)
        # compute the mode, the covariance and the gradient of the covariance
        f, K, grad_K = self.__compute_mode_cov_gradcov()
        # start calculatations of the log marginal and its gradient
        W = np.zeros((self._n, self._n))
        self._sqrt_W = np.zeros((self._n, self._n))
        self._L = np.zeros((self._n, self._n))
        self._grad_loglik = np.zeros(self._n)
        grad3_loglik = np.zeros(self._n)
        loglik = 0
        ## compute W, its sqrt, the log-likelihood and its gradient and third derivatives
        for i in range(self._n):
            z = self._y[i]*f[i]
            s, lsp, lspp, lsppp = self._sigmoid(z, return_log_derivatives=True)
            loglik += log(s)
            self._grad_loglik[i] = self._y[i] * lsp
            W[i,i] = -lspp
            self._sqrt_W[i,i] = sqrt(W[i,i])
            grad3_loglik[i] = self._y[i] * lsppp
        # compute L, b and a
        self._L = cholesky( identity(self._n) + self._sqrt_W @ K @ self._sqrt_W )
        b = W @ f + self._grad_loglik
        a = b - self._sqrt_W @ solve(self._L.T, solve(self._L, self._sqrt_W @ K @ b))
        # then derive first the marginal log-likelihood
        logZ = -np.dot(a, f)/2 + loglik - log(self._L.diagonal()).sum()
        if return_grad:
            # as for the gradient of the marginal
            grad_logZ = dict()
            R = self._sqrt_W @ solve(self._L.T, solve(self._L, self._sqrt_W))
            C = solve(self._L, self._sqrt_W @ K)
            s2 = extract_diagonal_matrix(extract_diagonal_matrix(K) - extract_diagonal_matrix(C.T @ C)) @ grad3_loglik
            for parameter, dK in grad_K.items():
                s1 = (np.dot(a, dK @ a) - np.trace(R @ dK))/2
                q = dK @ self._grad_loglik
                s3 = q - K @ R @ q
                grad_logZ[parameter] = s1 + np.dot(s2, s3)
            return logZ, grad_logZ
        return logZ
    
    def __compute_mode_cov_gradcov(self):
        f = np.zeros(self._n)
        # we first calculate the covariance matrix and its gradient
        K, grad_K = self._kernel(self._X, return_grad=True)
        # then we start the estimation of the mode using the Newton algorithm
        W = np.zeros((self._n, self._n))
        sqrt_W = np.zeros((self._n, self._n))
        L = np.zeros((self._n, self._n))
        grad_loglik = np.zeros(self._n)
        ## start loop
        for it in range(self._laplace_n_iter):
            loglik = 0
            ## compute W, its sqrt, the log-likelihood and its gradient
            for i in range(self._n):
                z = self._y[i]*f[i]
                s, lsp, lspp, lsppp = self._sigmoid(z, return_log_derivatives=True)
                loglik += log(s)
                grad_loglik[i] = self._y[i] * lsp
                W[i,i] = -lspp
                sqrt_W[i,i] = sqrt(W[i,i])
            ## compute L, b and a
            L = cholesky( identity(self._n) + sqrt_W @ K @ sqrt_W )
            b = W @ f + grad_loglik
            a = b - sqrt_W @ solve(L.T, solve(L, sqrt_W @ K @ b))
            ## then compute the next f
            f = K @ a
        # end loop: return the variables of interest
        return f, K, grad_K