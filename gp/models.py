from abc import ABCMeta, abstractmethod
import numpy as np
from numpy import identity
from numpy.linalg import solve, cholesky
from numpy import sqrt, log

from .optimization import GradientBasedMinimizer
from .sigmoids import Sigmoid
from .kernels import Kernel
from .utils import extract_diagonal_matrix, hermite_quadrature

# Abstract GP class
class GP(metaclass=ABCMeta):
    def __init__(self, kernel_function: Kernel, minimizer: GradientBasedMinimizer):
        self._d = kernel_function._d
        self._kernel = kernel_function
        self._minimizer = minimizer
        
        self._n = 0
        self._X = None
        self._y = None

        self._sqrt_W = None
        self._L = None

    def set_param(self, param):
        for key, value in param.items():
            if key in self._kernel._param:
                self._kernel._param[key] = value

    def get_param(self):
        return self._kernel._param

    @abstractmethod
    def fit(self, X, y):
        self._n = len(y)
        self._X = X.copy()
        self._y = y.copy()
        return self
    
    @abstractmethod
    def log_marginal_likelihood(self, param):
        pass

    @abstractmethod
    def predict(self, X):
        pass

# GP Binary Classifier
class GPBinaryClassifier(GP):
    def __init__(self, kernel_function: Kernel, sigmoid_function: Sigmoid, minimizer: GradientBasedMinimizer):
        super().__init__(kernel_function, minimizer)
        self._sigmoid = sigmoid_function

    def fit(self, X, y, laplace_n_iter=10, optim_n_iter=100):
        super().fit(X, y)

        # define the objective function and its gradient
        def negative_loglik(param):
            logZ, gradLogZ = self.log_marginal_likelihood(param, laplace_n_iter)
            return -logZ, -gradLogZ
        
        self._minimizer.set_objective_and_gradient(negative_loglik).minimize(
            x0=self._kernel._param, 
            n_iter=optim_n_iter
        )
        return self
    
    def log_marginal_likelihood(self, param, laplace_n_iter=10):
        self.set_param(param)
        f, K, grad_K = self.__compute_mode_cov_gradcov(laplace_n_iter)
        
        W = np.zeros((self._n, self._n))
        self._sqrt_W = np.zeros((self._n, self._n))
        self._L = np.zeros((self._n, self._n))
        grad_loglik = np.zeros(self._n)
        grad3_loglik = np.zeros(self._n)
        loglik = 0
        ## compute W, its sqrt, the log-likelihood and its gradient and third derivatives
        for i in range(self._n):
            z = self._y[i]*f[i]
            s, lsp, lspp, lsppp = self._sigmoid.evaluate(z, return_log_derivatives=True)
            loglik += s
            grad_loglik[i] = self._y[i] * lsp
            W[i,i] = -lspp
            self._sqrt_W[i,i] = sqrt(W[i,i])
            grad3_loglik[i] = self._y[i] * lsppp
        # compute L, b and a
        self._L = cholesky( identity(self._n) + self._sqrt_W @ K @ self._sqrt_W )
        b = W @ f + grad_loglik
        a = b - self._sqrt_W @ solve(self._L.T, solve(self._L, self._sqrt_W @ K @ b))
        # then derive first the marginal log-likelihood
        logZ = -np.dot(a, f)/2 + loglik - log(self._L.diagonal()).sum()
        
        # as for the gradient of the marginal
        grad_logZ = dict()
        R = self._sqrt_W @ solve(self._L.T, solve(self._L, self._sqrt_W))
        C = solve(self._L, self._sqrt_W @ K)
        s2 = extract_diagonal_matrix(extract_diagonal_matrix(K) - extract_diagonal_matrix(C.T @ C)) @ grad3_loglik
        for parameter, dK in grad_K.items():
            s1 = (np.dot(a, dK @ a) - np.trace(R @ dK))/2
            q = dK @ grad_loglik
            s3 = q - K @ R @ q
            grad_logZ[parameter] = s1 + np.dot(s2, s3)
        
        return logZ, grad_logZ
    
    def __compute_mode_cov_gradcov(self, laplace_n_iter):
        f = np.zeros(self._n)

        K, grad_K = self._kernel.evaluate_matrix(self._X, return_grad=True)

        W = np.zeros((self._n, self._n))
        sqrt_W = np.zeros((self._n, self._n))
        L = np.zeros((self._n, self._n))
        grad_loglik = np.zeros(self._n)
        # start the Newton algorithm
        for it in range(laplace_n_iter):
            loglik = 0
            # compute W, its sqrt, the log-likelihood and its gradient
            for i in range(self._n):
                z = self._y[i]*f[i]
                s, lsp, lspp, lsppp = self._sigmoid.evaluate(z, return_log_derivatives=True)
                loglik += s
                grad_loglik[i] = self._y[i] * lsp
                W[i,i] = -lspp
                sqrt_W[i,i] = sqrt(W[i,i])
            # compute L, b and a
            L = cholesky( identity(self._n) + sqrt_W @ K @ sqrt_W )
            b = W @ f + grad_loglik
            a = b - sqrt_W @ solve(L.T, solve(L, sqrt_W @ K @ b))
            # then compute the next f
            f = K @ a
        # end loop
        return f, K, grad_K

