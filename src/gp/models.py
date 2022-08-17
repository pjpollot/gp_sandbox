from abc import ABCMeta, abstractmethod
import numpy as np
from numpy import identity
from numpy.linalg import solve, cholesky
from numpy import sqrt, log

from .optimization import GradientBasedOptimizer, GradientDescentOptimizer
from .sigmoids import Logistic, Sigmoid
from .kernels import RBF, Kernel
from .utils import extract_diagonal_matrix, hermite_quadrature

# Abstract GP class
class GP(metaclass=ABCMeta):
    def __init__(self, kernel_function: Kernel, minimizer: GradientBasedOptimizer):
        self._kernel = kernel_function
        self._minimizer = minimizer
        # the training set information
        self._n = 0
        self._X = None
        self._y = None
        # additional useful information
        self._L = None
    
    def get_param(self):
        return self._kernel.get_param()

    @abstractmethod
    def fit(self, X, y, verbose=True):
        self._n = len(y)
        self._X = X.copy()
        self._y = y.copy()
        return self
    
    @abstractmethod
    def log_marginal_likelihood(self, param, return_grad=False, verbose=False):
        pass

    @abstractmethod
    def predict(self, X):
        pass

# GP Binary Classifier
class GPBinaryClassifier(GP):
    def __init__(self, kernel_function=RBF(), sigmoid_function=Logistic(), minimizer=GradientDescentOptimizer()):
        super().__init__(kernel_function, minimizer)
        self._sigmoid = sigmoid_function
        self._sqrt_W = None
        self._grad_loglik = None

    def fit(self, X, y, laplace_n_iter=10, optim_n_iter=50, verbose=False):
        super().fit(X, y)
        # define the objective function and its gradient
        if self._minimizer is None:
            # If there is no optimizer, then there is no optimization
            return self
        else:
            def negative_loglik(param):
                logZ, gradLogZ = self.log_marginal_likelihood(param, laplace_n_iter, True, verbose)
                m_gradLogZ = dict()
                for key, value in gradLogZ.items():
                    m_gradLogZ[key] = -value
                return -logZ, m_gradLogZ
            # then minimize it
            self._minimizer.set_objective_and_gradient(negative_loglik).minimize(
                x0=self._kernel._param, 
                n_iter=optim_n_iter
            )
            return self
    
    def predict(self, X, return_var=False, hermite_quad_deg=10):
        p = len(X)
        k = self._kernel(self._X, X)
        mean = k.T @ self._grad_loglik
        v = solve(self._L, self._sqrt_W @ k)
        var = self._kernel(X) - v.T @ v
        proba = np.zeros(p)
        for i in range(p):
            proba[i] = hermite_quadrature(self._sigmoid, deg=hermite_quad_deg, mean=mean[i], var=var[i,i])
        if return_var:
            return proba, var
        return proba
    
    def log_marginal_likelihood(self, param=None, laplace_n_iter=10, return_grad=False, verbose=False):
        if param is not None:
            # change the parameters of the GP
            self._kernel.set_param(param)
        # compute the mode, the covariance and the gradient of the covariance
        f, K, grad_K = self.__compute_mode_cov_gradcov(laplace_n_iter)
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
        if verbose:
            print("log-likelihood=%.5f" % (logZ))
            #print("for parameters {} \n".format(self.get_param()))
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
    
    def __compute_mode_cov_gradcov(self, laplace_n_iter):
        f = np.zeros(self._n)
        # we first calculate the covariance matrix and its gradient
        K, grad_K = self._kernel(self._X, return_grad=True)
        # then we start the estimation of the mode using the Newton algorithm
        W = np.zeros((self._n, self._n))
        sqrt_W = np.zeros((self._n, self._n))
        L = np.zeros((self._n, self._n))
        grad_loglik = np.zeros(self._n)
        ## start loop
        for it in range(laplace_n_iter):
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