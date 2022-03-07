import numpy as np
from numpy import pi, exp, log, sqrt
from numpy.linalg import solve, cholesky, inv
from scipy import rand
from scipy.stats import multivariate_normal

# The classic GP Regressor
class GPRegressor:
    """
    input_dim: the input dimension
    kernel_function: the kernel function
    noise: the noise parameter (sigma_n) in the regression
    epsilon: correction hyperparameter to avoid singular covariance matrix
    """
    def __init__(self, input_dim, kernel_function, noise=1e-10, epsilon=1e-10):
        self._d = input_dim
        self._kernel = kernel_function
        self._param = kernel_function._param
        self._param["log_noise"] = log(noise)
        self._eps = epsilon

        self._grad = dict()
        for key in self._param:
            self._grad[key] = None        

        self._n = 0
        self._X = None
        self._y = None
        self._chol = None
        self._alpha = None
        self._loglik = None

    def set_param(self, param, verbose=True):
        for key in self._kernel._param:
            print(key)
            if key in param:
                self._kernel._param[key] = param[key]
                self._param[key] = param[key]
        if "log_noise" in param:
            self._param["log_noise"] = param["log_noise"]

        self._grad = dict()
        for key in self._param:
            self._grad[key] = None  

        self._n = 0
        self._X = None
        self._y = None
        self._chol = None
        self._alpha = None
        self._loglik = None

        if verbose:
            print("GP's parameters successfully changed! the Regressor needs to be trained again.")
        return self._param
    
    def fit(self, X, y):
        self._n = len(X)
        self._X = X
        self._y = y
        
        # Calculate the matrix K + sigma_n * I and its gradient
        M = np.zeros((self._n, self._n))
        grad_M = dict()
        for key in self._param:
            grad_M[key] = np.zeros((self._n,self._n))

        for i in range(self._n):
            for j in range(i+1):
                M[i,j], g = self._kernel.evaluate(X[i,:],X[j,:], return_grad=True)
                M[j,i] = M[i,j]
                # Additionally, if we're in the diagonal
                if i==j:
                    M[i,i] += exp(2*self._param["log_noise"]) + self._eps
                    g["log_noise"] = 2*exp(2*self._param["log_noise"])
                else:
                    g["log_noise"] = 0
                
                for key in self._param:
                    grad_M[key][i,j] = g[key]
            
        ## Return its Cholesky matrix
        self._chol = cholesky(M)
        
        # Calculate alpha, a vector that simplifies the further calculations (prediction, etc.)
        z = solve(self._chol, y)
        self._alpha = solve(self._chol.T, z)

        # Calculate the log-likelihood
        self._loglik = -(y.T @ self._alpha)/2 - self._n * log(2*pi)/2
        for i in range(self._n):
            self._loglik -= log(self._chol[i,i])

        # Calculate the gradient of the log-likelihood
        inv_M = inv(M)

        A = np.zeros((self._n, self._n))
        for i in range(self._n):
            for j in range(i+1):
                A[i,j] = self._alpha[i]*self._alpha[j]
                A[j,i] = A[i,j]

        for key in self._param:
            Q = (A - inv_M) @ grad_M[key]
            self._grad[key] = np.trace(Q)/2
        
        return self


    """
    return_cov: return the covariance over the predictions
    return_std: return the standard deviation over the predictions
    """
    def predict(self, X, return_cov=False, return_std=False):
        p = len(X)

        # Calculate the matrix K(X,X*)
        k = np.zeros((self._n,p))
        for i in range(self._n):
            for j in range(p):
                k[i,j] = self._kernel.evaluate(self._X[i,:],X[j,:])

        # Get the posterior mean
        f = k.T @ self._alpha

        if return_cov or return_std:
            # Calculate the matrix K(X*,X*)
            K = np.zeros((p,p))
            for i in range(p):
                for j in range(i):
                    K[i,j] = self._kernel.evaluate(X[i,:],X[j,:])
                    K[j,i] = K[i,j]
                K[i,i] = self._kernel.evaluate(X[i,:],X[i,:])

            v = solve(self._chol, k)
            # Get the posterior covariance
            cov = K - (v.T @ v)
            if return_cov:
                return f, cov
            else:
                return f, sqrt(np.diag(cov))
        else:
            return f