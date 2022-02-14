import numpy as np
from numpy import pi, log, sqrt
from numpy.linalg import solve, cholesky
from scipy import rand
from scipy.stats import multivariate_normal

# The classic GP Regressor
class GPRegressor:
    """
    input_dim: the input dimension
    kernel_function: the kernel function
    noise: the noise parameter (sigma_n) in the regression
    """
    def __init__(self, input_dim, kernel_function, noise=1e-10):
        self._d = input_dim
        self._kernel = kernel_function
        self._param = kernel_function._param
        self._param["noise"] = noise

        self._n = 0
        self._X = None
        self._y = None
        self._chol = None
        self._alpha = None
        self._loglik = None
        self._grad = None

    """
    TODO: Calculate the gradient of the log likelihood
    """
    def fit(self, X, y):
        self._n = len(X)
        self._X = X
        self._y = y
        
        # Calculate the matrix K + sigma_n * I
        M = np.zeros((self._n, self._n))
        for i in range(self._n):
            for j in range(i):
                M[i,j] = self._kernel.evaluate(X[i,:],X[j,:])
                M[j,i] = M[i,j]
            M[i,i] = self._kernel.evaluate(X[i,:],X[i,:]) + self._param['noise']**2
        ## Return its Cholesky matrix
        self._chol = cholesky(M)
        
        # Calculate alpha, a vector that simplifies the further calculations (prediction, etc.)
        z = solve(self._chol, y)
        self._alpha = solve(self._chol.T, z)

        # Calculate the log likelihood
        self._loglik = -(y.T @ self._alpha)/2 - self._n * log(2*pi)/2
        for i in range(self._n):
            self._loglik -= log(self._chol[i,i])

    
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
        
    """
    epsilon: positive adjustement factor for the covariance matrix to cope with singularity
    """
    def sample(self, X, size=1, epsilon=1e-10, random_state=None):
        f, cov = self.predict(X, return_cov=True)
        K = cov + epsilon*np.identity(len(X))
        return multivariate_normal(mean=f, cov=K).rvs(size=size, random_state=random_state)