import numpy as np
from numpy import pi, exp, log, sqrt, identity
from numpy.linalg import solve, cholesky, inv
from scipy import rand
from scipy.stats import multivariate_normal

# Abstract GP class
class Abstract_GP:
    def __init__(self, kernel_function, epsilon):
        self._d = kernel_function._d
        self._kernel = kernel_function
        self._param = kernel_function._param
        self._eps = epsilon

        self._n = 0
        self._X = None
        self._y = None
        self._loglik = None

# The classic GP Regressor
class GPRegressor(Abstract_GP):
    """
    input_dim: the input dimension
    kernel_function: the kernel function
    noise: the noise parameter (sigma_n) in the regression
    epsilon: correction hyperparameter to avoid singular covariance matrix
    """
    def __init__(self, kernel_function, noise=1e-10, epsilon=1e-10):
        super().__init__(kernel_function, epsilon)
        self._param["log_noise"] = log(noise)

        self._grad = dict()
        for key in self._param:
            self._grad[key] = None        

        self._chol = None
        self._alpha = None

    def set_param(self, param, verbose=True):
        for key in self._kernel._param:
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
        self._X = X.copy()
        self._y = y.copy()
        
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
                    grad_M[key][j,i] = g[key] # By symmetry
            
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
    
    def fit_optimize(self, X, y, lr=1e-2, n_iter=100, verbose=True):

        if verbose:
            print("# Starting optimization")

        for it in range(n_iter):
            self.fit(X,y)

            if verbose:
                print(
                    "#{}: log_lik={}; parameters={}".format(
                        it+1, self._loglik, self._param
                    )
                )

            updated_param = dict()
            for key in self._param:
                updated_param[key] = self._param[key] + lr * self._grad[key]

            self.set_param(updated_param, verbose=False)
        
        return self.fit(X,y)

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

    def sample(self, X, size=1):
        p = len(X)
        mu, cov = self.predict(X, return_cov=True)
        cov += self._eps*identity(p) # adjustement factor

        return multivariate_normal.rvs(mean=mu, cov=cov, size=size)
"""
GP Binary Classifier
Output format: 1 and -1.
"""
class GPBinaryClassifier(Abstract_GP):
    def __init__(self, kernel_function, sigmoid_function, epsilon=1e-10):
        super().__init__(kernel_function, epsilon)
        self._sigmoid = sigmoid_function

        self._map = None

    def fit(self, X, y, laplace_approx_n_iter=100):
        self._n = len(X)
        self._X = X.copy()
        self._y = y.copy()

        # Compute the covariance matrix
        K = np.zeros((self._n, self._n))
        for i in range(self._n):
            for j in range(i+1):
                K[i,j] = self._kernel.evaluate(X[i,:], X[j,:])
                K[j,i] = K[i,j]

        # Laplace approximation using Newton approximation
        f = np.zeros(self._n)
        for it in range(laplace_approx_n_iter):
            # Compute W and its square root (supposing it is a positive diagonal matrix), and the gradient of the loglik p(y|f)
            W = np.zeros((self._n, self._n))
            grad_loglik = np.zeros(self._n)
            for i in range(self._n):
                z = self._y[i]*f[i]
                s, log_s_prime, log_s_second = self._sigmoid.evaluate(z, return_log_derivatives=True)
                W[i,i] =  -log_s_second
                grad_loglik[i] = self._y[i]*log_s_prime
            sqrt_W = sqrt(W)

            L = cholesky( identity(self._n) + sqrt_W @ K @ sqrt_W )
            b = W @ f + grad_loglik
            a = b - sqrt_W @ solve( L.T, solve(L, sqrt_W @ K @ b) )
            
            f = K @ a
        
        # derive the MAP
        self._map = f
        # and compute the approximate log-likelihood
        self._loglik = -f @ a
        for i in range(self._n):
            z = self._y[i]*f[i]
            s = self._sigmoid.evaluate(z)
            self._loglik += log(s) - log(L[i,i])
        return self