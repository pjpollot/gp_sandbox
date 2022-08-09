import numpy as np
from numpy import pi, exp, log, sqrt, identity
from numpy.linalg import solve, cholesky, inv
from scipy import rand
from scipy.stats import multivariate_normal

from .approximations import hermite_quadrature

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
        self._chol = None

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

    hermite_normalizer = 1/sqrt(pi)
    sqrt_2 = sqrt(2)

    def __init__(self, kernel_function, sigmoid_function, epsilon=1e-10):
        super().__init__(kernel_function, epsilon)
        self._sigmoid = sigmoid_function

        self._mode = None
        self._grad_mode_loglik = None
        self._sqrt_W = None

    def fit(self, X, y, laplace_approx_n_iter=10, verbose=False):
        self._n = len(X)
        self._X = X.copy()
        self._y = y.copy()

        # Compute the covariance matrix
        K = np.zeros((self._n, self._n))
        for i in range(self._n):
            for j in range(i+1):
                K[i,j] = self._kernel.evaluate(self._X[i,:], self._X[j,:])
                K[j,i] = K[i,j]

        # Laplace approximation using Newton algorithm
        f = 0. + np.zeros(self._n)
        W = np.zeros((self._n, self._n))
        sqrt_W = np.zeros((self._n, self._n))
        grad_loglik = np.zeros(self._n)
        for it in range(laplace_approx_n_iter):
            objective = 0
            # Compute the grad log likelihood and W
            for i in range(self._n):
                z = self._y[i]*f[i]
                s, lsp, lspp = self._sigmoid.evaluate(z, return_log_derivatives=True)
                grad_loglik[i] = self._y[i]*lsp
                W[i,i] = -lspp
                sqrt_W[i,i] = sqrt(W[i,i])
                objective += log(s)
            # Compute the new f
            L = cholesky(identity(self._n)+sqrt_W @ K @ sqrt_W)
            b = W @ f + grad_loglik
            a = b - sqrt_W @ solve(L.T, solve(L, sqrt_W @ K @ b))
            f = K @ a
            objective += -np.dot(a,f)/2
            if verbose:
                print(f'objective={objective}')
         
        # Compute the log-marginal-likelihood
        s_list = np.zeros(self._n)
        for i in range(self._n):
                z = self._y[i]*f[i]
                s_list[i], lsp, lspp = self._sigmoid.evaluate(z, return_log_derivatives=True)
                grad_loglik[i] = self._y[i]*lsp
                W[i,i] = -lspp
                sqrt_W[i,i] = sqrt(W[i,i])
        L = cholesky(identity(self._n)+sqrt_W @ K @ sqrt_W)
        b = W @ f + grad_loglik
        a = b - sqrt_W @ solve(L.T, solve(L, sqrt_W @ K @ b))

        self._loglik = -np.dot(a, f)/2
        for i in range(self._n):
            self._loglik += log(s_list[i]) - log(L[i,i])

        # Other attributions
        self._mode = f
        return self
    
    """
    def predict(self, x, hermite_quad_deg=10):
        k = np.zeros(self._n)
        for i in range(self._n):
            k[i] = self._kernel.evaluate(self._X[i,:], x)

        mean = np.dot(k, self._gmap)
        
        v = solve(self._chol, self._sqrt_W @ k)
        prior_var = self._kernel.evaluate(x, x)

        var = prior_var - np.dot(v, v)

        proba = hermite_quadrature(
            func=self._sigmoid.evaluate, 
            deg=hermite_quad_deg, 
            mean=mean, 
            var=var
        )

        return proba

    def sample(self, X, size=1, return_mean_cov=False):
        p = len(X)
        k = np.zeros((self._n, p))
        for i in range(self._n):
            for j in range(p):
                k[i,j] = self._kernel.evaluate(self._X[i,:], X[j,:])
        
        mean = k.T @ self._gmap
        
        V = solve(self._chol, self._sqrt_W @ k)

        prior_cov = np.zeros((p, p))
        for i in range(p):
            for j in range(i+1):
                prior_cov[i,j] = self._kernel.evaluate(X[i,:], X[j,:])
                prior_cov[j,i] = prior_cov[i,j]
        
        cov = prior_cov - V.T @ V

        samples = multivariate_normal.rvs(mean=mean, cov=cov, size=size)

        if return_mean_cov:
            return samples, mean, cov
        
        return samples
    """
        