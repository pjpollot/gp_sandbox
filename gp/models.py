from abc import abstractmethod
import numpy as np
from numpy import pi, exp, log, sqrt, identity
from numpy.linalg import solve, cholesky, inv
from scipy import rand
from scipy.stats import multivariate_normal

from .kernels import Kernel
from .sigmoids import Sigmoid

from .utils import extract_diagonal_matrix, hermite_quadrature

# Abstract GP class
class Abstract_GP:
    def __init__(self, kernel_function, epsilon, additional_parameters={}):
        self._d = kernel_function._d
        self._kernel = kernel_function
        self._eps = epsilon

        self._n = 0
        self._X = None
        self._y = None
        self._loglik = None
        self._chol = None

        self._param = kernel_function._param
        # Plus the additional parameters
        for key, value in additional_parameters.items():
            self._param[key] = value
        
        self._grad = dict()
        for key in additional_parameters:
            self._grad[key] = None

    @abstractmethod
    def set_param(self, param, verbose):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def fit_optimize(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


# The classic GP Regressor
class GPRegressor(Abstract_GP):
    """
    input_dim: the input dimension
    kernel_function: the kernel function
    noise: the noise parameter (sigma_n) in the regression
    epsilon: correction hyperparameter to avoid singular covariance matrix
    """
    def __init__(self, kernel_function:Kernel, noise=1e-10, epsilon=1e-10):
        super().__init__(kernel_function, epsilon, additional_parameters={'log_noise':log(noise)})
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

    def __init__(self, kernel_function:Kernel, sigmoid_function:Sigmoid, epsilon=1e-10):
        super().__init__(kernel_function, epsilon)
        self._sigmoid = sigmoid_function

        self._mode = None
        self._grad_mode_loglik = None
        self._sqrt_W = None
    
    def set_param(self, param, verbose=True):
        for key, value in param.items():
            if key in self._kernel._param:
                self._kernel._param[key] = value
                self._param[key] = value
            if verbose:
                print("GP's parameters successfully changed! the Regressor needs to be trained again.")

            return self._param
        

    def fit(self, X, y, laplace_approx_n_iter=10):
        self._n = len(X)
        self._X = X.copy()
        self._y = y.copy()

        # compute the covariance matrix and its gradient
        K = np.zeros((self._n, self._n))
        grad_K = dict()
        for key in self._param:
            grad_K[key] = np.zeros((self._n, self._n))
        for i in range(self._n):
            for j in range(self._n):
                K[i,j], g = self._kernel.evaluate(self._X[i,:], self._X[j,:], return_grad=True)
                K[j,i] = K[i,j]
                for key, value in g.items():
                    grad_K[key][i,j] = value
                    grad_K[key][j,i] = value

        # Laplace approximation using Newton algorithm
        f = np.zeros(self._n)
        W = np.zeros((self._n, self._n))
        sqrt_W = np.zeros((self._n, self._n))
        grad_loglik = np.zeros(self._n)
        for it in range(laplace_approx_n_iter):
            objective = 0
            # Compute the grad log likelihood and W
            for i in range(self._n):
                z = self._y[i]*f[i]
                s, lsp, lspp, lsppp = self._sigmoid.evaluate(z, return_log_derivatives=True)
                grad_loglik[i] = self._y[i]*lsp
                W[i,i] = -lspp
                sqrt_W[i,i] = sqrt(W[i,i])
                objective += log(s)
            # Compute the new f
            L = cholesky(identity(self._n)+sqrt_W @ K @ sqrt_W)
            b = W @ f + grad_loglik
            a = b - sqrt_W @ solve(L.T, solve(L, sqrt_W @ K @ b))
            f = K @ a
            objective -= np.dot(a,f)/2
         
        # END LOOP
        ## compute the log-marginal-likelihood
        self._loglik = 0
        third_derivatives_loglik = np.zeros(self._n)
        for i in range(self._n):
            z = self._y[i]*f[i]
            s, lsp, lspp, lsppp = self._sigmoid.evaluate(z, return_log_derivatives=True)
            grad_loglik[i] = self._y[i]*lsp
            W[i,i] = -lspp
            third_derivatives_loglik[i] = self._y[i]*lsppp
            sqrt_W[i,i] = sqrt(W[i,i])
            self._loglik += log(s)
        L = cholesky(identity(self._n)+sqrt_W @ K @ sqrt_W)
        b = W @ f + grad_loglik
        a = b - sqrt_W @ solve(L.T, solve(L, sqrt_W @ K @ b))
        self._loglik -= np.dot(a,f)/2
        for i in range(self._n):
            self._loglik -= log(L[i,i])
        # compute the gradient of the log-marginal
        R = sqrt_W @ solve(L.T, solve(L, sqrt_W))
        C = solve(L, sqrt_W @ K)
        s2 = -0.5*extract_diagonal_matrix(extract_diagonal_matrix(K) - extract_diagonal_matrix(C.T @ C)) @ third_derivatives_loglik 
        for key, G in grad_K.items():
            s1 = np.dot(a, G @ a)/2 - np.trace(R @ G)/2
            u = G @ grad_loglik
            s3 = u - K @ R @ u
            self._grad[key] = s1 + np.dot(s2, s3)
        ## as for the other attributes
        self._mode = f
        self._chol = L
        self._sqrt_W = sqrt_W
        self._grad_mode_loglik = grad_loglik
        return self
    
    def fit_optimize(self, X, y, lr=1e-2, n_iter=100, laplace_approx_n_iter=10, verbose=True):
        if verbose:
            print("# Starting optimization")

        for it in range(n_iter):
            self.fit(X, y, laplace_approx_n_iter)

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
        
        return self.fit(X, y, laplace_approx_n_iter)
    
    def predict(self, x, return_var=False, hermite_quad_deg=50):
        k = np.zeros(self._n)
        for i in range(self._n):
            k[i] = self._kernel.evaluate(self._X[i,:], x)
        
        mean = np.dot(k, self._grad_mode_loglik)
        
        v = solve(self._chol, self._sqrt_W @ k)
        prior_var = self._kernel.evaluate(x, x)

        var = prior_var - np.dot(v, v)

        proba = hermite_quadrature(
            func=self._sigmoid.evaluate,
            deg=hermite_quad_deg,
            mean=mean, 
            var=var
        )

        if return_var:
            return proba, var

        return proba