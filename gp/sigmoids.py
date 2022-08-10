from abc import abstractmethod
from numpy import exp

class Sigmoid:
    @abstractmethod
    def evaluate(self, z, return_log_derivatives):
        pass

# The standard logistic function
class Logistic(Sigmoid):
    def evaluate(self, z, return_log_derivatives=False):
        s = 1/(1+exp(-z))

        if return_log_derivatives:
            log_s_prime = 1-s
            log_s_second = -log_s_prime * s # or equivalently s*(s-1)
            log_s_third = log_s_second * (1-2*s)  # or equivalently s*(s-1)*(1-2*s)
            return s, log_s_prime, log_s_second, log_s_third
        
        return s