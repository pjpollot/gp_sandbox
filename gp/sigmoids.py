from numpy import exp

"""
Sigmoid functions
one method
-> evaluate: evaluate the sigmoid function, can return the first and second derivative if wanted
"""

# The standard logistic function
class Logistic:
    def evaluate(self, z, return_log_derivatives=False):
        s = 1/(1+exp(-z))

        if return_log_derivatives:
            log_s_prime = 1-s
            log_s_second = -s*(1-s)
            return s, log_s_prime, log_s_second
        else:
            return s