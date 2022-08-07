from numpy import exp

"""
Sigmoid functions
one method
-> evaluate: evaluate the sigmoid function, can return the first and second derivative if wanted
"""

# The standard logistic function
class Logistic:
    def evaluate(z, return_derivatives=False):
        s = 1/(1+exp(-z))

        if return_derivatives:
            s_prime = s*(1-s)
            s_second = s_prime*(1-2*s)
            return s, s_prime, s_second
        
        return s