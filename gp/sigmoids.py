from numpy import exp

"""
Sigmoid functions
two static methods:
-> evaluate: evaluate the function
-> derivative: evaluate the derivative of the function
"""

# The standard logistic function
class Logistic:
    @staticmethod
    def evaluate(z):
        return 1/(1+exp(-z))

    @staticmethod
    def derivative(z):
        e = exp(z)
        return e/(1+e)**2