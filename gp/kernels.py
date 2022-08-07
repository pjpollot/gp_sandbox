from math import exp, log

"""
Kernels' abstract class
common methods:
-> evaluate: return the evaluation of the kernel, and if wanted, its gradient
"""
class Kernel:
    def __init__(self, input_dim, parameters):
        self._d = input_dim
        self._param = parameters

# A case in point
class RBF(Kernel):
    def __init__(self, input_dim, l=1., sigma=1.):
        parameters = {
            "log_l": log(l),
            "log_sigma": log(sigma) 
        }
        super().__init__(input_dim, parameters)
    
    def evaluate(self, x, y, return_grad=False):
        sqr_sum = 0
        for i in range(self._d):
            sqr_sum = (x[i]-y[i])**2
        
        ker = exp(2*self._param["log_sigma"]) *exp( - sqr_sum*exp(-2*self._param["log_l"])/2 )

        if not return_grad:
            return ker
        else:
            grad = {
                "log_l": sqr_sum*exp(-2*self._param["log_l"])*ker,
                "log_sigma":2*ker
            }
            return ker, grad