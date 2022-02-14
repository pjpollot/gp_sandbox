from math import exp

# Abstract class
class Kernel:
    def __init__(self, input_dim, parameters):
        self._d = input_dim
        self._param = parameters
    
    def evaluate(self, x, y, return_grad=False):
        return None

# A case in point
class RBF(Kernel):
    def __init__(self, input_dim, l=1, sigma=1):
        parameters = {
            "l": l,
            "sigma": sigma 
        }
        super().__init__(input_dim, parameters)
    
    def evaluate(self, x, y, return_grad=False):
        sqr_sum = 0
        for i in range(self._d):
            sqr_sum = (x[i]-y[i])**2
        
        ker = self._param["sigma"]**2 *exp( - sqr_sum/(2*self._param["l"]**2) )

        if not return_grad:
            return ker
        else:
            grad = {
                "l": sqr_sum*ker/self._param["l"]**3,
                "sigma": 2*ker/self._param["sigma"]
            }
            return ker, grad