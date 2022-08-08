from numpy.polynomial.hermite import hermgauss
from numpy import pi, sqrt

INV_SQRT_PI = 1/sqrt(pi)

def hermite_quadrature(func, deg: int, mean=0., var=1.):
    u, w = hermgauss(deg)
    c = sqrt(2*var)

    I = 0
    for i in range(deg):
        I += INV_SQRT_PI*w[i]*func(mean+c*u[i])
    return I