import numpy as np
from scipy.integrate import quad, fixed_quad, quadrature

def integrand(t, P, R, p):
    dist = np.sum(np.square(P - R))
    return (-0.4867 - 0.71843 * (t**2)) * np.exp(-p *dist * (t**2))

alpha = 0.3425250914E+01
beta = 5.033151319

A = np.array([0, 1.43233673, -0.96104039])
B = np.array([0, 0, 0.24026010])
P = (alpha * A + beta * B) / (alpha + beta)
p = alpha + beta
R = np.array([0, 1.43233673, -0.96104039])

quad(integrand, 0, 1, epsabs=1e-10, args=(P, R, p))
fixed_quad(integrand, 0, 1, args=(P, R, p))
quadrature(integrand, 0, 1, args=(P, R, p), tol=1e-10)