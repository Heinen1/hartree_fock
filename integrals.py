import sys
#sys.setrecursionlimit(1500) # carefull with recursions!
import numpy as np
from scipy import special
from scipy.integrate import quad
from scipy.special import binom
from scipy import linalg

# nuclear charges per atom type
nuclear_charges = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8}

def parameters(bf1, bf2):
    """Return usefull parameters that are calculated from coordinates and exponents

    Args:
        bf1 (class): Basis function 1
        bf2 (class): Basis function 2

    Returns:
        floats: sum of exponents, product of exponents, reduced sum of exponents
    """
    alpha_sum = bf1.alpha + bf2.alpha
    alpha_product = bf1.alpha * bf2.alpha
    P = ((bf1.coordinates * bf1.alpha) + (bf2.coordinates * bf2.alpha)) / alpha_sum

    return alpha_sum, alpha_product, P

def kinetic_recursive(molecule):
    """Generate kinetic integral matrix of between two basis function

    Args:
        molecule (list): basis functions of primitive gaussians

    Returns:
        ndarray: kinetic integrals between basis functions
    """
    nbasis = len(molecule)
    Tmatrix = np.zeros([nbasis, nbasis])
    indices = np.array(range(3))

    for i in range(nbasis):
        for j in range(nbasis):
            nprimitives_i = len(molecule[i])
            nprimitives_j = len(molecule[j])
            
            for k in range(nprimitives_i):
                for l in range(nprimitives_j):
                    N1 = normalization(molecule[i][k].alpha, *molecule[i][k].angular)
                    N2 = normalization(molecule[j][l].alpha, *molecule[j][l].angular)

                    alpha_sum, alpha_product, P = parameters(molecule[i][k], molecule[j][l])

                    S = list()
                    for index in range(3):
                        S.append(E(
                            P[index],
                            molecule[i][k].coordinates[index],
                            molecule[j][l].coordinates[index],
                            alpha_sum,
                            molecule[i][k].angular[index],
                            molecule[j][l].angular[index])
                        )

                    T = list()
                    for index in range(3):
                        Sindex = indices[indices != index]
                        T.append(K(
                            P[index],
                            molecule[i][k].coordinates[index],
                            molecule[j][l].coordinates[index],
                            molecule[i][k].alpha,
                            molecule[j][l].alpha,
                            molecule[i][k].angular[index],
                            molecule[j][l].angular[index]) * S[Sindex[0]] * S[Sindex[1]]) 

                    dist = (np.sum(np.square(molecule[i][k].coordinates - molecule[j][l].coordinates)))
                    prefactor = np.exp(-(alpha_product/alpha_sum)*abs(dist))

                    Tmatrix[i, j] += N1 * N2 * prefactor * (np.pi / (alpha_sum))**1.5 * molecule[i][k].coeff * molecule[j][l].coeff * sum(T)
    return Tmatrix

def K(P, A, B, alpha, beta, a, b):
    """Recursive function for calculating kinetic integral

    Args:
        P (float): alpha * bf1 + beta * bf2 / (alpha + beta)
        A (float): coordinates of center of basis function A
        B (float): coordinates of center of basis function B
        alpha (float): exponent of basis function A
        beta (float): exponent of basis function B
        a (float): angular moment of basis function A
        b (float): angular moment of basis function B
        

    Returns:
        float: kinetic integral
    """
    p = alpha + beta
    if (a == 0) and (b == 0):
        return 2 * alpha * beta * E(P, A, B, p, 1, 1)
    elif (a == 0):
        return -alpha * b * E(P, A, B, p, 1, b - 1) + 2 * alpha * beta * E(P, A, B, p, 1, b + 1)
    elif (b == 0):
        return -a * beta * E(P, A, B, p, a - 1, 1) + 2 * alpha * beta * E(P, A, B, p, a + 1, 1)
    else:
        term1 = a * b * E(P, A, B, p, a - 1, b - 1)
        term2 = 2 * alpha * b * E(P, A, B, p, a + 1, b - 1)
        term3 = 2 * a * beta * E(P, A, B, p, a - 1, b + 1)
        term4 = 4 * alpha * beta * E(P, A, B, p, a + 1, b + 1)

        return 0.5*(term1 - term2 - term3 + term4)

def normalization(alpha, lx, ly, lz):
    """Get normalization constant for Gaussian basis function

    Args:
        alpha (float): exponent of Gaussian
        lx (float): angular momentum in x-direction
        ly (float): angular momentum in y-direction
        lz (float): angular momentum in z-direction

    Returns:
        float: normalization constant
    """
    prefactor = ((2 * alpha) / np.pi)**0.75
    norminator = (4 * alpha)**((lx + ly + lz)/2)
    
    # add abs() in argument to prevent negative values from arising
    fact_x = np.math.factorial(np.math.factorial(abs(2*lx - 1)))
    fact_y = np.math.factorial(np.math.factorial(abs(2*ly - 1)))
    fact_z = np.math.factorial(np.math.factorial(abs(2*lz - 1)))
    denominator = np.sqrt(fact_x * fact_y * fact_z)
    
    return prefactor * norminator / denominator

    
def overlap_recursive(molecule):
    """Calculate overlap matrix between Gaussian primitive basis functions.
    Math: S(i,j) = int_infty^+infty[ Xi Xj ] dV

    Args:
        molecule (list): basis functions of primitive gaussians

    Returns:
        ndarray: overlap matrix of nbasis functions from molecule
    """
    nbasis = len(molecule)
    
    Smatrix = np.zeros([nbasis, nbasis])
    
    for i in range(nbasis):
        for j in range(nbasis):
            
            nprimitives_i = len(molecule[i])
            nprimitives_j = len(molecule[j])
            
            for k in range(nprimitives_i):
                for l in range(nprimitives_j):
                    N1 = normalization(molecule[i][k].alpha, *molecule[i][k].angular)
                    N2 = normalization(molecule[j][l].alpha, *molecule[j][l].angular)

                    alpha_sum, alpha_product, P = parameters(molecule[i][k], molecule[j][l])
                    
                    S = 1
                    for index in range(3):
                        S *= E(P[index],
                        molecule[i][k].coordinates[index],
                        molecule[j][l].coordinates[index], 
                        alpha_sum, 
                        molecule[i][k].angular[index], 
                        molecule[j][l].angular[index])

                    dist = np.sum(np.square(molecule[i][k].coordinates - molecule[j][l].coordinates))
                    prefactor = np.exp(-(alpha_product/alpha_sum)*abs(dist)) * (np.pi / (alpha_sum))**1.5

                    Smatrix[i, j] += N1 * N2 * prefactor * molecule[i][k].coeff * molecule[j][l].coeff * S
    return Smatrix

def E(P, A, B, p, a, b):
    """Recursive relation for binomial expansion

    Args:
        P (float): alpha * bf1 + beta * bf2 / (alpha + beta)
        A (float): coordinates of center of basis function A
        B (float): coordinates of center of basis function B
        p (float): sum of exponents of basis functions A and B
        a (float): angular moment of basis function A
        b (float): angular moment of basis function B

    Returns:
        float: recursive overlap of gaussian basis function
    """
    if (a == 0) and (b == 0):
        return 1.0
    elif (a == 1 and b == 0):
        return -(A - P)
    elif (b == 0):
        return -(A - P) * E(P, A, B, p, a-1, 0) + ((a-1) / (2*p))*E(P, A, B, p, a-2, 0)
    else:
        return E(P, A, B, p, a+1, b-1) + (A - B) * E(P, A, B, p, a, b-1)

def nuclear_electron_recursive(molecule, atom_coords, atom_types):
    """Calculate nuclear-electron attraction integral matrix using Gaussian basis functions.

    Args:
        molecule (list): basis functions of primitive gaussians
        atom_coords (list): atomic coordinates
        Zlist (list): atom types for each atom in molecule

    Returns:
        ndarray: nuclear-repulsion integral matrix
    """
    nbasis = len(molecule)
    atoms = len(atom_coords)    
    VNeMatrix = np.zeros([nbasis, nbasis])
    Z_charge = [nuclear_charges[atom_type] for atom_type in atom_types]
    
    for atom in range(atoms):
        R = atom_coords[atom]
        for i in range(nbasis):
            for j in range(nbasis):
                nprimitives_i = len(molecule[i])
                nprimitives_j = len(molecule[j])
                
                for k in range(nprimitives_i):
                    for l in range(nprimitives_j):
                        N1 = normalization(molecule[i][k].alpha, *molecule[i][k].angular)
                        N2 = normalization(molecule[j][l].alpha, *molecule[j][l].angular)
                        alpha_sum, alpha_product, _ = parameters(molecule[i][k], molecule[j][l])

                        dist = np.sum(np.square(molecule[i][k].coordinates - molecule[j][l].coordinates))
                        prefactor = np.exp(-(alpha_product/alpha_sum)*dist)

                        vne_int, _ = quad(boys_integrand, 0, 1,  limit=500, epsabs=1e-10, args=(
                            molecule[i][k], molecule[j][l], R)
                        )

                        VNeMatrix[i, j] += -Z_charge[atom] * N1 * N2 * molecule[i][k].coeff * molecule[j][l].coeff  * vne_int * prefactor * (2 * np.pi) / alpha_sum
    return VNeMatrix

def boys_integrand(t, bf1, bf2, R):
    """Integrand to Boys functions F(t) = int_0^1 t^(2n) * exp(-x * t^2) dt

    Args:
        t (float): integrating variable
        bf1 (class): basis function 1
        bf2 (class): basis function 2
        R (ndarray): nuclear coordinates

    Returns:
        float: integrand
    """
    alpha = bf1.alpha
    beta = bf2.alpha
    A = bf1.coordinates
    B = bf2.coordinates
    alpha_sum = alpha + beta
    P = ((alpha * A) + (beta * B)) / alpha_sum
    dist = (np.sum(np.square(P - R)))
    
    n = 1
    for index in range(3):
        n *= N(P[index], A[index], B[index], alpha, beta, bf1.angular[index], bf2.angular[index], t, R[index])

    return np.exp(-alpha_sum*(t**2) * abs(dist)) * n


def N(P, A, B, alpha, beta, a, b, t, R):
    """Recursive function need to calculate the nuclear-electron integral

    Args:
        P (float): alpha * bf1 + beta * bf2 / (alpha + beta)
        A (float): coordinates of center of basis function A
        B (float): coordinates of center of basis function B
        alpha (float): exponent of basis function A
        beta (float): exponent of basis function B
        a (float): angular moment of basis function A
        b (float): angular moment of basis function B
        t (float): integrating variable of Boys function
        R (float): nuclear coordinate

    Returns:
        float: result of recusion
    """
    p = alpha + beta
    if (a == 0) and (b == 0):
       return 1.0
    elif (a == 1) and (b == 0):
        return -(A - P + ((t**2) * (P - R)))
    elif (b == 0):
        term1 = -(A - P + (t**2)*(P - R)) * N(P, A, B, alpha, beta, a - 1, 0, t, R)
        term2 = ((a-1) / (2*p)) * (1 - t**2) * N(P, A, B, alpha, beta, a - 2, 0, t, R)
        return term1 + term2
    else:
        term1 = N(P, A, B, alpha, beta, a + 1, b - 1, t, R)
        term2 = (A - B) * N(P, A, B, alpha, beta, a, b - 1, t, R)
        return term1 + term2

def electron_electron_repulsion(molecule):
    nbasis = len(molecule)   
    VeeMatrix = np.zeros([nbasis, nbasis, nbasis, nbasis])
    for i in range(nbasis):
        for j in range(nbasis):
            for k in range(nbasis):
                for l in range(nbasis):
                    nprimitives_i = len(molecule[i])
                    nprimitives_j = len(molecule[j])
                    nprimitives_k = len(molecule[k])
                    nprimitives_l = len(molecule[l])

                    #Imatrix = 0
                    for ii in range(nprimitives_i):
                        for jj in range(nprimitives_j):
                            for kk in range(nprimitives_k):
                                for ll in range(nprimitives_l):
                                    Ni = normalization(molecule[i][ii].alpha, *molecule[i][ii].angular)
                                    Nj = normalization(molecule[j][jj].alpha, *molecule[j][jj].angular)
                                    Nk = normalization(molecule[k][kk].alpha, *molecule[k][kk].angular)
                                    Nl = normalization(molecule[l][ll].alpha, *molecule[l][ll].angular)
                                    N = Ni * Nj * Nk * Nl
                                    cicjckcl = molecule[i][ii].coeff * molecule[j][jj].coeff * molecule[k][kk].coeff * molecule[l][ll].coeff
                                    alpha_sum_ij, alpha_product_ij, P = parameters(molecule[i][ii], molecule[j][jj])  
                                    alpha_sum_kl, alpha_product_kl, Q = parameters(molecule[k][kk], molecule[l][ll])  
                                    dist_ij = np.sum(np.square(molecule[i][ii].coordinates - molecule[j][jj].coordinates))
                                    dist_kl = np.sum(np.square(molecule[k][kk].coordinates - molecule[l][ll].coordinates))
                                    E_AB = np.exp(-(alpha_product_ij/alpha_sum_ij)*dist_ij)
                                    E_CD = np.exp(-(alpha_product_kl/alpha_sum_kl)*dist_kl)

                                    i1, j1, k1 = molecule[i][ii].angular
                                    i2, j2, k2 = molecule[j][jj].angular
                                    i3, j3, k3 = molecule[k][kk].angular
                                    i4, j4, k4 = molecule[l][ll].angular
                                    PAx, PAy, PAz = P - molecule[i][ii].coordinates
                                    PBx, PBy, PBz = P - molecule[j][jj].coordinates
                                    PCx, PCy, PCz = P - molecule[k][kk].coordinates
                                    PDx, PDy, PDz = P - molecule[l][ll].coordinates
                                    PQx, PQy, PQz = P - Q
                                    Imatrix = 0
                                    for t in range(i1 + i2 + 1):
                                        for u in range(j1 + j2 + 1):
                                            for v in range(k1 + k2 + 1):
                                                for t2 in range(i3 + i4 + 1):
                                                    for u2 in range(j3 + j4 + 1):
                                                        for v2 in range(k3 + k4 + 1):
                                                            E1 = mdE(i1, i2, E_AB, PAx, PBx, alpha_sum_ij, t)
                                                            E2 = mdE(j1, j2, E_AB, PAy, PBy, alpha_sum_ij, u)
                                                            E3 = mdE(k1, k2, E_AB, PAz, PBz, alpha_sum_ij, v)
                                                            E4 = mdE(i3, i4, E_CD, PCx, PDx, alpha_sum_kl, t2)
                                                            E5 = mdE(j3, j4, E_CD, PCy, PDy, alpha_sum_kl, u2)
                                                            E6 = mdE(k3, k4, E_CD, PCz, PDz, alpha_sum_kl, v2)
                                                            R1 = mdR(t + t2, u + u2, v + v2, 0, (alpha_sum_ij * alpha_sum_kl)/ (alpha_sum_ij + alpha_sum_kl), PQx, PQy, PQz)
                                                            Imatrix += E1*E2*E3*E4*E5*E6*R1 * (-1)**(t2 + u2 + v2)

                                    #Ix, abserror = quad(boys_integrand_vee, 0, 1,
                                    #args=(molecule[i][ii], molecule[j][jj], molecule[k][kk], molecule[l][ll], P, Q, alpha_sum_ij, alpha_sum_kl))
                                    prefactor = E_AB * E_CD * (2*(np.pi**2.5)) / (alpha_sum_ij * alpha_sum_kl * np.sqrt(alpha_sum_ij + alpha_sum_kl))

                                    VeeMatrix[i, j, k, l] += N * cicjckcl * prefactor * Imatrix
    return VeeMatrix


def boys(x, n):
    if x == 0:
        return 1.0 / (2 * n + 1)
    else:
        return special.gammainc(n + 0.5, x) *  special.gamma(n + 0.5) * (1.0 / (2 * x ** (n + 0.5)))
    

def mdR(t, u, v, n, eta, x, y, z):
    if t < 0 or u < 0 or v < 0 :
        return 0
    if t == 0 and u == 0 and v == 0:
        r = (x**2) + (y**2) + (z**2)
        return ((-2 * eta)**n) * boys(eta * r, n)
    elif t > 0:
        return (t - 1) * mdR(t - 2, u, v, n + 1, eta, x, y, z) + x * mdR(t - 1, u, v, n + 1, eta, x, y, z)
    elif u > 0:
        return (u - 1) * mdR(t, u - 2, v, n + 1, eta, x, y, z) + y * mdR(t, u - 1, v, n + 1, eta, x, y, z)
    elif v > 0:
        return (v - 1) * mdR(t, u, v - 2, n + 1, eta, x, y, z) + z * mdR(t, u, v -1, n + 1, eta, x, y, z)

def mdE(i, j, E_AB, X_PA, X_PB, alpha_sum, t):
    if (i < 0) or (j < 0) or (t < 0):
        return 0
    if (i == 0) and (j == 0):
        return 1
    if j > 0:
        term1 = (1 / (2 * alpha_sum)) * mdE(i, j - 1, E_AB, X_PA, X_PB, alpha_sum, 1)
        term2 = X_PB * mdE(i, j - 1, E_AB, X_PA, X_PB, alpha_sum, 1)
        term3 = (t + 1) * mdE(i, j - 1, E_AB, X_PA, X_PB, alpha_sum, 1)
        return term1 + term2 + term3  
    if j == 0:
        term1 = (1 / (2 * alpha_sum)) * mdE(i - 1, j, E_AB, X_PA, X_PB, alpha_sum, 1)
        term2 = X_PA * mdE(i - 1, j, E_AB, X_PA, X_PB, alpha_sum, 1)
        term3 = (t + 1) * mdE(i - 1, j, E_AB, X_PA, X_PB, alpha_sum, 1)
        return term1 + term2 + term3



def boys_integrand_vee(t, bf_a, bf_b, bf_c, bf_d, P, Q, p, q):
    IProduct = 1
    rho = (p * q) / (p + q)
    dist_PQ2 = np.sum(np.square(P - Q))
    for index in range(3):
        IProduct *= I(bf_a.angular[index],
                      bf_b.angular[index],
                      bf_c.angular[index],
                      bf_d.angular[index],
                      P[index],
                      Q[index],
                      bf_a.coordinates[index],
                      bf_b.coordinates[index],
                      bf_c.coordinates[index],
                      bf_d.coordinates[index],
                      p, q, t)
    integrand = IProduct * np.exp(-(rho * t**2) * dist_PQ2)

    return integrand

def I(a, b, c, d, P, Q, A, B, C, D, p, q, t):
    n = a + b
    m = c + d
    C00 = (P - A) - (q * ((P - Q) / (p + q)) * t**2)
    C00d = (P - B) - (q * ((P - Q) / (p + q)) * t**2)
    D00 = (Q - C) - (p * ((P - Q) / (p + q)) * t**2)
    D00d = (Q - D) - (p * ((P - Q) / (p + q)) * t**2)
    B00 = (t**2) / (2 * (p + q))
    B10 = (1 / (2 * p)) - ((q * t**2) / (2 * p * (p + q)))
    B01 = (1 / (2 * q)) - ((p * t**2) / (2 * q * (p + q)))
    
    I = np.ones([n + 1, m + 1])
    #I[0, 0] = (np.pi / np.sqrt()) * np.exp()

    # vertical recurrence relation
    I = I_vertical(I, n, m, C00, D00, B01, B10, B01)
    # horitzon recurrence relation
    I = I_horizontal(I, a, c, A-B, C-D)

    return I

def I_vertical(I, n, m, C00, D00, B00, B10, B01):
    if n > 0:
        I[1, 0] = C00 * I[0, 0]
    if m > 0:
        I[0, 1] = D00 * I[0, 0]
    
    for a in range(2, n+1):
        I[a, 0] = (a - 1) * B10 * I[a-2, 0] + C00 * I(a-1, 0)
    for b in range(2, m+1):
        I[0, b] = (b - 1) * B01 * I[0, b-2] + D00 * I(0, b-1)

    if (m == 0) or (n == 0):
        return I

    for a in range(1, n+1):
        I[a, 1] = (n - 1) * B10 * I[a-2, 1] + B00 * I[a-1, 0] + C00 * I[a-1, 1]
        for b in range(2, m+1):
            I[a, b] = (b - 1) * B01 * I[a, b-2] + a * B00 * I[a-1, b-1] + D00 * I[a, b-1]
    
    return I

def I_horizontal(I, i, k, AB, CD):
    ndim, mdim = I.shape
    j = ndim - i -1
    l = mdim - k - 1
    ijkl = 0

    for m in range(l+1):
        ijm0 =0
        for n in range(j+1):
            ijm0 += binom(j, n) * np.power(AB, j-n) * I[n+i, m+k]
        ijkl += binom(l, m) * np.power(CD, l-m) * ijm0
    
    return ijkl

def nuclear_nuclear_repulsion_energy(atom_coords, atom_types):
    """Nuclear repulsion energy

    Args:
        atom_coords (list): nuclear coordinates of atoms in molecule

    Returns:
        float: total nuclear repulsion energy (Ha)
    """
    n_atoms = len(atom_coords)
    E_NN = 0
    
    for i in range(n_atoms):
        Z_i = nuclear_charges[atom_types[i]]
        for j in range(i+1, len(atom_coords)):
                Z_j = nuclear_charges[atom_types[j]]
                R = np.sqrt(np.sum(np.square(atom_coords[i] - atom_coords[j])))
                E_NN += (Z_i * Z_j) / R
    return E_NN




