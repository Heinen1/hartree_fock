import numpy as np
from scipy import special
from scipy.integrate import quad
from scipy.special import binom
from scipy import linalg

def kinetic_recursive(molecule):
    # hm = higher momentum
    nbasis = len(molecule)
    
    T = np.zeros([nbasis, nbasis])
    
    for i in range(nbasis):
        for j in range(nbasis):
            
            nprimitives_i = len(molecule[i])
            nprimitives_j = len(molecule[j])
            
            for k in range(nprimitives_i):
                for l in range(nprimitives_j):
                    ax, ay, az = molecule[i][k].angular
                    bx, by, bz = molecule[j][l].angular

                    N1 = normalization(molecule[i][k].alpha, ax, ay, az)
                    N2 = normalization(molecule[j][l].alpha, bx, by, bz)
                    N = N1 * N2

                    p = (molecule[i][k].coordinates * molecule[i][k].alpha) + (molecule[j][l].coordinates * molecule[j][l].alpha)
                    P = p / (molecule[i][k].alpha + molecule[j][l].alpha)
                    
                    sx = E(P[0], molecule[i][k].coordinates[0], molecule[j][l].coordinates[0], molecule[i][k].alpha, molecule[j][l].alpha, ax, bx)
                    sy = E(P[1], molecule[i][k].coordinates[1], molecule[j][l].coordinates[1], molecule[i][k].alpha, molecule[j][l].alpha, ay, by)
                    sz = E(P[2], molecule[i][k].coordinates[2], molecule[j][l].coordinates[2], molecule[i][k].alpha, molecule[j][l].alpha, az, bz)

                    kx = K(P[0], molecule[i][k].coordinates[0], molecule[j][l].coordinates[0], molecule[i][k].alpha, molecule[j][l].alpha, ax, bx) * sy * sz
                    ky = K(P[1], molecule[i][k].coordinates[1], molecule[j][l].coordinates[1], molecule[i][k].alpha, molecule[j][l].alpha, ay, by) * sx * sz
                    kz = K(P[2], molecule[i][k].coordinates[2], molecule[j][l].coordinates[2], molecule[i][k].alpha, molecule[j][l].alpha, az, bz) * sx * sy
                    
                    sum_alpha = molecule[i][k].alpha + molecule[j][l].alpha
                    multiple_alpha = molecule[i][k].alpha * molecule[j][l].alpha
                    dist = (np.sum(np.square(molecule[i][k].coordinates - molecule[j][l].coordinates)))
                    prefactor = np.exp(-(multiple_alpha/sum_alpha)*abs(dist))

                    T[i, j] += N * prefactor * (np.pi / (sum_alpha))**1.5 * molecule[i][k].coeff * molecule[j][l].coeff * (kx + ky + kz)
    return T

def K(P, A, B, alpha, beta, a, b):
    if (a == 0) and (b == 0):
        return 2 * alpha * beta * E(P, A, B, alpha, beta, 1, 1)
    elif (a == 0):
        return -alpha * b * E(P, A, B, alpha, beta, 1, b - 1) + 2 * alpha * beta * E(P, A, B, alpha, beta, 1, b + 1)
    elif (b == 0):
        return -a * beta * E(P, A, B, alpha, beta, a - 1, 1) + 2 * alpha * beta * E(P, A, B, alpha, beta, a + 1, 1)
    else:
        term1 = a * b * E(P, A, B, alpha, beta, a - 1, b - 1)
        term2 = 2 * alpha * b * E(P, A, B, alpha, beta, a + 1, b - 1)
        term3 = 2 * a * beta * E(P, A, B, alpha, beta, a - 1, b + 1)
        term4 = 4 * alpha * beta * E(P, A, B, alpha, beta, a + 1, b + 1)

        return 0.5*(term1 - term2 - term3 + term4)



def overlap(molecule):
    
    nbasis = len(molecule)
    
    S = np.zeros([nbasis, nbasis])
    
    for i in range(nbasis):
        for j in range(nbasis):
            
            nprimitives_i = len(molecule[i])
            nprimitives_j = len(molecule[j])
            
            for k in range(nprimitives_i):
                for l in range(nprimitives_j):
                    
                    N = molecule[i][k].A * molecule[j][l].A
                    p = molecule[i][k].alpha + molecule[j][l].alpha
                    q = molecule[i][k].alpha * molecule[j][l].alpha / p
                    Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                    Q2 = np.dot(Q,Q)  # R^2
                    
                    # overlap is between basis functions. Each basis set functio
                    # conists of multiple primitives (so we need to add them)
                    # due to summation, the integral can be split (and added)
                    S[i,j] += N * molecule[i][k].coeff * molecule[j][l].coeff * np.exp(-q*Q2) * (np.pi/p)**(3/2) 
    
    return S

def overlap_integrand(r, p):
    return np.square(r) * np.exp(-p * np.square(r))

def overlap_numerical(molecule):
    nbasis = len(molecule)
    
    S = np.zeros([nbasis, nbasis])
    
    for i in range(nbasis):
        for j in range(nbasis):
            
            nprimitives_i = len(molecule[i])
            nprimitives_j = len(molecule[j])
            
            for k in range(nprimitives_i):
                for l in range(nprimitives_j):
                    N = molecule[i][k].A * molecule[j][l].A
                    p = molecule[i][k].alpha + molecule[j][l].alpha
                    q = molecule[i][k].alpha * molecule[j][l].alpha / p
                    Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                    Q2 = np.dot(Q,Q)
                    K = np.exp(-q*Q2)
                    r = quad(overlap_integrand, 0, np.inf, args=(p))
                    
                    S[i, j] += N * molecule[i][k].coeff * molecule[j][l].coeff * 4 * np.pi * K * r[0]
    return S

def normalization(alpha, lx, ly, lz):
    prefactor = ((2 * alpha) / np.pi)**0.75
    norminator = (4 * alpha)**((lx + ly + lz)/2)
    
    # add abs() in argument to prevent negative values from arising
    fact_x = np.math.factorial(np.math.factorial(abs(2*lx - 1)))
    fact_y = np.math.factorial(np.math.factorial(abs(2*ly - 1)))
    fact_z = np.math.factorial(np.math.factorial(abs(2*lz - 1)))
    denominator = np.sqrt(fact_x * fact_y * fact_z)
    
    return prefactor * norminator / denominator

    
def overlap_hm_recursive(molecule):
    # hm = higher momentum
    nbasis = len(molecule)
    
    S = np.zeros([nbasis, nbasis])
    
    for i in range(nbasis):
        for j in range(nbasis):
            
            nprimitives_i = len(molecule[i])
            nprimitives_j = len(molecule[j])
            
            for k in range(nprimitives_i):
                for l in range(nprimitives_j):
                    ax, ay, az = molecule[i][k].angular
                    bx, by, bz = molecule[j][l].angular

                    N1 = normalization(molecule[i][k].alpha, ax, ay, az)
                    N2 = normalization(molecule[j][l].alpha, bx, by, bz)
                    N = N1 * N2

                    p = (molecule[i][k].coordinates * molecule[i][k].alpha) + (molecule[j][l].coordinates * molecule[j][l].alpha)
                    P = p / (molecule[i][k].alpha + molecule[j][l].alpha)
                    
                    Sx = E(P[0], molecule[i][k].coordinates[0], molecule[j][l].coordinates[0], molecule[i][k].alpha, molecule[j][l].alpha, ax, bx)
                    Sy = E(P[1], molecule[i][k].coordinates[1], molecule[j][l].coordinates[1], molecule[i][k].alpha, molecule[j][l].alpha, ay, by)
                    Sz = E(P[2], molecule[i][k].coordinates[2], molecule[j][l].coordinates[2], molecule[i][k].alpha, molecule[j][l].alpha, az, bz)
                    
                    sum_alpha = molecule[i][k].alpha + molecule[j][l].alpha
                    multiple_alpha = molecule[i][k].alpha * molecule[j][l].alpha
                    dist = (np.sum(np.square(molecule[i][k].coordinates - molecule[j][l].coordinates)))
                    prefactor = np.exp(-(multiple_alpha/sum_alpha)*abs(dist))

                    S[i, j] += N * prefactor * (np.pi / (sum_alpha))**1.5 * molecule[i][k].coeff * molecule[j][l].coeff * Sx * Sy * Sz
    return S

def E(P, A, B, alpha, beta, a, b):
    if (a == 0) and (b == 0):
        return 1.0
    elif (a == 1 and b == 0):
        return -(A - P)
    elif (b == 0):
        return -(A - P) * E(P, A, B, alpha, beta, a-1, 0) + ((a-1) / (2*(alpha + beta)))*E(P, A, B, alpha, beta, a-2, 0)
    else:
        return E(P, A, B, alpha, beta, a+1, b-1) + (A - B) * E(P, A, B, alpha, beta, a, b-1)

def overlap_hm(molecule):
    # hm = higher momentum
    nbasis = len(molecule)
    
    S = np.zeros([nbasis, nbasis])
    
    for i in range(nbasis):
        for j in range(nbasis):
            
            nprimitives_i = len(molecule[i])
            nprimitives_j = len(molecule[j])
            
            for k in range(nprimitives_i):
                for l in range(nprimitives_j):
                    ax, ay, az = molecule[i][k].angular
                    bx, by, bz = molecule[j][l].angular

                    N1 = normalization(molecule[i][k].alpha, ax, ay, az)
                    N2 = normalization(molecule[j][l].alpha, bx, by, bz)
                    N = N1 * N2

                    p = (molecule[i][k].coordinates * molecule[i][k].alpha) + (molecule[j][l].coordinates * molecule[j][l].alpha)
                    P = p / (molecule[i][k].alpha + molecule[j][l].alpha)

                    Sx = binomial_expansion(P[0], molecule[i][k].coordinates[0], molecule[j][l].coordinates[0], molecule[i][k].alpha, molecule[j][l].alpha, ax, bx)
                    Sy = binomial_expansion(P[1], molecule[i][k].coordinates[1], molecule[j][l].coordinates[1], molecule[i][k].alpha, molecule[j][l].alpha, ay, by)
                    Sz = binomial_expansion(P[2], molecule[i][k].coordinates[2], molecule[j][l].coordinates[2], molecule[i][k].alpha, molecule[j][l].alpha, az, bz)
                    
                    sum_alpha = molecule[i][k].alpha + molecule[j][l].alpha
                    multiple_alpha = molecule[i][k].alpha * molecule[j][l].alpha
                    dist = (np.sum(np.square(molecule[i][k].coordinates - molecule[j][l].coordinates)))
                    prefactor = np.exp(-(multiple_alpha/sum_alpha)*abs(dist))

                    S[i, j] += N * prefactor * (np.pi / (sum_alpha))**1.5 * molecule[i][k].coeff * molecule[j][l].coeff * Sx * Sy * Sz
    return S

def binomial_expansion(P, A, B, alpha, beta, a, b):
    result = 0.0
    
    # add +1 to range to ensure that loop starts
    for i in range(a+1):
        for j in range(b+1):
            df = np.math.factorial(np.math.factorial(abs(i+j-1)))
            db = binom(a, i)*binom(b, j)
            product = np.power(P-A, a-i) * np.power(P-B, b-j)
            result += df*db*product/np.power(2*(alpha + beta), (i+j)/2)
    return result
    

def kinetic(molecule):
    nbasis = len(molecule)
    
    T = np.zeros([nbasis, nbasis])
    
    for i in range(nbasis):
        for j in range(nbasis):
            n_primitives_i = len(molecule[i])
            n_primitives_j = len(molecule[j])
            
            for k in range(n_primitives_i):
                for l in range(n_primitives_j):
                    
                    c1c2 = molecule[i][k].coeff * molecule[j][l].coeff
                    N = molecule[i][k].A * molecule[j][l].A
                    p = molecule[i][k].alpha + molecule[j][l].alpha
                    q = molecule[i][k].alpha * molecule[j][l].alpha / p
                    Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                    Q2 = np.dot(Q, Q)
                    
                    P = molecule[i][k].alpha * molecule[i][k].coordinates  + molecule[j][l].alpha  * molecule[j][l].coordinates 
                    Pp = P / p
                    PG = Pp - molecule[j][l].coordinates
                    PGx2 = PG[0] * PG[0]
                    PGy2 = PG[1] * PG[1]
                    PGz2 = PG[2] * PG[2]
                    
                    s = N * c1c2 * np.exp(-q * Q2) * (np.pi / p) ** 1.5
                    
                    T[i, j] += 3 * molecule[j][l].alpha * s
                    T[i, j] -= 2 * molecule[j][l].alpha * molecule[j][l].alpha * s * (PGx2 + 0.5 / p)
                    T[i, j] -= 2 * molecule[j][l].alpha * molecule[j][l].alpha * s * (PGy2 + 0.5 / p)
                    T[i, j] -= 2 * molecule[j][l].alpha * molecule[j][l].alpha * s * (PGz2 + 0.5 / p)
    return T

def boys(x, n):
    if x == 0:
        return 1.0 / (2 * n + 1)
    else:
        return special.gammainc(n + 0.5, x) *  special.gamma(n + 0.5) * (1.0 / (2 * x ** (n + 0.5)))
    
def electron_nuclear_attraction(molecule, atom_coordinates, Z):
    natoms = len(Z)
    nbasis= len(molecule)
    
    V_ne = np.zeros([nbasis, nbasis])
    
    for atom in range(natoms):
        for i in range(nbasis):
            for j in range(nbasis):
                n_primitives_i = len(molecule[i])
                n_primitives_j = len(molecule[j])
                
                for k in range(n_primitives_i):
                    for l in range(n_primitives_j):
                        
                        c1c2 = molecule[i][k].coeff * molecule[j][l].coeff
                        N = molecule[i][k].A * molecule[j][l].A
                        p = molecule[i][k].alpha + molecule[j][l].alpha
                        q = molecule[i][k].alpha * molecule[j][l].alpha / p
                        Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                        Q2 = np.dot(Q, Q)
                        
                        P = molecule[i][k].alpha * molecule[i][k].coordinates  + molecule[j][l].alpha  * molecule[j][l].coordinates 
                        Pp = P / p
                        PG = Pp - atom_coordinates[atom]
                        PG2 = np.dot(PG, PG)
                        
                        s = N * c1c2 * np.exp(-q * Q2) * (np.pi / p) ** 1.5
                        
                        V_ne[i, j] += -Z[atom] * N * c1c2 * np.exp(-q * Q2) * (2.0 * np.pi / p) * boys(p * PG2, 0)
    return V_ne


def electron_electron_repulsion(molecule):
    nbasis = len(molecule)
    
    V_ee = np.zeros([nbasis, nbasis, nbasis, nbasis])
    
    for i in range(nbasis):
        for j in range(nbasis):
            for k in range(nbasis):
                for l in range(nbasis):
                    
                    n_primitives_i = len(molecule[i])
                    n_primitives_j = len(molecule[j])
                    n_primitives_k = len(molecule[k])
                    n_primitives_l = len(molecule[l])
                    
                    for ii in range(n_primitives_i):
                        for jj in range(n_primitives_j):
                            for kk in range(n_primitives_k):
                                for ll in range(n_primitives_l):
                                    
                                    N = molecule[i][ii].A * molecule[j][jj].A * molecule[k][kk].A * molecule[l][ll].A
                                    cicjckcl = molecule[i][ii].coeff * molecule[j][jj].coeff * molecule[k][kk].coeff * molecule[l][ll].coeff
                    
                                    pij = molecule[i][ii].alpha + molecule[j][jj].alpha
                                    pkl = molecule[k][kk].alpha + molecule[l][ll].alpha
                         
                                    Pij = molecule[i][ii].alpha*molecule[i][ii].coordinates +\
                                          molecule[j][jj].alpha*molecule[j][jj].coordinates
                                    Pkl = molecule[k][kk].alpha*molecule[k][kk].coordinates +\
                                          molecule[l][ll].alpha*molecule[l][ll].coordinates
                            
                                    Ppij = Pij/pij
                                    Ppkl = Pkl/pkl
                                    
                                    PpijPpkl  = Ppij - Ppkl
                                    PpijPpkl2 = np.dot(PpijPpkl,PpijPpkl)
                                    denom     = 1.0/pij + 1.0/pkl
                            
                                    qij = molecule[i][ii].alpha * molecule[j][jj].alpha / pij
                                    qkl = molecule[k][kk].alpha * molecule[l][ll].alpha / pkl

                                    Qij = molecule[i][ii].coordinates - molecule[j][jj].coordinates
                                    Qkl = molecule[k][kk].coordinates - molecule[l][ll].coordinates
                                    
                                    Q2ij = np.dot(Qij,Qij)
                                    Q2kl = np.dot(Qkl,Qkl)
                                    
                                    term1 = 2.0*np.pi*np.pi/(pij*pkl)
                                    term2 = np.sqrt(  np.pi/(pij+pkl) )
                                    term3 = np.exp(-qij*Q2ij) 
                                    term4 = np.exp(-qkl*Q2kl)
                                                      
                                    V_ee[i,j,k,l] += N * cicjckcl * term1 * term2 * term3 * term4 * boys(PpijPpkl2/denom,0)
    return V_ee


def nuclear_nuclear_repulsion_energy(atom_coords, zlist):
    
    assert (len(atom_coords) == len(zlist))  # uit test
    n_atoms = len(zlist)
    E_NN = 0
    
    for i in range(n_atoms):
        Zi = zlist[i]
        for j in range(n_atoms):
            if j > i:
                Zj = zlist[j]
                
                Rijx = atom_coords[i][0] - atom_coords[j][0]
                Rijy = atom_coords[i][1] - atom_coords[j][1]
                Rijz = atom_coords[i][2] - atom_coords[j][2]
                
                Rij = np.sqrt(Rijx**2 + Rijy**2 + Rijz**2)
                
                E_NN += (Zi * Zj) / Rij
    return E_NN


