import numpy as np
from scipy import special
from scipy.integrate import quad
from scipy.special import binom
from scipy import linalg

def kinetic_recursive(molecule):
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

def nuclear_electron_recursive(molecule, atom_coords, Z):
    nbasis = len(molecule)
    atoms = len(Z)    
    VNe = np.zeros([nbasis, nbasis])
    
    for atom in atoms:
        R = atom_coords[atom]
        Rx = R[0]
        Ry = R[1]
        Rz = R[2]
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

                        nx = Nrec(P[0], molecule[i][k].coordinates[0], molecule[j][l].coordinates[0], molecule[i][k].alpha, molecule[j][l].alpha, ax, bx, t, R[0])
                        ny = Nrec(P[1], molecule[i][k].coordinates[1], molecule[j][l].coordinates[1], molecule[i][k].alpha, molecule[j][l].alpha, ay, by, t, R[1])
                        nz = Nrec(P[2], molecule[i][k].coordinates[2], molecule[j][l].coordinates[2], molecule[i][k].alpha, molecule[j][l].alpha, az, bz, t, R[2])
                        Pt = nx * ny * nz

                        sum_alpha = molecule[i][k].alpha + molecule[j][l].alpha
                        multiple_alpha = molecule[i][k].alpha * molecule[j][l].alpha
                        dist = (np.sum(np.square(molecule[i][k].coordinates - molecule[j][l].coordinates)))
                        prefactor = np.exp(-(multiple_alpha/sum_alpha)*abs(dist))

                        VNe[i, j] += N * prefactor * (np.pi / (sum_alpha))**1.5 * molecule[i][k].coeff * molecule[j][l].coeff * (kx + ky + kz)
    return VNe

def Nrec(P, A, B, alpha, beta, a, b, t, R):
    p = alpha + beta
    if (a == 0) and (b == 0):
       return 1.0
    elif (b == 0):
        term1 = -(A - P + t*(P - R)) * Nrec(P, A, B, alpha, beta, a, 0, t, R)
        term2 = (a / (2*p)) * (1 - t**2) * Nrec(P, A, B, alpha, beta, a - 1, 0, t, R)
        return term1 + term2
    else:
        term1 = Nrec(P, A, B, alpha, beta, a + 1, b - 1, t, R)
        term2 = (A - B) * Nrec(P, A, B, alpha, beta, a, b - 1, t, R)
        return term1 + term2



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
    




