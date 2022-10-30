import numpy as np
from scipy import special
from scipy.integrate import quad
from scipy.special import binom
from scipy import linalg

# nuclear charges per atom type
nuclear_charges = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8}

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

def nuclear_electron_recursive(molecule, atom_coords, Zlist):
    nbasis = len(molecule)
    atoms = len(atom_coords)    
    VNe = np.zeros([nbasis, nbasis])
    Z_charge = [nuclear_charges[Z] for Z in Zlist]
    
    for atom in range(atoms):
        R = atom_coords[atom]
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
                        c1 = molecule[i][k].coeff
                        c2 = molecule[j][l].coeff 

                        p = (molecule[i][k].coordinates * molecule[i][k].alpha) + (molecule[j][l].coordinates * molecule[j][l].alpha)
                        sum_alpha = molecule[i][k].alpha + molecule[j][l].alpha
                        P = p / sum_alpha
                        
                        multiple_alpha = molecule[i][k].alpha * molecule[j][l].alpha
                        dist = (np.sum(np.square(molecule[i][k].coordinates - molecule[j][l].coordinates)))
                        prefactor = np.exp(-(multiple_alpha/sum_alpha)*(dist))

                        vne_int, _ = quad(VNeIntegrand, 0, 1, epsabs=1e-10, args=(molecule[i][k], molecule[j][l], R))

                        VNe[i, j] += -Z_charge[atom] * N1 * N2 * c1 * c2 * vne_int * prefactor * (2 * np.pi) / sum_alpha
    return VNe

def VNeIntegrand(t, bf1, bf2, R):
    alpha = bf1.alpha
    beta = bf1.alpha
    A = bf1.coordinates
    B = bf2.coordinates
    p = alpha + beta
    P = ((alpha * A) + (beta * B)) / p
    dist = (np.sum(np.square(P - R)))
    
    nx = Nrec(P[0], A[0], B[0], alpha, beta, bf1.angular[0], bf2.angular[0], t, R[0])
    ny = Nrec(P[1], A[1], B[1], alpha, beta, bf1.angular[1], bf2.angular[1], t, R[1])
    nz = Nrec(P[2], A[2], B[2], alpha, beta, bf1.angular[2], bf2.angular[2], t, R[2])

    return np.exp(-p*(t**2) * abs(dist)) * nx * ny * nz


def Nrec(P, A, B, alpha, beta, a, b, t, R):
    p = alpha + beta
    if (a == 0) and (b == 0):
       return 1.0
    elif (a == 1) and (b == 0):
        return -(A - P + (t**2 * (P - R)))
    elif (b == 0):
        term1 = -(A - P + (t**2)*(P - R)) * Nrec(P, A, B, alpha, beta, a - 1, 0, t, R)
        term2 = ((a-1) / (2*p)) * (1 - t**2) * Nrec(P, A, B, alpha, beta, a - 2, 0, t, R)
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
                    # Replacement operator: /. t --> (x + 1) / 2 means: to replace variable t with (x + 1) / 2
                    # Boys function integration limits: [0, 1]
                    # Gauss-Chebysheve quadratic integrationlimits: [-1, 1] --> (x + 1) / 2 gives [0, 1]
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

def nuclear_nuclear_repulsion_energy(atom_coords):
    n_atoms = len(atom_coords)
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
    




