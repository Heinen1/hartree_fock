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

def nuclear_nuclear_repulsion_energy(atom_coords):
    """Nuclear repulsion energy

    Args:
        atom_coords (list): nuclear coordinates of atoms in molecule

    Returns:
        float: total nuclear repulsion energy (Ha)
    """
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





