# all integrales assume that a 1s orbital are used (no angular momentum terms)
# https://www.youtube.com/watch?v=Dzx8fxqV_CE
# https://www.mathematica-journal.com/2012/02/16/evaluation-of-gaussian-molecular-integrals/
# SO, (3.229) overlap matrix H2-molecule in STO-3G basis at r = 1.4 a.u.
# off-diagional = 0.6593
import numpy as np
from scipy import special
from scipy import linalg
from scipy.integrate import quad
from scipy.special import binom


class atom():
    # only valid for STO-3G basis
    def __init__(self, element, coords):
        self.element = element
        self.coords = np.array(coords)
        self.bf = list()
        
        if element == 'H':
            # 1s
            alpha = [0.3425250914E+01, 0.6239137298E+00, 0.1688554040E+00]
            coefs = [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]        
            self.bf = [[primitive_gaussian(i, j, coords, 0, 0, 0) for i, j in zip(alpha, coefs)]]

        
        if element == 'O':
            # 1s
            alpha = [130.7093214, 23.80886605, 6.443608313]
            coefs = [0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]
            bf_1s = self.basis_function(alpha, coefs, 0, 0, 0)
            
            # 2s
            alpha = [5.033151319, 1.169596125, 0.38038896]
            coefs = [-0.09996722919, 0.3995128261, 0.700115468]
            bf_2s = self.basis_function(alpha, coefs, 0, 0, 0)
            
            # 2px
            alpha = [5.033151319, 1.169596125, 0.38038896]
            coefs = [0.155916275, 0.6076837186, 0.3919573931]
            bf_2px = self.basis_function(alpha, coefs, 1, 0, 0)
            
            # 2py
            alpha = [5.033151319, 1.169596125, 0.38038896]
            coefs = [0.155916275, 0.607683718, 0.3919573931]
            bf_2py = self.basis_function(alpha, coefs, 0, 1, 0)
            
            # 2pz
            alpha = [5.033151319, 1.169596125, 0.38038896]
            coefs = [0.155916275, 0.6076837186, 0.3919573931]
            bf_2pz = self.basis_function(alpha, coefs, 0, 0, 1)

            self.bf = [bf_1s, bf_2s, bf_2px, bf_2py, bf_2pz]
        
    def basis_function(self, alpha, coefs, lx, ly, lz):
        return [primitive_gaussian(i, j, self.coords, lx, ly, lz) for i, j in zip(alpha, coefs)]

class primitive_gaussian():
    # general form gaussian G
    # G_nlm(r, phi, psi) = N_n * r^(n-1) * exp(-a * r^2) * Y_l^m(phi, psi)
    # 1s-orbital: n = 1, l = m = 0
    # G_100 = N_1 * exp( -a * r^2) * Y_0^0
    # Y_0^0 = 1 / sqrt(4 * pi)
    # Integrate A.1 from S&Z give N = (2 * a / pi)^0.75 for 1s
    
    def __init__(self, alpha, coeff, coordinates, lx, ly, lz):
        self.alpha = alpha
        self.coeff = coeff  # contraction coefficients
        self.coordinates = np.array(coordinates)
        self.A = (2.0 *  alpha / np.pi)**0.75 # other terms for l1, l2 l
        self.angular = np.array([lx, ly, lz])
        

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

                    Sx = binomial_expansion(P[0], molecule[i][k].coordinates[0], molecule[j][l].coordinates[0], molecule[i][k].alpha, molecule[j][l].alpha, ax, bx)
                    Sy = binomial_expansion(P[1], molecule[i][k].coordinates[1], molecule[j][l].coordinates[1], molecule[i][k].alpha, molecule[j][l].alpha, ay, by)
                    Sz = binomial_expansion(P[2], molecule[i][k].coordinates[2], molecule[j][l].coordinates[2], molecule[i][k].alpha, molecule[j][l].alpha, az, bz)
                    
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
    elif

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
    print(result)
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


def compute_G(density_matrix, Vee):
    nbasis_functions = density_matrix.shape[0]
    G = np.zeros([nbasis_functions, nbasis_functions])
    
    for i in range(nbasis_functions):
        for j in range(nbasis_functions):
            for k in range(nbasis_functions):
                for l in range(nbasis_functions):
                    density = density_matrix[k, l]
                    J = Vee[i, j, k, l]
                    K = Vee[i, l, k, j]  # check
                    G[i, j] += density * (J - 0.5 * K)
    return G

def compute_density_matrix(mos):
    nbasis_functions = mos.shape[0]  # AO = rows
    density_matrix = np.zeros([nbasis_functions, nbasis_functions])
    # compute P = occupation * CC^dagger
    occupation = 2
    for i in range(nbasis_functions):
        for j in range(nbasis_functions):
            for oo in range(number_occupied_orbitals):
                C = mos[i, oo] # mo is natomic_orbitals x nMOs
                C_dagger = mos[j, oo]    
                density_matrix[i, j] += occupation * C * C_dagger
    
    return density_matrix

def compute_electronic_energy_expectation_value(density_matrix, T, Vne, G):
    nbasis_functions = density_matrix.shape[0]
    Hcore = T + Vne
    electronic_energy = 0
    for i in range(nbasis_functions):
        for j in range(nbasis_functions):
            electronic_energy +=  density_matrix[i, j] * (Hcore[i, j] + 0.5 * G[i, j]) # check for factor 0.5            
    
    return electronic_energy


def scf_cycle(molecular_terms, scf_parameters, molecule):
    S, T, Vne, Vee = molecular_terms
    tolerance, max_scf_steps = scf_parameters
    electronic_energy = 0.0
    nbasis_functions = len(molecule)
    density_matrix = np.zeros([nbasis_functions, nbasis_functions])
    
    # 1 enter into SCF cycles
    for scf_step in range(max_scf_steps):
        
        electronic_energy_old = electronic_energy
        
        # 2 compuyte 2-electron term a
        G = compute_G(density_matrix, Vee)
    
        # 3 for F, make S unit, then get eigenvalues and eigenvectors, transform eigen vectyors back (w.o. unit S)
        F = T + Vne + G
        
        # S^(-1/2) S S^(-1/2)
        S_inverse = linalg.inv(S)
        S_inverse_sqrt = linalg.sqrtm(S_inverse)
        
        # S^(-1/2) F S^(-1/2)
        F_unitS = np.dot(S_inverse_sqrt, np.dot(F, S_inverse_sqrt))
        eigenvalues, eigenvectors = linalg.eigh(F_unitS)
        mos = np.dot(S_inverse_sqrt, eigenvectors)

        # 4 form new density matrix using MOs
        density_matrix = compute_density_matrix(mos)
        
        # 5 Compute eletronic energy, expectation value
        electronic_energy = compute_electronic_energy_expectation_value(density_matrix, T, Vne, G)
        print(electronic_energy)
        
        # 6 Check convergence
        if abs(electronic_energy - electronic_energy_old) < tolerance:
            print("Converged")
            return electronic_energy
    
    print("Not converged")
    return electronic_energy

# create many H2 molecules in STO-3G basis
distances = [round(i*0.1, 2) for i in range(5, 51)]  # unit = bohr (a.u. of position)
molecule_coordinates = [ [[0, 0, 0], [0, 0, distance]] for distance in distances ]
total_energies = []

for molecule_coordinate in molecule_coordinates:
    H1_pg1a = primitive_gaussian(0.3425250914E+01, 0.1543289673E+00, molecule_coordinate[0], 0, 0, 0)
    H1_pg1b = primitive_gaussian(0.6239137298E+00 , 0.5353281423E+00, molecule_coordinate[0], 0, 0, 0)
    H1_pg1c = primitive_gaussian(0.1688554040E+00 ,  0.4446345422E+00, molecule_coordinate[0], 0, 0, 0)
    H2_pg1a = primitive_gaussian(0.3425250914E+01, 0.1543289673E+00, molecule_coordinate[1], 0, 0, 0) # distance is in bohr (1.4 bohr)
    H2_pg1b = primitive_gaussian(0.6239137298E+00 , 0.5353281423E+00, molecule_coordinate[1], 0, 0, 0)
    H2_pg1c = primitive_gaussian(0.1688554040E+00 ,  0.4446345422E+00,  molecule_coordinate[1], 0, 0, 0)
    number_occupied_orbitals = 1
    H1_1s = [H1_pg1a, H1_pg1b, H1_pg1c]
    H2_1s = [H2_pg1a, H2_pg1b, H2_pg1c]
    molecule = [H1_1s, H2_1s]
    atom_coordinates = np.array([molecule_coordinate[0], molecule_coordinate[1]])
    Z = [1.0, 1.0]

    # compute SCF energy
    S = overlap(molecule)
    T = kinetic(molecule)
    Vne = electron_nuclear_attraction(molecule, atom_coordinates, [1.0, 1.0])
    Vee = electron_electron_repulsion(molecule)
    Enn = nuclear_nuclear_repulsion_energy(atom_coordinates, Z)
    molecular_terms = [S, T, Vne, Vee]
    scf_parameters = [1e-5, 20, ]
    electronic_energy = scf_cycle(molecular_terms, scf_parameters, molecule)
    # compute total energy =SCF energy + E_NN
    total_energy = electronic_energy + Enn
    total_energies.append(total_energy)
    
conversion = 0.529  # to Angstrom
conversion = 1
# plot dissociation curve
import matplotlib.pyplot as plt
plt.xlabel("Bond distance (A)")
plt.ylabel("Total energy (Ha)")
plt.plot(np.array(distances) * conversion, total_energies)


# STO-3G basis for 1s orbital on hydrogen
# exponents and zeta taken from basissetexchange.org
H1_pg1a = primitive_gaussian(0.3425250914E+01, 0.1543289673E+00, [0, 0, 0], 0, 0, 0)
H1_pg1b = primitive_gaussian(0.6239137298E+00 , 0.5353281423E+00, [0, 0, 0], 0, 0, 0)
H1_pg1c = primitive_gaussian(0.1688554040E+00 ,  0.4446345422E+00, [0, 0, 0], 0, 0, 0)
H2_pg1a = primitive_gaussian(0.3425250914E+01, 0.1543289673E+00, [0, 0, 1.4], 0, 0, 0) # distance is in bohr (1.4 bohr)
H2_pg1b = primitive_gaussian(0.6239137298E+00 , 0.5353281423E+00, [0, 0, 1.4], 0, 0, 0)
H2_pg1c = primitive_gaussian(0.1688554040E+00 ,  0.4446345422E+00, [0, 0, 1.4], 0, 0, 0)

H1_1s = [H1_pg1a, H1_pg1b, H1_pg1c]
H2_1s = [H2_pg1a, H2_pg1b, H2_pg1c]
molecule = [H1_1s, H2_1s]
atom_coordinates = np.array([[0, 0, 0], [0, 0, 1.4]])
Z = [1.0, 1.0]
print(overlap(molecule))
print(overlap_numerical(molecule))
print(overlap_hm(molecule))
print(kinetic(molecule))
print(electron_nuclear_attraction(molecule, atom_coordinates, Z)) # atomic units
print("Elec-Electron repulsion", electron_electron_repulsion(molecule))
print("Nuc-Nuc repulsion", nuclear_nuclear_repulsion_energy(atom_coordinates, Z))


H1_xyz = [0, 1.43233673, -0.96104039]
H2_xyz = [0, -1.43233673, -0.96104039]
O_xyz = [0, 0, 0.24026010]

# H2O in STO-3G

H1 = atom('H', H1_xyz)
H2 = atom('H', H2_xyz)
O = atom('O', O_xyz)
molecule = H1.bf + H2.bf + O.bf

print(overlap(molecule))
print(overlap_numerical(molecule))
print(overlap_hm(molecule))


# 6-31G basis for 1s orbital on hydrogen
# exponents and zeta taken from basissetexchange.org
H1_pg1a = primitive_gaussian(0.1873113696E+02 , 0.3349460434E-01, [0, 0, 0], 0, 0, 0)
H1_pg1b = primitive_gaussian(0.2825394365E+01, 0.2347269535E+00, [0, 0, 0], 0, 0, 0)
H1_pg1c = primitive_gaussian(0.6401216923E+00,  0.8137573261E+00, [0, 0, 0], 0, 0, 0)
H1_pg2 = primitive_gaussian(0.1612777588E+00,  1.0000000, [0, 0, 0], 0, 0, 0)
H2_pg1a = primitive_gaussian(0.1873113696E+02 , 0.3349460434E-01, [1.4, 0, 0], 0, 0, 0)
H2_pg1b = primitive_gaussian(0.2825394365E+01, 0.2347269535E+00, [1.4, 0, 0], 0, 0, 0)
H2_pg1c = primitive_gaussian(0.6401216923E+00,  0.8137573261E+00, [1.4, 0, 0], 0, 0, 0)
H2_pg2 = primitive_gaussian(0.1612777588E+00,  1.0000000, [1.4, 0, 0], 0, 0, 0)

H1_1s = [H1_pg1a, H1_pg1b, H1_pg1c]
H1_2s = [H1_pg2]
H2_1s = [H2_pg1a, H2_pg1b, H2_pg1c]
H2_2s = [H2_pg2]
molecule = [H1_1s, H1_2s, H2_1s, H2_2s]
atom_coordinates = np.array([[0, 0, 0], [1.4, 0, 0]])
Z = [1.0, 1.0]
print(overlap(molecule))
print(overlap_numerical(molecule))
print(overlap_hm(molecule))
#print(kinetic(molecule))
#print(electron_nuclear_attraction(molecule, atom_coordinates, Z))
#print("Elec-Electron repulsion", electron_electron_repulsion(molecule))

"""
H1_pga = primitive_gaussian(0.3425250914E+01, 0.1543289673E+00, H1_xyz, 0, 0, 0)
H1_pgb = primitive_gaussian(0.6239137298E+00 , 0.5353281423E+00, H1_xyz, 0, 0, 0)
H1_pgc = primitive_gaussian(0.1688554040E+00 ,  0.4446345422E+00, H1_xyz, 0, 0, 0)

H2_pga = primitive_gaussian(0.3425250914E+01, 0.1543289673E+00, H2_xyz, 0, 0, 0)
H2_pgb = primitive_gaussian(0.6239137298E+00 , 0.5353281423E+00, H2_xyz, 0, 0, 0)
H2_pgc = primitive_gaussian(0.1688554040E+00 ,  0.4446345422E+00, H2_xyz, 0, 0, 0)

O_pg1a = primitive_gaussian(130.7093214, 0.1543289673E+00, O_xyz, 0, 0, 0)
O_pg1b = primitive_gaussian(23.80886605 , 0.5353281423E+00, O_xyz, 0, 0, 0)
O_pg1c = primitive_gaussian(6.443608313 ,  0.4446345422E+00, O_xyz, 0, 0, 0)

O_pg2a = primitive_gaussian(5.033151319, -0.09996722919, O_xyz, 0, 0, 0)
O_pg2b = primitive_gaussian(1.169596125 , 0.3995128261, O_xyz, 0, 0, 0)
O_pg2c = primitive_gaussian(0.38038896 ,  0.7001154689, O_xyz, 0, 0, 0)

O_pg2pxa = primitive_gaussian(5.033151319, 0.155916275, O_xyz, 1, 0, 0)
O_pg2pxb = primitive_gaussian(1.169596125 , 0.6076837186, O_xyz, 1, 0, 0)
O_pg2pxc = primitive_gaussian(0.38038896 ,  0.3919573931, O_xyz, 1, 0, 0)

O_pg2pya = primitive_gaussian(5.033151319, 0.155916275, O_xyz, 0, 1, 0)
O_pg2pyb = primitive_gaussian(1.169596125 , 0.6076837186, O_xyz, 0, 1, 0)
O_pg2pyc = primitive_gaussian(0.38038896 ,  0.3919573931, O_xyz, 0, 1, 0)

O_pg2pza = primitive_gaussian(5.033151319, 0.155916275, O_xyz, 0, 0, 1)
O_pg2pzb = primitive_gaussian(1.169596125 , 0.6076837186, O_xyz, 0, 0, 1)
O_pg2pzc = primitive_gaussian(0.38038896 ,  0.3919573931, O_xyz, 0, 0, 1)



H1_1s = [H1_pga, H1_pgb, H1_pgc]
H2_1s = [H2_pga, H2_pgb, H2_pgc]
O_1s = [O_pg1a, O_pg1b, O_pg1c]
O_2s = [O_pg2a, O_pg2b, O_pg2c]
O_2px = [O_pg2pxa, O_pg2pxb, O_pg2pxc]
O_2py = [O_pg2pya, O_pg2pyb, O_pg2pyc]
O_2pz = [O_pg2pza, O_pg2pzb, O_pg2pzc]

""" 
