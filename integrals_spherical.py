"""Integrals are only valid for 1s orbitals"""
import numpy as np
from scipy import special
from scipy.integrate import quad

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


def electron_electron_repulsion_s(molecule):
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