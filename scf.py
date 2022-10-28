import numpy as np
from integrals import *


number_occupied_orbitals = 1

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