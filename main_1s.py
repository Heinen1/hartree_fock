"""Calculate integrals for 1s-orbitals only."""
import numpy as np
from integrals_spherical import *  # integrals valid for 1s only
from classes.c_primitive_gaussian import primitive_gaussian
from scf import scf_cycle
from integrals import *

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
print(kinetic(molecule))
print(electron_nuclear_attraction(molecule, atom_coordinates, Z)) # atomic units
print(nuclear_electron_recursive(molecule, atom_coordinates, ['H', 'H']))
print("Elec-Electron repulsion", electron_electron_repulsion(molecule))
print("Nuc-Nuc repulsion", nuclear_nuclear_repulsion_energy(atom_coordinates, Z))


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
#print(kinetic(molecule))
print(electron_nuclear_attraction(molecule, atom_coordinates, Z))
#print("Elec-Electron repulsion", electron_electron_repulsion(molecule))

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