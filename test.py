import json
from classes.c_basisset import basisset
from classes.c_atom2 import atom2
from classes.c_atom import atom
from scf import scf_cycle

bf = basisset('sto-3g.1.nw')
bf = basisset('3-21g.1.nw')

H1_xyz = [0, 1.43233673, -0.96104039]
H2_xyz = [0, -1.43233673, -0.96104039]
O_xyz = [0, 0, 0.24026010]

H1a = atom2('H', H1_xyz)
H2a = atom2('H', H2_xyz)
O1a = atom2('O', O_xyz)
water1 = H1a.bf + H2a.bf + O1a.bf
molecule = H1a.bf + H2a.bf + O1a.bf

H1a = atom2('H', H1_xyz, basis='3-21g')
H2a = atom2('H', H2_xyz, basis='3-21g')
O1a = atom2('O', O_xyz, basis='3-21g')
water_3_21g = H1a.bf + H2a.bf + O1a.bf

# define each atom in water molecule
H1 = atom('H', H1_xyz)
H2 = atom('H', H2_xyz)
O = atom('O', O_xyz)

water = H1.bf + H2.bf + O.bf

S_water_sto3g = overlap_recursive(water1)
S_water_3_21g = overlap_recursive(water_3_21g)

H2_1 = atom2('H', [0, 0, 0])
H2_2 = atom2('H', [1.4, 0, 0])
hydrogen = H2_1.bf + H2_2.bf
S_H2 = overlap_recursive(hydrogen)


atom_coordinates = np.array([H1_xyz, H2_xyz, O_xyz])
Z = [1.0, 1.0, 8.0]
# compute SCF energy
S = overlap_recursive(molecule)
T = kinetic_recursive(molecule)
Vne = nuclear_electron_recursive(molecule, atom_coordinates, ['H', 'H', 'O'])
Vee = electron_electron_repulsion(molecule)
Enn = nuclear_nuclear_repulsion_energy(atom_coordinates, ['H', 'H', 'O'])
molecular_terms = [S, T, Vne, Vee]
scf_parameters = [1e-5, 40, ]
electronic_energy = scf_cycle(molecular_terms, scf_parameters, molecule)
# compute total energy =SCF energy + E_NN
total_energy = electronic_energy + Enn