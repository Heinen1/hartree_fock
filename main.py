import numpy as np
from integrals import *
from classes.c_readxyz import readxyz

def main():
    # load xyz into molecule object and load basis functions automatically
    molecule = readxyz('water.xyz')

    # store atom coordinates and elements in array
    atom_coords = np.array([atom['coords'] for atom in molecule.atoms])
    atom_elements = np.array([atom['element'] for atom in molecule.atoms])

    S = overlap_recursive(molecule.molecule_bf)
    T = kinetic_recursive(molecule.molecule_bf)
    VNe = nuclear_electron_recursive(
        molecule.molecule_bf,
        atom_coords,
        atom_elements)
    VNN = nuclear_nuclear_repulsion_energy(
        atom_coords, 
        atom_elements)
    Vee = electron_electron_repulsion(molecule.molecule_bf)

if __name__ == '__main':
    main()
