# https://www.mathematica-journal.com/2012/02/16/evaluation-of-gaussian-molecular-integrals/
import numpy as np
from classes.c_atom import atom
from classes.c_primitive_gaussian import primitive_gaussian
from integrals import *

def main():
    # define coordinates of water molecule in bohr
    H1_xyz = [0, 1.43233673, -0.96104039]
    H2_xyz = [0, -1.43233673, -0.96104039]
    O_xyz = [0, 0, 0.24026010]

    # define each atom in water molecule
    H1 = atom('H', H1_xyz)
    H2 = atom('H', H2_xyz)
    O = atom('O', O_xyz)

    # molecule definition consists of atomic basis functions (bf)
    molecule = H1.bf + H2.bf + O.bf

    #print(overlap_hm_recursive(molecule))
    #print(kinetic_recursive(molecule))
    print(nuclear_electron_recursive(molecule,
        np.array([H1_xyz, H2_xyz, O_xyz]),
        np.array([H1.element, H2.element, O.element])))

if __name__ == '__main':
    main()