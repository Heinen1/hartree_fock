import os
from classes.c_atom import atom

nuclear_charges = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4,
'B': 5, 'C': 6, 'N': 7, 'O': 8}

class readxyz:
    """Load molecule from XYZ file and read automotically basis functions.
    """
    def __init__(self, filename, basis_set=None):
        """
        Read XYZ file if it exists.

        Parameters
        ----------
        filename : string
            Name of XYZ file.
        """
        self.filename = filename
        self.basis_set = basis_set
        self.atoms = list()
        self.molecule_bf = list()

        if os.path.exists(self.filename):
            self.read_file()
            self.get_basis_functions()
        else:
            print("File ", filename, " does not exist")
    
    def get_basis_functions(self):
        """
        Load basis set functions with specified basis set (STO-3G default)
        """
        for atom in self.atoms:
            self.molecule_bf += atom2(  atom['element'],
                                        atom['coords'],
                                        self.basis_set).bf

    def read_file(self):
        """
        Open XYZ file and loop over all lines. From third row onwards,
        the atom types with there x, y, and z coordinates exist.
        """
        with open(self.filename) as xyz_file:
            for line_id, line in enumerate(xyz_file):
                if line_id > 1:
                    line_split = line.split()
                    element_type = line_split[0]
                    coords = [float(xyz) for xyz in line_split[1:]]
                    dict_atom = {'element': element_type,
                                'coords': coords,
                                'charge': nuclear_charges[element_type]}
                    self.atoms.append(dict_atom)
