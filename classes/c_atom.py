import numpy as np
from classes.c_primitive_gaussian import primitive_gaussian

class atom():
    """
    Each atom consists of atomic orbital and contracted basis functions
    """
    def __init__(self, element, coords):
        """
        Create list of basis functions for each atom of alpha coeffcient
        and contraction coefficients.

        Args:
            element (string): atom type
            coords (list): xyz coordinates in bohr!
        """
        self.element = element
        self.coords = np.array(coords)
        self.bf = list()
        self.active = 1
        
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
