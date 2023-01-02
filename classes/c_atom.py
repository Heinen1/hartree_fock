import numpy as np
from classes.c_basisset import basisset
from classes.c_primitive_gaussian import primitive_gaussian as prim_gauss


class atom:
    """
    Create object for each newly defined atom
    """
    def __init__(self, element, coords, basis=None):
        self.element = element
        self.coords = np.array(coords)
        self.bf = list()
        self.load_basis_set(basis)

        # loop over basis set parameters which are read in from the
        # basis set exchange .org website (NWCHem format)
        for key, value in self.basis.params[element].items():
            zip_alpha_coefs = list(zip(value['alpha'],
                                       value['coeff']))
            self.add_primitive_gaussians_to_bf(value['shell_sub'],
                                               zip_alpha_coefs)

    def load_basis_set(self, basis):
        """
        Load the basisset file from basissetexchange.org.

        Parameters
        ----------
        basis : string
            Name of the basis set.

        Returns
        -------
        None.

        """
        self.basis = basisset('sto-3g.1.nw')

        if basis == '3-21g':
            self.basis = basisset('3-21g.1.nw')

    def add_primitive_gaussians_to_bf(self, sub_shell, alpha_coefs):
        """
        Add coefficients and exponents to the basis set list.

        Parameters
        ----------
        sub_shell : string
            Sub shell of of basis set.
        alpha_coefs : zip
            Exponents (alpha) and coefficients of contracted Gaussian.

        Returns
        -------
        None.

        """
        if sub_shell == 'P':
            self.bf += [[prim_gauss(i, j, self.coords, 1, 0, 0
                                    ) for i, j in alpha_coefs]]
            self.bf += [[prim_gauss(i, j, self.coords, 0, 1, 0
                                    ) for i, j in alpha_coefs]]
            self.bf += [[prim_gauss(i, j, self.coords, 0, 0, 1
                                    ) for i, j in alpha_coefs]]
        else:
            self.bf += [[prim_gauss(i, j, self.coords, 0, 0, 0
                                    ) for i, j in alpha_coefs]]
