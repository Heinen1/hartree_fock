import numpy as np

class primitive_gaussian():
    """
    Primative gaussian has a alpha exponent, contraction coefficients, nuclear
    coordinations and angular momentum li.
    """    
    # general form gaussian G
    # G_nlm(r, phi, psi) = N_n * r^(n-1) * exp(-a * r^2) * Y_l^m(phi, psi)
    # 1s-orbital: n = 1, l = m = 0
    # G_100 = N_1 * exp( -a * r^2) * Y_0^0
    # Y_0^0 = 1 / sqrt(4 * pi)
    # Integrate A.1 from S&Z give N = (2 * a / pi)^0.75 for 1s
    
    def __init__(self, alpha, coeff, coordinates, lx, ly, lz):
        """
        Construct required parameter to define atomic centerd gaussian basis function.

        Args:
            alpha (float): alpha exponent coefficient
            coeff (float): contraction coeffcient
            coordinates (list): xyz-coordinates of atom
            lx (int): angular momentum in x
            ly (int): angular momentum in y
            lz (int): angular momentum in z
        """        
        self.alpha = alpha
        self.coeff = coeff 
        self.coordinates = np.array(coordinates)
        self.A = (2.0 *  alpha / np.pi)**0.75 # normalization for functions with lx=ly=lz=0
        self.angular = np.array([lx, ly, lz])