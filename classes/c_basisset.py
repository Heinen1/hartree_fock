# define first 10 chemical element types
elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O']
# define atom  shells(called sub shells)
shell_subs = {1: 'S', 2: 'P', 3: 'D'}


class basisset:
    """
    Read NWChem format file fromm basissetexchange.org (BSSE) and 
    create basis set using a dictionary format:
    'element' : {"gto_1": {"coef": [],
                           "exp": []},
                 "gto_2": : {"coef": [],
                           "exp": []}
    }
    """

    def __init__(self, filename):
        """
        Load exponents and coefficients from BSSE.

        Parameters
        ----------
        filename : string
            NWChem file from BSSE.

        Returns
        -------
        None.

        """
        self.filename = filename
        self.params = dict()
        self.read()

    def read(self):
        """
        Open basis set file from BSSE.

        Returns
        -------
        None.

        """
        self.n_last = 0
        self.new_element = None

        with open(self.filename) as bf_file:
            for line in bf_file:
                cond1 = not line.startswith("#")
                cond2 = not line.startswith("BASIS")
                cond3 = not line.startswith("END")

                # skip lines that start with comment or BASIS and END keyword
                if cond1 and cond2 and cond3 and line.rstrip():
                    self.define_bf_parameters(line)

    def define_bf_parameters(self, line):
        """
        Store exponents and coefficients in basis set dictionary

        Parameters
        ----------
        line_array : list
            Line of NWchem basis set file.

        Returns
        -------
        None.

        """
        line_array = line.split()
        element = line_array[0]
        shell = line_array[1]

        if element in elements:
            self.new_element = element
            self.new_shell = shell
            if self.new_element not in self.params:
                self.n_last = 0
                self.params[self.new_element] = {}
            self.n_last = len(self.params[self.new_element])
        elif self.new_element:
            line_array = [float(ls) for ls in line_array]

            for col in range(1, len(line_array)):
                counter = self.n_last + col - 1

                self.initialize_new_element(self.new_element, counter)
                self.set_new_element(counter, col, line_array)

    def initialize_new_element(self, element, counter):
        """
        Initialize dictionary for each element of basis functions.

        Parameters
        ----------
        element : string
            Element type of basis functions.
        counter : int
            Number of GTO occuring for that basis function.

        Returns
        -------
        None.

        """
        if counter not in self.params[self.new_element]:
            self.params[element][counter] = {}
            self.params[element][counter]['alpha'] = list()
            self.params[element][counter]['coeff'] = list()

    def set_new_element(self, counter, l_col, line_split):
        """
        Add coefficient and alpha exponent to dictionary of basis functions.

        Parameters
        ----------
        counter : int
            Number of GTO occuring for that basis function..
        l_col : int
            NWChem data can have 1-2 columns that list GTO's.
        line_split : list
            line from the NWchem data file.

        Returns
        -------
        None.
        """
        self.params[self.new_element][counter]['alpha'].append(line_split[0])
        self.params[self.new_element][counter]['coeff'].append(
            line_split[l_col])
        self.params[self.new_element][counter]['shell'] = self.new_shell
        self.params[self.new_element][counter]['shell_sub'] = shell_subs[l_col]
