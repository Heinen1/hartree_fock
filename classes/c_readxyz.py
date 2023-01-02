import os

class readxyz:
    def __init__(self, filename):
        self.filename = filename
        self.atoms = list()

        if os.path.exists(self.filename):
            self.read_file()
        else:
            print("File ", filename, " does not exists")
    

    def read_file(self):
        with open(self.filename) as xyz_file:
            for line_id, line in enumerate(xyz_file):
                if line_id > 2:
                    line_split = line.split()
                    element_type = line_split[0]
                    coords = [float(xyz) for xyz in line_split[1:]]
                    self.atoms.append([element_type, coords])
