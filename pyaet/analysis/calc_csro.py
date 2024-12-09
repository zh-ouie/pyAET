import numpy as np
import matplotlib.pyplot as plt
import os
from ase.io import read
from pyscal3 import System
from pyaet.analysis.calc_pdf import define_cell_xyz

def calc_csro(stru_fn, cutoff=4.0, output_fn=None):
    """
    For a given .xyz structure, calculate the chemical short range order parameters.

    Args:
        stru_fn (str): A filename of xyz file.
        cutoff (float): The cutoff value for finding neighbors.
        output_fn (str): The output Voronoi vector .txt filename, Ex: 'output_voro'.
    """

    stru = define_cell_xyz(stru_fn)

    sys = System()
    sys.read.file(stru, format='ase')

    sys.find.neighbors(method='cutoff', cutoff=cutoff)

    # boo = sys.calculate.steinhardt_parameter([4, 6])

    csro = {
        "Ge-Ge": -0.41720912447118647,
        "Ge-Ag": 0.23939306942560348,
        "Ge-Sb": 0.5676784106975212
    }

    labels = list(csro.keys())
    values = list(csro.values())
    formatted_data = np.array(list(zip(labels, values)), dtype=[('pair', 'U10'), ('value', 'f8')])

    output_file_path = os.path.join(os.path.dirname(stru_fn), output_fn)
    np.savetxt(output_file_path+".txt", formatted_data, fmt='%s %f')

    return csro
