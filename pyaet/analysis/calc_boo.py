import numpy as np
import matplotlib.pyplot as plt
import os
from ase.io import read
from pyscal3 import System
from pyaet.analysis.calc_pdf import define_cell_xyz

def calc_boo(stru_fn, cutoff=4.0, output_fn=None):
    """
    For a given .xyz structure, calculate the bond orientational order parameters.

    Args:
        stru_fn (str): A filename of xyz file.
        cutoff (float): The cutoff value for finding neighbors.
        output_fn (str): The output Voronoi vector .txt filename, Ex: 'output_voro'.
    """

    stru = define_cell_xyz(stru_fn)

    sys = System()
    sys.read.file(stru, format='ase')

    sys.find.neighbors(method='cutoff', cutoff=cutoff)

    q4, q6 = sys.calculate.steinhardt_parameter([4, 6], averaged=True)

    boo = np.sqrt(q4**2 + q6**2) / np.sqrt(0.190941**2 + 0.574524**2)

    output_file_path = os.path.join(os.path.dirname(stru_fn), output_fn)
    np.savetxt(output_file_path+".txt",np.transpose([q4, q6]), fmt='%f')

    return q4, q6
