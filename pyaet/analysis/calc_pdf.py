import numpy as np
import os
import matplotlib.pyplot as plt
from ase.io import read
from matscipy.neighbours import neighbour_list
from scipy.spatial import distance_matrix
from pyscal3 import System


def calc_pdf(stru_fn, rmax=10.0, rmin=0.1, dr=0.1, pdf_type='gr', output_fn=None):
    """
    For a given .xyz structure, calculate the PDF. If `pdf_type` equals 'Gr', it gives Gr in PDFgui.

    Args:
        stru_fn (str): A filename of xyz file.
        rmax (float): The rmax value.
        rmin (float): The rmin value.
        dr (float): The r step.
        pdf_type (str): `Gr` for PDF in PDFgui. `gr` for RDF.
        output_fn (str): The output PDF .txt filename, Ex: 'output_pdf'.
    """
    stru = define_cell_xyz(stru_fn)

    sys = System()
    sys.read.file(stru, format='ase')

    # https: // github.com / pyscal / pyscal3 / blob / 38
    # f3975c99b6a5b7c9ad8108906a8803e7a4f345 / src / pyscal3 / operations / calculations.py  # L219
    output_pdf, r = sys.calculate.radial_distribution_function(rmin=rmin, rmax=rmax, bins=int((rmax-rmin)/dr))

    output_file_path = os.path.join(os.path.dirname(stru_fn), output_fn)
    np.savetxt(output_file_path+".txt", np.transpose([r, output_pdf]), fmt='%f')

    return r, output_pdf

def define_cell_xyz(stru_fn):
    """
    Define the cell for a given xyz, and convert it to ASE Atoms class.

    Args:
        stru_fn (str): A filename of xyz file.
    """
    atoms = read(stru_fn)
    pos = atoms.get_positions()

    pos_new = np.zeros(pos.shape)
    pos_new[:, 0] = pos[:, 0] - np.min(pos[:, 0])
    pos_new[:, 1] = pos[:, 1] - np.min(pos[:, 1])
    pos_new[:, 2] = pos[:, 2] - np.min(pos[:, 2])

    atoms.set_positions(pos_new)
    atoms.set_cell([np.max(pos_new[:, 0]), np.max(pos_new[:, 0]), np.max(pos_new[:, 0])])

    return atoms

# def calc_pdf(stru_fn, rmax=10.0, rmin=0.0, dr=0.1, pdf_type='gr', output_fn=None):
#     """
#     For a given .xyz structure, calculate the PDF. If `pdf_type` equals 'Gr', it gives Gr in PDFgui.
#
#     Args:
#         stru_fn (str): A filename of xyz file.
#         rmax (float): The rmax value.
#         rmin (float): The rmin value.
#         dr (float): The r step.
#         pdf_type (str): `Gr` for PDF in PDFgui. `gr` for RDF.
#         output_fn (str): The output PDF .txt filename, Ex: 'output_pdf'.
#     """
#     stru = read(stru_fn)
#
#     positions = stru.positions
#     dist_matrix = distance_matrix(positions, positions)
#
#     distances = []
#     n_atoms = len(stru)
#     for i in range(n_atoms):
#         for j in range(i+1, n_atoms):
#             if dist_matrix[i,j] <= rmax and dist_matrix[i,j] > rmin:
#                 distances.append(dist_matrix[i, j])
#
#     grid = np.arange(rmin, rmax+dr, dr)
#     hist_grid = np.meshgrid(grid, indexing='ij')[0]
#
#     n, x = np.histogram(distances, bins=hist_grid, density=False)
#
#     r = (x[1:] + x[:-1]) / 2
#     shell_volume = 4 * np.pi * (x[1:] ** 3 - x[:-1] ** 3) / 3
#     mean_density = 0.1
#
#     if pdf_type == 'gr':
#         output_pdf = n / (shell_volume * n_atoms * mean_density)
#
#     output_file_path = os.path.join(os.path.dirname(stru_fn), output_fn)
#     np.savetxt(output_file_path+".txt", np.transpose([r, output_pdf]), fmt='%f')
#
#     return r, output_pdf
