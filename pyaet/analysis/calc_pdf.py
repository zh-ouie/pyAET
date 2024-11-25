import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from matscipy.neighbours import neighbour_list
from scipy.spatial import distance_matrix


def calc_pdf(stru_fn, rmax=10.0, rmin=0.0, dr=0.1, pdf_type='gr', output_fn=None):
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

    stru = read(stru_fn)

    positions = stru.positions
    dist_matrix = distance_matrix(positions, positions)

    distances = []
    n_atoms = len(stru)
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            if dist_matrix[i,j] <= rmax and dist_matrix[i,j] > rmin:
                distances.append(dist_matrix[i, j])

    grid = np.arange(rmin, rmax+dr, dr)
    hist_grid = np.meshgrid(grid, indexing='ij')[0]

    n, x = np.histogram(distances, bins=hist_grid, density=False)

    r = (x[1:] + x[:-1]) / 2
    shell_volume = 4 * np.pi * (x[1:] ** 3 - x[:-1] ** 3) / 3
    mean_density = 0.1

    if pdf_type == 'gr':
        output_pdf = n / (shell_volume * n_atoms * mean_density)

    return r, output_pdf
