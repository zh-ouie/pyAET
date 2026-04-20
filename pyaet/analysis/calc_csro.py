import numpy as np
import matplotlib.pyplot as plt
import os
from ase.io import read
from pyscal3 import System
from pyaet.analysis.calc_pdf import define_cell_xyz
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from collections import defaultdict


def calc_csro(stru_fn, cutoff=4.0, output_fn=None):
    """
    For a given .xyz structure, calculate the chemical short range order parameters.

    Args:
        stru_fn (str): A filename of xyz file.
        cutoff (float): The cutoff value for finding neighbors.
        output_fn (str): The output Voronoi vector .txt filename, Ex: 'output_voro'.
    """

    stru = define_cell_xyz(stru_fn)

    adaptor = AseAtomsAdaptor()
    structure = adaptor.get_structure(stru)

    comp = structure.composition
    x = comp.fractional_composition.as_dict()
    elements = sorted(x.keys())
    n_elements = len(elements)

    all_neighbors = structure.get_all_neighbors(cutoff)

    # initialize counts.
    N_ij = defaultdict(int)  # N_ij[i,j]：i 原子周围 j 原子的数量
    N_i = defaultdict(int)  # N_i[i]：i 原子的总邻居数
    for elem in elements:
        N_i[elem] = 0
        for other in elements:
            N_ij[(elem, other)] = 0

    # 4. loop over every atom and its neighbor
    for site, neighbors in zip(structure, all_neighbors):
        elem_i = site.specie.symbol
        num_neighbors = len(neighbors)
        N_i[elem_i] += num_neighbors

        for neighbor_site in neighbors:
            elem_j = neighbor_site.specie.symbol
            N_ij[(elem_i, elem_j)] += 1

    # calculate CSRO parameter α_ij = 1 - (N_ij / N_i) / x_j
    alpha = {}
    for (elem_i, elem_j), n_ij in N_ij.items():
        n_i = N_i[elem_i]
        if n_i == 0:
            continue
        alpha_ij = 1 - (n_ij / n_i) / x[elem_j]
        alpha[(elem_i, elem_j)] = alpha_ij

    sro_matrix = np.empty((n_elements, n_elements))

    for i, elem_i in enumerate(elements):
        for j, elem_j in enumerate(elements):
            sro_matrix[i, j] = alpha.get((elem_i, elem_j))

    """
    alpha
    {('Ni', 'Ni'): -0.06177914858189948,
     ('Ni', 'Pd'): 0.04715355571885482,
     ('Ni', 'Pt'): 0.06021515432165225,
     ('Pd', 'Ni'): 0.0835587878690549,
     ('Pd', 'Pd'): -0.051255369749094726,
     ('Pd', 'Pt'): -0.1089608677044347,
     ('Pt', 'Ni'): 0.08022419873865516,
     ('Pt', 'Pd'): -0.1284649374815281,
     ('Pt', 'Pt'): 0.06955654141795131}
     
    elements: ['Ni', 'Pd', 'Pt']
    
    sro_matrix:
    [[
         1-1 (Ni-Ni), 1-2 (Ni-Pd), 1-3 (Ni-Pt),
         2-1 (Pd-Ni), 2-2 (Pd-Pd), 2-3 (Pd-Pt),
         3-1 (Pt-Ni), 3-2 (Pt-Pd), 3-3 (Pt-Pt)
    ]]
    
    array([[-0.06177915,  0.04715356,  0.06021515],
       [ 0.08355879, -0.05125537, -0.10896087],
       [ 0.0802242 , -0.12846494,  0.06955654]])
    """

    # https: // github.com / caimeiniu / pyhea / blob / master / examples / heatmap.png
    # 3*3 matrix:
    # [[
    #     1-1, 1-2, 1-3,
    #     2-1, 2-2, 2-3,
    #     3-1, 3-2, 3-3
    # ]]

    labels = [' '.join(pair) for pair in alpha.keys()]
    values = list(alpha.values())

    output_file_path = os.path.join(os.path.dirname(stru_fn), output_fn)
    formatted_data = np.array(list(zip(labels, values)), dtype=[('pair', 'U12'), ('value', 'f4')])
    np.savetxt(output_file_path+".txt", formatted_data, fmt='%-12s %.5f')

    #todo: the current algorithm does not give aij = aji. here we use a temp fix.
    # please fix it later when using pycsro.
    sro_matrix = (sro_matrix + sro_matrix.T) / 2.0

    return elements, sro_matrix*3

#
# def calc_csro_pyscal(stru_fn, cutoff=4.0, output_fn=None):
#     """
#     For a given .xyz structure, calculate the chemical short range order parameters.
#
#     Args:
#         stru_fn (str): A filename of xyz file.
#         cutoff (float): The cutoff value for finding neighbors.
#         output_fn (str): The output Voronoi vector .txt filename, Ex: 'output_voro'.
#     """
#
#     stru = define_cell_xyz(stru_fn)
#
#     sys = System()
#     sys.read.file(stru, format='ase')
#     atom_labels = sys.atoms['species']
#     atom_types = sys.atoms['types']
#
#     sys.find.neighbors(method='cutoff', cutoff=cutoff)
#
#     # boo = sys.calculate.steinhardt_parameter([4, 6])
#
#     csro = {
#         "Ge-Ge": -0.41720912447118647,
#         "Ge-Ag": 0.23939306942560348,
#         "Ge-Sb": 0.5676784106975212
#     }
#
#     # https: // github.com / caimeiniu / pyhea / blob / master / examples / heatmap.png
#     # 3*3 matrix:
#     # [[
#     #     1-1, 1-2, 1-3,
#     #     2-1, 2-2, 2-3,
#     #     3-1, 3-2, 3-3
#     # ]]
#
#     labels = list(csro.keys())
#     values = list(csro.values())
#
#     output_file_path = os.path.join(os.path.dirname(stru_fn), output_fn)
#     formatted_data = np.array(list(zip(labels, values)), dtype=[('pair', 'U12'), ('value', 'f4')])
#     np.savetxt(output_file_path+".txt", formatted_data, fmt='%12s %.5f')
#
#     return csro
