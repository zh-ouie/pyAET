import numpy as np
import os
import matplotlib.pyplot as plt
from ase.io import read
from pyscal3 import System
from pyaet.analysis.calc_pdf import define_cell_xyz
from pymatgen.core import Structure
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.ase import AseAtomsAdaptor
from collections import Counter

def calc_voronoi(stru_fn, output_fn=None):
    """
    For a given .xyz structure, calculate the Voronoi indices.

    Args:
        stru_fn (str): A filename of xyz file.
        output_fn (str): The output Voronoi vector .txt filename, Ex: 'output_voro'.
    """

    stru = define_cell_xyz(stru_fn)

    adaptor = AseAtomsAdaptor()
    structure = adaptor.get_structure(stru)

    vnn = VoronoiNN(allow_pathological=True)
    voronoi_result = vnn.get_all_voronoi_polyhedra(structure)

    n3_list = []
    n4_list = []
    n5_list = []
    n6_list = []

    for i, neighbor_dict in enumerate(voronoi_result):
        face_edge_counts = []

        for facet_info in neighbor_dict.values():
            n_edges = facet_info['n_verts']
            face_edge_counts.append(n_edges)

        cnt = Counter(face_edge_counts)
        n3_list.append(cnt.get(3, 0))
        n4_list.append(cnt.get(4, 0))
        n5_list.append(cnt.get(5, 0))
        n6_list.append(cnt.get(6, 0))

    n3 = np.array(n3_list)
    n4 = np.array(n4_list)
    n5 = np.array(n5_list)
    n6 = np.array(n6_list)


    voronoi_indices = list(zip(n3, n4, n5, n6))
    counts = Counter(voronoi_indices)
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    types = [f"<{t[0]},{t[1]},{t[2]},{t[3]}>" for t in [item[0] for item in sorted_items]]
    fractions = [count / len(voronoi_indices) for count in [item[1] for item in sorted_items]]

    output_file_path = os.path.join(os.path.dirname(stru_fn), output_fn)
    data = np.array(
        list(zip(types, fractions)),
        dtype=[('index', 'U12'), ('fraction', 'f4')]
    )
    np.savetxt(output_file_path+".txt", data, fmt='%-12s %.5f')

    return types, fractions


# def calc_voronoi_pyscal(stru_fn, cutoff=4.0, output_fn=None):
#     """
#     For a given .xyz structure, calculate the Voronoi vector.
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
#
#     sys.find.neighbors(method='voronoi')
#
#     n3_list = []
#     n4_list = []
#     n5_list = []
#     n6_list = []
#
#     for atom in sys.atoms:
#         # obtain the vertices of each face, which equals to the number of edges.
#         if hasattr(atom, 'face_vertices'):
#             edges = atom.face_vertices  # eg: [4, 5, 6, 4, 5, ...]
#         else:
#             edges = []
#
#         # 统计不同边数的面数量
#         cnt = Counter(edges)
#         n3_list.append(cnt.get(3, 0))
#         n4_list.append(cnt.get(4, 0))
#         n5_list.append(cnt.get(5, 0))
#         n6_list.append(cnt.get(6, 0))
#
#     n3 = np.array(n3_list)
#     n4 = np.array(n4_list)
#     n5 = np.array(n5_list)
#     n6 = np.array(n6_list)
#
#     top_n = 10
#     voronoi_indices = list(zip(n3, n4, n5, n6))
#     counts = Counter(voronoi_indices)
#     sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
#     sorted_items = sorted_items[:top_n]
#
#     types = [f"<{t[0]},{t[1]},{t[2]},{t[3]}>" for t in [item[0] for item in sorted_items]]
#     fractions = [count / len(voronoi_indices) for count in [item[1] for item in sorted_items]]
#
#     # plot
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(types, fractions, color='tab:blue', edgecolor='black', linewidth=0.8)
#
#
#
#     return types, fractions
