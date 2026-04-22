import os
import pickle

import numpy as np

from pyaet.resire_numpy.interp_pj_realspace import interp_pj_realspace
from pyaet.src.io_helper import read_mat_file


class RESIRE_Reconstructor:
    def __init__(self):
        self.InputProjections = None
        self.InputAngles = None

        self.reconstruction = None
        self.errR = None
        self.dim1 = None
        self.dim2 = None
        self.n1_oversampled = None
        self.n2_oversampled = None
        self.num_projs = None
        self.Rarr_record = None
        self.Rarr2_record = None

        self.monitor_R = 0
        self.monitorR_loopLength = 10

        self.filename_Projections = ""
        self.filename_Angles = ""
        self.filename_Results = "RESIRE_rec"

        self.num_iterations = 50
        self.oversampling_ratio = 3
        self.gridding_method = 1

        self.vector1 = np.array([0, 0, 1])
        self.vector2 = np.array([0, 1, 0])
        self.vector3 = np.array([1, 0, 0])

        self.save_temp = False
        self.save_loopLength = 50
        self.saveFilename = "./results/temp_rec"

        self.step_size = 2.0
        self.dtype = "float64"
        self.sum_rot_pjs = None
        self.use_parallel = False
        self.xj = None
        self.yj = None
        self.zj = None
        self.Rot_x = None
        self.Rot_y = None
        self.Support = None
        self.initial_model = 0

    def read_files(self):
        if self.file_exist(self.filename_Projections):
            if self.filename_Projections.endswith(".mat"):
                self.InputProjections = read_mat_file(self.filename_Projections)
            else:
                self.InputProjections = np.load(self.filename_Projections)
        else:
            raise Exception("RESIRE: Projections file does not exist!")

        if self.file_exist(self.filename_Angles):
            if self.filename_Angles.endswith(".mat"):
                self.InputAngles = read_mat_file(self.filename_Angles)
            else:
                self.InputAngles = np.load(self.filename_Angles)
        else:
            raise Exception("RESIRE: Angles file does not exist!")

        print("self.InputProjections.shape", self.InputProjections.shape)
        print("self.InputAngles.shape", self.InputAngles.shape)

    def check_prepare_data(self):
        self.num_projs = self.InputProjections.shape[2]
        self.dim1 = self.InputProjections.shape[0]
        self.dim2 = self.InputProjections.shape[1]

        if self.num_projs != self.InputAngles.shape[0]:
            raise Exception("RESIRE: Number of projections and Angles does not match!")

        if self.InputAngles.shape[1] > 3:
            raise Exception("RESIRE: Input Angle 2nd dimensions larger than three!")

        if self.InputAngles.shape[1] == 1:
            self.InputAngles = np.concatenate(
                (
                    np.zeros((self.num_projs, 1)),
                    self.InputAngles,
                    np.zeros((self.num_projs, 1)),
                ),
                axis=1,
            )

        if self.gridding_method != 1:
            raise Exception("RESIRE: Unrecognized gridding method.")

        self.n1_oversampled = round(self.dim1 * self.oversampling_ratio)
        self.n2_oversampled = round(self.dim2 * self.oversampling_ratio)

    def run_gridding(self):
        print("RESIRE: Interpolate real space projections...\n\n")
        if self.gridding_method == 1:
            print("start interp_pj_realspace")
            interp_pj_realspace(self)
            print("end interp_pj_realspace")

    def clear_calc_variables(self):
        self.InputProjections = None
        self.InputAngles = None
        self.sum_rot_pjs = None
        self.xj = None
        self.yj = None
        self.zj = None
        self.Rot_x = None
        self.Rot_y = None

    def save_results(self):
        dir_path = os.path.dirname(self.filename_Results)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(f"{self.filename_Results}.pkl", "wb") as f:
            pickle.dump(self, f)

    def set_parameters(self, resire_param):
        for key, value in resire_param.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise Exception(f"RESIRE: Invalid option {key} provided.")

    def file_exist(self, file_name):
        return os.path.isfile(file_name)
