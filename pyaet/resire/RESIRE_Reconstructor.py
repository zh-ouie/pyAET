'''
I have converted the resire class from MATLAB to Python. Please note that in Python, the class properties are defined in the constructor __init__ and the methods are defined as regular Python methods. Also, MATLAB's indexing starts from 1, while Python's indexing starts from 0.


This code defines the resire class in Python, mirroring the properties and methods from the MATLAB version. Please note that I've kept some methods empty as placeholders, and you should implement them according to your specific requirements. Also, make sure to include the missing methods from your original MATLAB code.
'''


import numpy as np
from scipy.ndimage import map_coordinates
import os
import pickle
from pyaet.resire.interp_pj_realspace import interp_pj_realspace

class RESIRE_Reconstructor:

    def __init__(self):
        # internal variables to be cleared after run
        self.InputProjections = None
        self.InputAngles = None

        # internal variables to be kept after run
        self.reconstruction = None
        self.errR = None
        self.dim1 = None  # projection size1
        self.dim2 = None  # projection size2
        self.n1_oversampled = None  # projection size1 after oversampling
        self.n2_oversampled = None  # projection size2 after oversampling
        self.num_projs = None
        self.Rarr_record = None
        self.Rarr2_record = None

        # R factor monitoring
        self.monitor_R = 0
        self.monitorR_loopLength = 10

        # filenames
        self.filename_Projections = ''
        self.filename_Angles = ''
        self.filename_Results = 'RESIRE_rec'

        # reconstruction parameters
        self.num_iterations = 50
        self.oversampling_ratio = 3
        self.gridding_method = 1

        # axes vectors for phi, theta, psi
        self.vector1 = np.array([0, 0, 1])
        self.vector2 = np.array([0, 1, 0])
        self.vector3 = np.array([1, 0, 0])

        # save temp reconstruction
        self.save_temp = False
        self.save_loopLength = 50
        self.saveFilename = './results/temp_rec'

        self.step_size = 2.0  # normalized step_size is chosen in range (1,3)
        self.dtype = 'float32'  # data type, single for memory efficiency
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
            self.InputProjections = np.load(self.filename_Projections)
        else:
            raise Exception('RESIRE: Projections file does not exist!')
        if self.file_exist(self.filename_Angles):
            self.InputAngles = np.load(self.filename_Angles)
        else:
            raise Exception('RESIRE: Angles file does not exist!')

    def check_prepare_data(self):
        # set number of projections
        self.num_projs = self.InputProjections.shape[2]
        self.dim1 = self.InputProjections.shape[0]
        self.dim2 = self.InputProjections.shape[1]

        # input angle and projection size check
        if self.num_projs != self.InputAngles.shape[0]:
            raise Exception('RESIRE: Number of projections and Angles does not match!')

        # input angle check
        if self.InputAngles.shape[1] > 3:
            raise Exception('RESIRE: Input Angle 2nd dimensions larger than three!')

        # if only one angle set is given, make them three Euler angles
        if self.InputAngles.shape[1] == 1:
            self.InputAngles = np.concatenate((np.zeros((self.num_projs, 1)),
                                               self.InputAngles,
                                               np.zeros((self.num_projs, 1))), axis=1)

        # check if interp method is legitimate
        if self.gridding_method != 1:
            raise Exception('RESIRE: Unrecognized gridding method.')

        self.n1_oversampled = round(self.dim1 * self.oversampling_ratio)
        self.n2_oversampled = round(self.dim2 * self.oversampling_ratio)

    def run_gridding(self):
        print('RESIRE: Interpolate real space projections...\n\n')
        if self.gridding_method == 1:
            interp_pj_realspace(self)

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
        #check if folder exists, otherwise, create the folder.
        dir_path = os.path.dirname(self.filename_Results)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(f'{self.filename_Results}.pkl',"wb") as f:
            pickle.dump(self, f)

    def set_parameters(self, resire_param):
        for key, value in resire_param.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(key, value)
            else:
                raise Exception(f'RESIRE: Invalid option {key} provided.')

    def file_exist(self, FileName):
        return os.path.isfile(FileName)

    # def set_parameters_old(self, **kwargs):
    #     for key, value in kwargs.items():
    #         if hasattr(self, key):
    #             setattr(self, key, value)
    #         else:
    #             raise Exception(f'RESIRE: Invalid option {key} provided.')
