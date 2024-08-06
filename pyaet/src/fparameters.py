import numpy as np
from scipy.io import loadmat
import os

def fparameters(Z):
    # file_dir = os.path.dirname(os.getcwd())
    # print(file_dir)
    # fparams = sio.loadmat(os.path.join(file_dir,'pyAET/pyaet/src/fparameters.mat'))['fparams']
    fparams = loadmat('/Users/longyang/Documents/Tongji/dev/pyAET/pyaet/src/fparameters.mat')['fparams'] #todo: fix file path.
    dd = fparams[Z-1, :] #in python it starts from 0. When user inputs Z=1, we want Z=0 (Hydrogen), it should be 0th row.
    return dd