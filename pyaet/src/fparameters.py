import numpy as np
import scipy.io as sio

def fparameters(Z):
    # Initialize the fparams array with zeros
    fparams = sio.loadmat('fparameters.mat')['fparams']
    dd = fparams(Z-1,:) #in python it starts from 0. When user inputs Z=1, we want Z=0 (Hydrogen), it should be 0th row.
    return dd
