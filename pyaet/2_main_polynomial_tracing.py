import numpy as np
from scipy.interpolate import interp3
from scipy.ndimage import label, generate_binary_structure
from scipy.optimize import least_squares
import math


def main_polynomial_tracing(Dsetvol_file_path):
    # Add path for user-defined functions
    # addpath('src/')

    # Add the path to load the reconstruction volume (you can comment it and move
    # the reconstruction into an input folder)
    # addpath('../3_Final_reconstruction_volume/') ;

    # Read in files: reconstruction volume
    Dsetvol = np.load(Dsetvol_file_path)  # Assuming it's a numpy file

    # Constants
    MaxIter = 14
    CritIter = 7
    Th = 1
    Res = 0.347 / 3
    minDist = 2 / Res
    SearchRad = 3

    # Upsample the reconstruction matrix by 3*3*3 using linear interpolation
    xx = np.arange(1, Dsetvol.shape[0] + 1) - int((Dsetvol.shape[0] + 1) / 2)
    yy = np.arange(1, Dsetvol.shape[1] + 1) - int((Dsetvol.shape[1] + 1) / 2)
    zz = np.arange(1, Dsetvol.shape[2] + 1) - int((Dsetvol.shape[2] + 1) / 2)

    xxi = np.arange(3 * xx[0], xx[-1] * 3) / 3
    yyi = np.arange(3 * yy[0], yy[-1] * 3) / 3
    zzi = np.arange(3 * zz[0], zz[-1] * 3) / 3

    xxi = xxi[2:]
    yyi = yyi[2:]
    zzi = zzi[2:]

    Y, X, Z = np.meshgrid(yy, xx, zz)
    Yi, Xi, Zi = np.meshgrid(yyi, xxi, zzi)

    Dsetvol = interp3(X, Y, Z, Dsetvol, Xi, Yi, Zi, method='spline', fill_value=0)
    FinalVol = my_paddzero(Dsetvol, np.array(Dsetvol.shape) + 20)

    # Get polynomial power array
    fitCoeff = get_polynomial_power_array()

    # Get the local maxima from the reconstruction volume
    se = generate_binary_structure(3, 1)
    dilatedBW = label(FinalVol == dilate(FinalVol, se) & (FinalVol > Th))[0]
    maxVals = FinalVol[dilatedBW > 0]
    sortInd = np.argsort(maxVals)[::-1]
    maxNum = min(100000, len(sortInd))
    maxPos = np.argwhere(dilatedBW > 0)[sortInd[:maxNum]]

    print('numpeak =', len(maxPos))

    maxXYZ = np.zeros((len(maxPos), 3), dtype=int)
    for i in range(len(maxPos)):
        xx, yy, zz = maxPos[i]
        maxXYZ[i, :] = [xx, yy, zz]

    # Initialize parameters
    Q = 0.5
    Alpha = 1
    cropHalfSize = SearchRad
    X, Y, Z = np.meshgrid(np.arange(-cropHalfSize, cropHalfSize + 1),
                         np.arange(-cropHalfSize, cropHalfSize + 1),
                         np.arange(-cropHalfSize, cropHalfSize + 1))
    SphereInd = np.where(X**2 + Y**2 + Z**2 <= (SearchRad + 0.5)**2)
    XYZdata = {'X': X[SphereInd], 'Y': Y[SphereInd], 'Z': Z[SphereInd]}

    Orders = fitCoeff[:, :3]
    PosArr = np.zeros_like(maxXYZ, dtype=float)
    TotPosArr = np.zeros_like(maxXYZ, dtype=float)

    exitFlagArr = np.zeros(len(maxXYZ), dtype=int)
    CoeffArr = np.tile(fitCoeff[:, 3], (len(maxXYZ), 1)).T

    # Perform the main tracing loop
    for i in range(len(maxXYZ)):
        endFlag = 0
        consecAccum = 0
        iterNum = 0
        while not endFlag:
            iterNum += 1
            if iterNum > MaxIter:
                exitFlagArr[i] = -4
                endFlag = 1
            cropXind = maxXYZ[i, 0] + np.arange(-cropHalfSize, cropHalfSize + 1)
            cropYind = maxXYZ[i, 1] + np.arange(-cropHalfSize, cropHalfSize + 1)
            cropZind = maxXYZ[i, 2] + np.arange(-cropHalfSize, cropHalfSize + 1)

            cropVol = FinalVol[cropXind, cropYind, cropZind]

            Pos = PosArr[i, :]
            GaussWeight = np.exp(-1 * Alpha * ((XYZdata['X'] - Pos[0])**2 +
                                               (XYZdata['Y'] - Pos[1])**2 +
                                               (XYZdata['Z'] - Pos[2])**2) / cropHalfSize**2)

            # Define the objective function for optimization
            fun = lambda p, xdata: calculate_3D_polynomial_Rogers(xdata['X'], xdata['Y'], xdata['Z'], Pos, Orders, p) * GaussWeight

            # Initial coefficients for optimization
            p0 = CoeffArr[:, i]

            # Use scipy's least_squares function for optimization
            res = least_squares(fun, p0, args=(XYZdata,), method='trf')
            p1 = res.x
            CoeffArr[:, i] = p1

            dX, dY, dZ = calc_dX_dY_dZ_Rogers(Orders, p1)
            if dX == -100 and dY == -100 and dZ == -100:
                exitFlagArr[i] = -1
                endFlag = 1
            else:
                maxedShift = max([dX, dY, dZ], -1 * [Q, Q, Q])
                minedShift = np.minimum(maxedShift, [Q, Q, Q])
                PosArr[i, :] = PosArr[i, :] + minedShift
                if np.max(np.abs(PosArr[i, :])) > cropHalfSize:
                    exitFlagArr[i] = -2
                    endFlag = 1
                elif np.max(np.abs(minedShift)) < Q:
                    if consecAccum == CritIter - 1:
                        goodAtomTotPos = TotPosArr[:i, :]
                        goodAtomTotPos = goodAtomTotPos[exitFlagArr[:i] == 0, :]
                        Dist = np.sqrt(np.sum(
                            (goodAtomTotPos - np.tile(PosArr[i, :] + maxXYZ[i, :], (goodAtomTotPos.shape[0], 1))) ** 2,
                            axis=1))
                        if np.min(Dist) < minDist:
                            exitFlagArr[i] = -3
                        else:
                            TotPosArr[i, :] = PosArr[i, :] + maxXYZ[i, :]
                        endFlag = 1
                    else:
                        consecAccum += 1
                else:
                    consecAccum = 0

        print(f'peak {i}, flag {exitFlagArr[i]}')

    # Do raw classification and get all candidates for manual tracing
    FinalVol_single = FinalVol.astype(np.single)

    classify_info = {
        'Num_species': 3,
        'halfSize': 3,
        'plothalfSize': 1,
        'O_Ratio': 1,
        'SPHyn': 1,
        'PLOT_YN': 0,
        'separate_part': 120
    }

    atom_pos = TotPosArr[exitFlagArr == 0, :].T
    atom_pos_all = atom_pos / 3 - 2

    b1 = np.where(np.logical_or(atom_pos[0, :] < 15, atom_pos[0, :] > FinalVol_single.shape[0] - 15))[0]
    b2 = np.where(np.logical_or(atom_pos[1, :] < 15, atom_pos[1, :] > FinalVol_single.shape[1] - 15))[0]
    b3 = np.where(np.logical_or(atom_pos[2, :] < 15, atom_pos[2, :] > FinalVol_single.shape[2] - 15))[0]

    bT = np.union1d(np.union1d(b1, b2), b3)
    atom_pos[:, bT] = []

    temp_model, temp_atomtype = initial_class_kmean(
        FinalVol_single, atom_pos, classify_info)

    atom_pos_o = temp_model / 3 - 2

    # Calculate support from reconstruction and get the atoms inside
    support_para = {
        'th_dis_r_afterav': 0.90,
        'dilate_size': 15,
        'erode_size': 15,
        'bw_size': 50000
    }

    # Implement the functions obtain_tight_support and my_paddzero similarly
    tight_support1 = obtain_tight_support(Dsetvol, support_para)
    support_para['erode_size'] = 20
    tight_support2 = obtain_tight_support(Dsetvol, support_para)

    # Exclude atoms near the boundary first
    bdl_1 = 8
    bdl_2 = FinalVol_single.shape[0] - 8
    ind_out1 = np.logical_or.reduce((atom_pos_o[0, :] <= bdl_1, atom_pos_o[1, :] <= bdl_1, atom_pos_o[2, :] <= bdl_1))
    ind_out2 = np.logical_or.reduce((atom_pos_o[0, :] >= bdl_2, atom_pos_o[1, :] >= bdl_2, atom_pos_o[2, :] >= bdl_2))
    atom_pos_o[:, ind_out1 | ind_out2] = []

    # Add the missing atoms inside tighter support
    temp_pos_arr1 = []
    ind_arr1 = []
    for i in range(atom_pos_o.shape[1]):
        temp_pos = np.round(atom_pos_o[:, i]).astype(int)
        if tight_support1[temp_pos[0], temp_pos[1], temp_pos[2]] == 1:
            temp_pos_arr1.append(atom_pos_o[:, i])
            ind_arr1.append(i)

    # Exclude traced atoms outside the looser support
    ind_arr2 = []
    for i in range(atom_pos_all.shape[1]):
        if np.min(np.linalg.norm(atom_pos_all[:, i] - np.array(temp_pos_arr1).T, axis=0)) < 1e-4:
            ind_arr2.append(i)
        else:
            temp_pos = np.round(atom_pos_all[:, i]).astype(int)
            if tight_support2[temp_pos[0], temp_pos[1], temp_pos[2]] == 1:
                ind_arr2.append(i)

    temp_pos_arr2 = atom_pos_all[:, ind_arr2]
    # Save 'temp_pos_arr2' to a file or process it further


# User-defined functions

def my_paddzero(data, new_shape):
    # Implement 'my_paddzero' function logic here
    pass


def get_polynomial_power_array():
    # Implement 'get_polynomial_power_array' function logic here
    pass


def calculate_3D_polynomial_Rogers(X, Y, Z, Pos, Orders, p):
    # Implement 'calculate_3D_polynomial_Rogers' function logic here
    pass


def calc_dX_dY_dZ_Rogers(Orders, p):
    # Implement 'calc_dX_dY_dZ_Rogers' function logic here
    pass


def initial_class_kmean(FinalVol_single, atom_pos, classify_info):
    # Implement 'initial_class_kmean' function logic here
    pass


def obtain_tight_support(Dsetvol, support_para):
    # Implement 'obtain_tight_support' function logic here
    pass


# Call the main function
main_polynomial_tracing('MG_reconstruction_volume.npy')  # Pass the path to your numpy file here
