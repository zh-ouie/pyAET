import numpy as np


def interp2_bilinear(pj, rot_x, rot_y, origin_offset=1):
    pj = pj.astype(np.float64, copy=False)
    rot_x = rot_x.astype(np.float64, copy=False)
    rot_y = rot_y.astype(np.float64, copy=False)
    nrows, ncols = pj.shape

    dataF = np.asfortranarray(pj).ravel(order="F")

    x = rot_x - origin_offset
    y = rot_y - origin_offset
    x1 = np.floor(x).astype(np.int64)
    y1 = np.floor(y).astype(np.int64)
    x2 = x1 + 1
    y2 = y1 + 1

    mask_x_last = x == (nrows - 1)
    mask_y_last = y == (ncols - 1)
    x2 = np.where(mask_x_last, x2 - 1, x2)
    x1 = np.where(mask_x_last, x1 - 1, x1)
    y2 = np.where(mask_y_last, y2 - 1, y2)
    y1 = np.where(mask_y_last, y1 - 1, y1)

    oob = (x1 < 0) | (x2 > (nrows - 1)) | (y1 < 0) | (y2 > (ncols - 1))

    x1c = np.clip(x1, 0, nrows - 1)
    x2c = np.clip(x2, 0, nrows - 1)
    y1c = np.clip(y1, 0, ncols - 1)
    y2c = np.clip(y2, 0, ncols - 1)

    idx11 = x1c + y1c * nrows
    idx12 = x1c + y2c * nrows
    idx21 = x2c + y1c * nrows
    idx22 = x2c + y2c * nrows

    f11 = dataF[idx11]
    f12 = dataF[idx12]
    f21 = dataF[idx21]
    f22 = dataF[idx22]

    wx1 = x2 - x
    wx2 = x - x1
    wy1 = y2 - y
    wy2 = y - y1

    a = f11 * wx1 + f21 * wx2
    b = f12 * wx1 + f22 * wx2
    result = a * wy1 + b * wy2
    result[oob] = 0.0
    return result
