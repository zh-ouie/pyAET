import os
import time
import numpy as np
from pyaet.src.strel3d import strel3d
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label, uniform_filter
from scipy.signal import fftconvolve
from pyaet.src.otsu_thresh_3D import otsu_thresh_3D


def _matlab_even_origin(structure):
    return tuple(-1 if size % 2 == 0 else 0 for size in structure.shape)


def _profile_enabled():
    value = os.environ.get("PYAET_SUPPORT_PROFILE", os.environ.get("PYAET_TRACING_PROFILE", "0"))
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _profile_print(label, start_time):
    if _profile_enabled():
        print(f"    support_{label}_s = {time.perf_counter() - start_time:.3f}", flush=True)


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def _distance_transform(mask):
    backend = os.environ.get("PYAET_SUPPORT_EDT_BACKEND", "auto").strip().lower()
    if backend in ("edt", "auto"):
        try:
            import edt
            return edt.edt(
                mask,
                black_border=False,
                parallel=_env_int("PYAET_SUPPORT_EDT_THREADS", os.cpu_count() or 1),
            )
        except Exception:
            if backend == "edt":
                raise
    return distance_transform_edt(mask)


def _remove_small_objects(mask, min_size, connectivity):
    if connectivity != mask.ndim:
        raise ValueError(f"Unsupported connectivity={connectivity}; expected {mask.ndim}")
    structure = np.ones((3,) * mask.ndim, dtype=bool)
    labels, num_labels = label(mask, structure=structure)
    if num_labels == 0:
        return mask
    counts = np.bincount(labels.ravel())
    keep = counts >= min_size
    keep[0] = False
    return keep[labels]


def _support_before_erode(RECvol, para_info):
    th_dis_r_afterav = para_info.get('th_dis_r_afterav', 0.90)
    dilate_size = para_info.get('dilate_size', 11)

    # Smooth the volume
    t_step = time.perf_counter()
    curr_RECvol = uniform_filter(RECvol.astype(np.float32, copy=False), size=9, mode='nearest').astype(np.float32, copy=False)
    _profile_print("smooth", t_step)

    # Otsu threshold
    t_step = time.perf_counter()
    max_val = np.float32(curr_RECvol.max())
    im = (curr_RECvol / max_val * np.float32(255.0)).astype(np.float32, copy=False)
    ot, _ = otsu_thresh_3D(im)
    ot = np.float32(ot) * max_val / np.float32(255.0) * np.float32(th_dis_r_afterav)

    curr_support = curr_RECvol > ot
    _profile_print("threshold", t_step)
    if _env_flag("PYAET_SUPPORT_EDT_DILATE", True) and int(dilate_size) == 15:
        # Make the mask slightly larger
        t_step = time.perf_counter()
        se = strel3d(3)
        curr_support = binary_dilation(curr_support, structure=se.astype(bool), border_value=0)
        _profile_print("dilate3", t_step)

        t_step = time.perf_counter()
        if np.any(curr_support):
            # strel3d(15) is the integer lattice ball with radius 7.
            # The distance-transform threshold matches binary_dilation with
            # border_value=0 for this spherical structuring element.
            curr_support = _distance_transform(~curr_support) <= 7.0
        _profile_print("dilate15_edt", t_step)
    else:
        # Make the mask slightly larger
        t_step = time.perf_counter()
        se = strel3d(3)
        curr_support = binary_dilation(curr_support, structure=se.astype(bool), border_value=0)
        _profile_print("dilate3", t_step)

        # Make the mask quite larger
        t_step = time.perf_counter()
        se = strel3d(dilate_size)
        curr_support = binary_dilation(curr_support, structure=se.astype(bool), border_value=0)
        _profile_print(f"dilate{dilate_size}", t_step)

    if 'bw_size' in para_info:
        #remove small isolated area by bw_size
        t_step = time.perf_counter()
        bw_size = para_info['bw_size']
        # curr_support = scipy.ndimage.binary_opening(curr_support, structure=np.ones((3, 3, 3)), iterations=bw_size)
        curr_support = _remove_small_objects(curr_support.astype(bool, copy=False), bw_size, connectivity=RECvol.ndim)
        _profile_print("remove_small", t_step)

    return curr_support


def prepare_tight_support_base(RECvol, para_info):
    """Return the shared support mask before the final erosion step."""
    return _support_before_erode(RECvol, para_info)


def obtain_tight_support(RECvol, para_info):
    """
    Obtain tight support.

    Parameters:
    - RECvol (numpy.ndarray): Reconstruction volume.
    - para_info (dict): Parameters for support generation.

    Returns:
    - curr_support (numpy.ndarray): Tight support.
    """

    erode_size = para_info.get('erode_size', 11)
    curr_support = _support_before_erode(RECvol, para_info)

    # Make the mask quite smaller
    se = strel3d(erode_size)
    curr_support = binary_erosion(
        curr_support,
        structure=se.astype(bool),
        border_value=1,
        origin=_matlab_even_origin(se),
    )

    return curr_support


def _binary_erosion_matlab(mask, structure):
    return binary_erosion(
        mask,
        structure=structure.astype(bool, copy=False),
        border_value=1,
        origin=_matlab_even_origin(structure),
    )


def _fft_binary_erosion(mask, structure):
    blocked = fftconvolve(
        (~mask).astype(np.float32, copy=False),
        structure.astype(np.float32, copy=False),
        mode='same',
    )
    return blocked < 0.5


def _structure_offsets(structure):
    center = np.asarray(structure.shape, dtype=np.int64) // 2
    center -= (np.asarray(structure.shape, dtype=np.int64) % 2 == 0)
    return (
        np.argwhere(structure).astype(np.int64, copy=False)
        - center
    )


def eroded_support_contains_points(curr_support, points, erode_size):
    """
    Evaluate binary erosion at selected point coordinates.

    This is equivalent to indexing
    binary_erosion(curr_support, strel3d(erode_size), border_value=1)
    at round(points) - 1, but avoids materializing the full eroded volume.
    """

    t_start = time.perf_counter()
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[0] != 3:
        raise ValueError("points must have shape (3, N)")
    inside = np.zeros(points.shape[1], dtype=bool)
    if points.size == 0:
        _profile_print(f"point_erode{erode_size}", t_start)
        return inside

    coords = np.round(points).astype(np.int64, copy=False) - 1
    shape = np.asarray(curr_support.shape, dtype=np.int64)
    valid = np.all((coords >= 0) & (coords < shape[:, None]), axis=0)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        _profile_print(f"point_erode{erode_size}", t_start)
        return inside

    pts = coords[:, valid].T
    structure = strel3d(erode_size).astype(bool, copy=False)
    offsets = _structure_offsets(structure)
    chunk_size = int(os.environ.get("PYAET_SUPPORT_POINT_CHUNK", "2048"))
    offset_batch = int(os.environ.get("PYAET_SUPPORT_OFFSET_BATCH", "256"))

    for start in range(0, pts.shape[0], chunk_size):
        stop = min(start + chunk_size, pts.shape[0])
        pts_chunk = pts[start:stop]
        good = np.ones(stop - start, dtype=bool)
        for off_start in range(0, offsets.shape[0], offset_batch):
            if not np.any(good):
                break
            off_stop = min(off_start + offset_batch, offsets.shape[0])
            neigh = pts_chunk[:, None, :] + offsets[off_start:off_stop][None, :, :]
            in_bounds = np.all((neigh >= 0) & (neigh < shape[None, None, :]), axis=2)
            values = np.ones(in_bounds.shape, dtype=bool)
            if np.any(in_bounds):
                sample = neigh[in_bounds]
                values[in_bounds] = curr_support[sample[:, 0], sample[:, 1], sample[:, 2]]
            good &= np.all(values, axis=1)
        inside[valid_indices[start:stop]] = good

    _profile_print(f"point_erode{erode_size}", t_start)
    return inside


def obtain_tight_support_multi(RECvol, para_info, erode_sizes):
    """
    Return multiple tight supports that differ only by erosion size.

    The expensive smoothing, thresholding, dilation, and connected-component
    cleanup are shared; this is equivalent to calling obtain_tight_support
    repeatedly with the same parameters except erode_size.
    """

    curr_support = _support_before_erode(RECvol, para_info)
    return tuple(
        _binary_erosion_matlab(curr_support, strel3d(erode_size).astype(bool))
        for erode_size in erode_sizes
    )
