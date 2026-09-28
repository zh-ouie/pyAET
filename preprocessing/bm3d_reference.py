"""BM3D denoising with DCT/Haar transforms and ordered block matching.

Inputs and sigma use the same intensity scale. The normal-noise profile uses
single-precision matching and filtering, with explicit search boundaries,
tie handling and adaptive sampling parameters.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.fft import dctn, idctn
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class StageParameters:
    block: int
    step: int = 2
    max_group: int = 16
    radius: int = 50
    match_threshold: float = 600 / 255 ** 2
    threshold: float = 3.4
    beta: float = 0.2
    adaptive_nonzero: int = 0
    include_last: bool = True
    dynamic_positions: bool = False
    minimum_nonzero_weight: int = 1
    nonzero_weight_offset: int = 0
    clip_blocks: bool = False
    singleton_weight_offset: int = 0
    legacy_ht_rules: bool = False
    legacy_matching: str = ''


HT_PARAMETERS = StageParameters(block=8)
WIENER_PARAMETERS = StageParameters(block=16, match_threshold=400 / 255 ** 2, beta=0.5)


@lru_cache(maxsize=8)
def haar_matrix(size):
    """Orthonormal dyadic Haar; rows ordered coarse to fine."""
    if size < 1 or size & (size - 1):
        raise ValueError('Haar group size must be a positive power of two')
    if size == 1:
        matrix = np.ones((1, 1))
    else:
        matrix = np.vstack((np.kron(haar_matrix(size // 2), [[1., 1.]]),
                            np.kron(np.eye(size // 2), [[1., -1.]]))) / np.sqrt(2.)
    matrix.setflags(write=False)
    return matrix


def filter_group(noisy, sigma, threshold=3.4, pilot=None, minimum_nonzero_weight=1,
                 nonzero_weight_offset=0, legacy_ht_rules=False, wiener_weight_epsilon=0.):
    """Return filtered pixels, weight and the coefficient counter used for sampling.

    With legacy_ht_rules the counter includes the historical branch-specific
    adjustment; it is not literally a count of nonzero numerical values.
    """
    noisy = np.asarray(noisy, dtype=float)
    transform = haar_matrix(len(noisy))
    spectrum = dctn(noisy, axes=(-2, -1), norm='ortho')
    coefficients = np.einsum('ij,jxy->ixy', transform, spectrum)
    if pilot is None:
        keep = np.abs(coefficients) > threshold * sigma
        # Legacy MEX probes isolate two different Haar branches. For N>=4
        # the group DC is always retained and counted, even when it is zero.
        # N=1/2 thresholds every coefficient but starts its counter at one.
        if legacy_ht_rules and len(noisy) >= 4:
            keep[0, 0, 0] = True
        coefficients *= keep
        nonzero = int(keep.sum()) + int(legacy_ht_rules and len(noisy) <= 2)
        weight = 1. / max(nonzero + nonzero_weight_offset, minimum_nonzero_weight)
    else:
        pilot_spectrum = dctn(pilot, axes=(-2, -1), norm='ortho')
        estimate = np.einsum('ij,jxy->ixy', transform, pilot_spectrum)
        square = estimate ** 2
        gain = np.divide(square, square + sigma ** 2,
                         out=np.ones_like(square), where=(square + sigma ** 2) != 0)
        coefficients *= gain
        total = float(np.sum(gain ** 2)) + wiener_weight_epsilon
        weight = 1. / total if total else 1.
        nonzero = int(np.count_nonzero(gain))
    filtered = idctn(np.einsum('ji,jxy->ixy', transform, coefficients),
                    axes=(-2, -1), norm='ortho')
    return filtered, weight, nonzero


def _positions(count, step, include_last):
    positions = list(range(0, count, step))
    if include_last and positions[-1] != count - 1:
        positions.append(count - 1)
    return positions


@lru_cache(maxsize=4)
def _legacy_dct_matrix(size):
    matrix = dctn(np.eye(size), axes=(0,), norm='ortho').astype(np.float32)
    matrix.setflags(write=False)
    return matrix


def _legacy_matching_vectors(patches, mode):
    """Preserve the legacy single-precision operation order for tied matches.

    HT is called on a transposed image; undo that within each patch. Its MEX
    matches centered DCT spectra plus the separately accumulated patch mean.
    Wiener uses uncentered spectra. Algebraically equivalent pixel distances
    do not preserve the rounding and therefore can choose different groups.
    """
    a = (patches.swapaxes(-1, -2) if mode == 'ht' else patches).astype(np.float32)
    size = a.shape[-1]
    means = np.zeros(a.shape[:-2], dtype=np.float32)
    if mode == 'ht':
        flat = a.reshape(*a.shape[:-2], -1)
        for k in range(size * size):
            means += flat[..., k]
        means /= np.float32(size * size)
        a -= means[..., None, None]
    matrix = _legacy_dct_matrix(size)
    intermediate = a[..., 0, None] * matrix[:, 0]
    for k in range(1, size):
        intermediate += a[..., k, None] * matrix[:, k]
    spectrum = matrix[:, 0, None] * intermediate[..., 0, None, :]
    for k in range(1, size):
        spectrum += matrix[:, k, None] * intermediate[..., k, None, :]
    return np.concatenate((means[..., None], spectrum.reshape(*a.shape[:-2], -1)), axis=-1)


def _legacy_distances(reference, candidates, block):
    delta = reference[0] - candidates[:, 0]
    distance = ((delta * delta) * np.float32(block)) * np.float32(block)
    for k in range(1, candidates.shape[1]):
        delta = reference[k] - candidates[:, k]
        distance += delta * delta
    return distance


def stage(image, sigma, parameters=HT_PARAMETERS, pilot=None, trace=False):
    """Denoise one stage with clipped valid-patch search and deterministic ties.

    Generic reference traversal is column-major. The legacy candidate transposes
    HT input and selects row-major Wiener traversal to match the original kernels.
    Adaptive inner-loop skipping advances B-1 pixels when the filter's
    coefficient counter is below the configured threshold. The legacy candidate
    enables the empirically validated MEX counter rules as well.
    """
    image = np.asarray(image, dtype=float)
    p = parameters
    if image.ndim != 2 or not np.isfinite(image).all() or not np.isfinite(sigma) or sigma < 0:
        raise ValueError('Expected a finite 2-D image and nonnegative sigma')
    if (p.block < 1 or p.block > min(image.shape) or p.step < 1 or p.radius < 0
            or p.max_group < 1 or p.max_group & (p.max_group - 1)):
        raise ValueError('Invalid block, step, radius or group size')
    if pilot is not None:
        pilot = np.asarray(pilot, dtype=float)
        if pilot.shape != image.shape or not np.isfinite(pilot).all():
            raise ValueError('Pilot must be finite and have the input image shape')
    patches = np.lib.stride_tricks.sliding_window_view(image, (p.block, p.block))
    matching = patches if pilot is None else np.lib.stride_tricks.sliding_window_view(pilot, (p.block, p.block))
    nr, nc = patches.shape[:2]
    # A column-major patch index makes tie-breaking and trace coordinates explicit.
    vectors = np.ascontiguousarray(matching.transpose(1, 0, 2, 3).reshape(nr * nc, -1))
    if p.legacy_matching:
        if p.legacy_matching not in ('ht', 'wiener'):
            raise ValueError('Unknown legacy matching mode')
        vectors = np.ascontiguousarray(
            _legacy_matching_vectors(matching, p.legacy_matching).transpose(1, 0, 2).reshape(nr * nc, -1))
    noisy_vectors = None
    if p.legacy_matching == 'wiener':
        noisy_vectors = np.ascontiguousarray(
            _legacy_matching_vectors(patches, 'wiener').transpose(1, 0, 2).reshape(nr * nc, -1))
    accumulator_dtype = np.float32 if p.legacy_matching else float
    numerator = np.zeros_like(image, dtype=accumulator_dtype)
    denominator = np.zeros_like(image, dtype=accumulator_dtype)
    window = np.outer(np.kaiser(p.block, p.beta), np.kaiser(p.block, p.beta))
    window = window.astype(accumulator_dtype)
    visits = []
    group_histogram = {}
    rows = _positions(nr, p.step, p.include_last)
    cols = _positions(nc, p.step, p.include_last)
    select_matches = None
    if p.legacy_matching:
        from .bm3d_matching_accel import select_matches
    processed = 0
    # Both kernels visit image rows first. HT already receives a transposed
    # image, whereas Wiener needs its reference loops exchanged explicitly.
    row_major = p.legacy_matching == 'wiener'
    outer_positions, inner_positions = (rows, cols) if row_major else (cols, rows)
    for outer in outer_positions:
        next_row = 0
        current_rows = list(inner_positions)
        row_index = 0
        while row_index < len(current_rows):
            inner = current_rows[row_index]
            row, col = (outer, inner) if row_major else (inner, outer)
            row_index += 1
            if not p.dynamic_positions and row < next_row:
                continue
            rr = np.arange(max(0, row - p.radius), min(nr, row + p.radius + 1))
            cc = np.arange(max(0, col - p.radius), min(nc, col + p.radius + 1))
            candidates = (cc[:, None] * nr + rr[None, :]).ravel()
            if p.legacy_matching == 'wiener':
                candidates = (cc[None, :] * nr + rr[:, None]).ravel()
            ref = col * nr + row
            if select_matches is not None:
                selected = select_matches(vectors, candidates, ref, p.block, p.max_group,
                                          np.float32(p.match_threshold * p.block ** 2))
                size = len(selected)
            else:
                if p.legacy_matching:
                    distance = _legacy_distances(vectors[ref], vectors[candidates], p.block)
                else:
                    distance = cdist(vectors[ref:ref + 1], vectors[candidates], 'sqeuclidean')[0]
                eligible = distance < p.match_threshold * p.block ** 2
                eligible[candidates == ref] = True
                ids, distances = candidates[eligible], distance[eligible]
                # Always retain the reference, then sort other candidates by distance/index.
                other = ids != ref
                order = np.argsort(distances[other], kind='stable')
                ids = np.concatenate(([ref], ids[other][order]))
                size = 1 << (min(len(ids), p.max_group).bit_length() - 1)
                selected = ids[:size]
            gr, gc = selected % nr, selected // nr
            if p.legacy_matching == 'ht':
                from .bm3d_legacy_numeric import hard_filter
                filtered, weight, nonzero = hard_filter(vectors[selected], p.block, sigma,
                                                       p.threshold, _legacy_dct_matrix(p.block))
            elif p.legacy_matching == 'wiener':
                from .bm3d_legacy_numeric import wiener_filter
                epsilon = np.float32((2 * p.radius + 1) ** 2 * p.block ** 2) * np.float32(1e-38)
                filtered, weight, nonzero = wiener_filter(noisy_vectors[selected], vectors[selected],
                    p.block, sigma, epsilon, _legacy_dct_matrix(p.block))
            else:
                filtered, weight, nonzero = filter_group(
                    patches[gr, gc], sigma, p.threshold,
                    None if pilot is None else matching[gr, gc], p.minimum_nonzero_weight,
                    p.singleton_weight_offset if size == 1 else p.nonzero_weight_offset,
                    legacy_ht_rules=p.legacy_ht_rules)
            if p.clip_blocks:
                filtered = np.clip(filtered, 0., 1.)
            for k, (r, c) in enumerate(zip(gr, gc)):
                weighted_window = window * weight
                numerator[r:r + p.block, c:c + p.block] += filtered[k] * weighted_window
                denominator[r:r + p.block, c:c + p.block] += weighted_window
            processed += 1
            group_histogram[size] = group_histogram.get(size, 0) + 1
            if trace:
                visits.append(dict(reference=[row, col], selected=np.column_stack((gr, gc)).tolist(),
                                   nonzero=nonzero, weight=weight))
            if pilot is None and nonzero < p.adaptive_nonzero:
                next_row = row + p.block - 1
            elif p.dynamic_positions:
                next_row = row + p.step
            if p.dynamic_positions:
                current_rows = [next_row] if next_row < nr else (
                    [nr - 1] if p.include_last and row != nr - 1 else [])
                row_index = 0
    if np.any(denominator == 0):
        raise ValueError('Sampling left uncovered pixels; decrease step or include last reference')
    result = numerator / denominator
    return result, dict(references=processed, group_histogram=group_histogram,
                        visits=visits, min_denominator=float(denominator.min()))


def denoise(image, sigma, ht=HT_PARAMETERS, wiener=WIENER_PARAMETERS, trace=False):
    hard, hard_info = stage(image, sigma, ht, trace=trace)
    final, final_info = stage(image, sigma, wiener, pilot=hard, trace=trace)
    return final, hard, dict(hard=hard_info, final=final_info)


def legacy_candidate(image, sigma, trace=False):
    """Legacy normal-noise profile with ordered float32 matching and filtering.

    Uses a search radius of 50 and dynamic B-1 sampling.
    Supports normalized sigma times 255 up to 40.
    """
    from dataclasses import replace
    if sigma * 255 > 40:
        raise ValueError('Readable candidate currently implements the legacy normal-noise profile only (sigma*255 <= 40)')
    p = replace(HT_PARAMETERS, adaptive_nonzero=8, dynamic_positions=True,
                clip_blocks=True, legacy_ht_rules=True, legacy_matching='ht')
    hard, hard_info = stage(np.asarray(image).T, sigma, p, trace=trace)
    hard = hard.T
    final, final_info = stage(image, sigma, replace(WIENER_PARAMETERS, clip_blocks=True, legacy_matching='wiener'),
                             pilot=hard, trace=trace)
    return final, hard, dict(hard=hard_info, final=final_info)
