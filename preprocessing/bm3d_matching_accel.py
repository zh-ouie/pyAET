"""Optional Numba acceleration of the ordered float32 legacy block search.

No fast-math or parallel reductions: changing addition order changes ties.
The NumPy path in bm3d_reference remains available without Numba.
"""
import os
import numpy as np

try:
    if os.environ.get('AET_BM3D_DISABLE_JIT') == '1':
        raise ImportError('JIT disabled for parity diagnostics')
    from numba import njit
except ImportError:
    select_matches = None
else:
    @njit(cache=True, nogil=True, fastmath=False)
    def select_matches(vectors, candidates, reference, block, max_group, threshold):
        result = np.empty(max_group, dtype=np.int64)
        result[0] = reference
        if max_group == 1:
            return result
        ids = np.empty(max_group - 1, dtype=np.int64)
        distances = np.empty(max_group - 1, dtype=np.float32)
        count = 0
        bound = np.float32(threshold)
        width = vectors.shape[1]
        for candidate in candidates:
            if candidate == reference:
                continue
            delta = vectors[reference, 0] - vectors[candidate, 0]
            distance = np.float32(np.float32(delta * delta) * np.float32(block)) * np.float32(block)
            for k in range(1, width):
                delta = vectors[reference, k] - vectors[candidate, k]
                distance = np.float32(distance + np.float32(delta * delta))
                # Nonnegative increments permit rejection before the full sum.
                if k % 16 == 0 and distance >= bound:
                    break
            if distance >= bound:
                continue
            position = count if count < max_group - 1 else max_group - 2
            while position > 0 and distance < distances[position - 1]:
                ids[position] = ids[position - 1]
                distances[position] = distances[position - 1]
                position -= 1
            ids[position] = candidate
            distances[position] = distance
            if count < max_group - 1:
                count += 1
            if count == max_group - 1:
                bound = distances[count - 1]
        size = 1
        while size * 2 <= count + 1:
            size *= 2
        result[1:size] = ids[:size - 1]
        return result[:size]
