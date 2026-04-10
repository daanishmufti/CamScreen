import numpy as np
from collections import deque


def make_quad_filter(alpha=0.25, median_len=5, max_reset_frac=0.45): # returns a function that takes detected quad points and returns smoothed quad points, with parameters to control smoothing and reset behavior
    alpha = float(alpha)
    max_reset_frac = float(max_reset_frac)
    buf = deque(maxlen=int(median_len))
    prev = [None]

    def update(detected): 
        if detected is None:
            return None if prev[0] is None else [tuple(p) for p in prev[0].astype(int)]

        q = np.array(detected, dtype=np.float32)
        buf.append(q)

        if prev[0] is None:
            prev[0] = np.median(np.stack(buf), axis=0) if len(buf) >= 2 else q.copy()
            return [tuple(p) for p in prev[0].astype(int)]

        quad_width = np.linalg.norm(prev[0][0] - prev[0][1])
        max_reset_px = max(8.0, quad_width * max_reset_frac)
        jump = np.max(np.linalg.norm(q - prev[0], axis=1))

        if jump > max_reset_px:
            prev[0] = None
            buf.clear()
            return None

        prev[0] = (1.0 - alpha) * prev[0] + alpha * q
        return [tuple(p) for p in prev[0].astype(int)]

    return update
