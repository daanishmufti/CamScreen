import numpy as np
from collections import deque

def new_filter(alpha=0.18, buf_len=5, max_move=0.18):
    return {"alpha": alpha, "buf": deque(maxlen=buf_len), "prev": None, "max_move": max_move}

def update_filter(f, detected):
    if detected is None:
        if f["prev"] is None and len(f["buf"]) == f["buf"].maxlen:
            f["prev"] = np.median(np.stack(f["buf"]), axis=0)
        return None if f["prev"] is None else [tuple(p) for p in f["prev"].astype(int)]
    det = np.array(detected, dtype=np.float32)
    f["buf"].append(det)
    if f["prev"] is None:
        f["prev"] = np.median(np.stack(f["buf"]), axis=0) if len(f["buf"]) >= 2 else det.copy()
        return [tuple(p) for p in f["prev"].astype(int)]
    w = np.linalg.norm(f["prev"][0] - f["prev"][1])
    if np.max(np.linalg.norm(det - f["prev"], axis=1)) > max(4.0, w * f["max_move"]):
        return [tuple(p) for p in f["prev"].astype(int)]
    f["prev"] = (1.0 - f["alpha"]) * f["prev"] + f["alpha"] * det
    return [tuple(p) for p in f["prev"].astype(int)]
