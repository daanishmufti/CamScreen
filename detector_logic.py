import cv2
import numpy as np
import settings as _s

def compute_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def _detect_v_pair(segs, fw):
    slen = lambda s: float(np.hypot(s[2] - s[0], s[3] - s[1]))
    min_a = float(_s.get("lsd_v_min_deg"))
    vlines = [s for s in segs if min_a <= float(np.degrees(np.arctan2(s[3]-s[1], s[2]-s[0])) % 180) <= 180 - min_a]
    vlines.sort(key=slen, reverse=True)
    n = int(_s.get("lsd_v_candidates"))
    tol = float(_s.get("lsd_v_match_tol"))
    mid = fw / 2.0
    left = [s for s in vlines if (s[0]+s[2])/2.0 < mid][:n]
    right = [s for s in vlines if (s[0]+s[2])/2.0 >= mid][:n]
    if not left or not right:
        return None, None
    bl, br, ba = None, None, 0.0
    height_tol = 0.05 
    parallel_angle_tol = 2.0  
    def seg_angle(s):
        return float(np.degrees(np.arctan2(s[3]-s[1], s[2]-s[0])) % 180)
    def seg_height(s):
        return abs(s[3] - s[1])
    for l in left:
        for r in right:
            angle_diff = abs(seg_angle(l) - seg_angle(r))
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            if angle_diff > parallel_angle_tol:
                continue
            ll, rl = seg_height(l), seg_height(r)
            avg_h = (ll + rl) / 2.0
            if avg_h == 0:
                continue
            if abs(ll - rl) / avg_h > height_tol:
                continue
            len_l, len_r = slen(l), slen(r)
            a = (len_l + len_r) / 2.0
            if abs(len_l - len_r) / max(a, 1e-6) <= tol and a > ba:
                bl, br, ba = l, r, a
    if bl is None:
        bl, br = max(left, key=slen), max(right, key=slen)
    return bl, br

def lsd_lines_vis(edge_map):
    h, w = edge_map.shape[:2]
    raw, *_ = cv2.createLineSegmentDetector(0).detect(edge_map)
    segs = raw.reshape(-1, 4) if raw is not None else np.empty((0, 4), dtype=np.float32)
    vis = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
    for s in segs:
        cv2.line(vis, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (55, 55, 55), 1)
    ll, rl = _detect_v_pair(segs, w)
    if ll is not None:
        cv2.line(vis, (int(ll[0]), int(ll[1])), (int(ll[2]), int(ll[3])), (0, 255, 0), 2)
    if rl is not None:
        cv2.line(vis, (int(rl[0]), int(rl[1])), (int(rl[2]), int(rl[3])), (0, 200, 80), 2)
    return vis

def find_screen_quad(frame, edge_mask=None):
    h, w = frame.shape[:2]
    if edge_mask is None:
        edge_mask = compute_edges(frame)
    raw, *_ = cv2.createLineSegmentDetector(0).detect(edge_mask)
    if raw is None:
        return None
    ll, rl = _detect_v_pair(raw.reshape(-1, 4), w)
    if ll is None or rl is None:
        return None
    top = lambda ln: (float(ln[0]), float(ln[1])) if ln[1] <= ln[3] else (float(ln[2]), float(ln[3]))
    bot = lambda ln: (float(ln[2]), float(ln[3])) if ln[3] >= ln[1] else (float(ln[0]), float(ln[1]))
    clamp = lambda p: (int(max(0, min(w-1, round(p[0])))), int(max(0, min(h-1, round(p[1])))))
    return [clamp(top(ll)), clamp(top(rl)), clamp(bot(rl)), clamp(bot(ll))]

def validate_quad(quad, shape):
    if quad is None:
        return 'no_laptop'
    tl, tr, br, bl = quad
    w = np.linalg.norm(np.array(tr) - np.array(tl))
    h = np.linalg.norm(np.array(bl) - np.array(tl))
    if w < shape[1] * 0.15 or h < shape[0] * 0.10:
        return 'no_laptop'
    aspect = w / max(h, 1e-6)
    if aspect < 1.0 or aspect > 2.5:
        return 'no_laptop'
    def angle(a, b, c):
        ab = np.array(b) - np.array(a)
        cb = np.array(b) - np.array(c)
        cos_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    angles = [angle(tl, tr, br), angle(tr, br, bl), angle(br, bl, tl), angle(bl, tl, tr)]
    if any(abs(a - 90) > 25 for a in angles):
        return 'no_laptop'
    return 'ok'
