import cv2
import numpy as np
import settings as _settings

MIN_QUAD_W_FRAC = 0.15
MIN_QUAD_H_FRAC = 0.10
LSD_H_MAX_DEG = 30
LSD_V_MIN_DEG = 60
LSD_V_CANDIDATES = 8
LSD_V_MATCH_TOL = 0.30


def compute_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = cv2.magnitude(sobel_x, sobel_y)
    edge_strength = cv2.normalize(edge_strength, None, 0, 255, cv2.NORM_MINMAX)
    return edge_strength.astype(np.uint8)


def _run_lsd(edge_map):
    lsd = cv2.createLineSegmentDetector(0)
    detected_lines, *_ = lsd.detect(edge_map)
    if detected_lines is None:
        return np.empty((0, 4), dtype=np.float32)
    return detected_lines.reshape(-1, 4)


def _seg_length(seg):
    x1, y1, x2, y2 = seg
    return float(np.hypot(x2 - x1, y2 - y1))


def _seg_angle_deg(seg):
    x1, y1, x2, y2 = seg
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)


def _classify_segs(segs):
    min_vertical_angle = float(_settings.get("lsd_v_min_deg") or LSD_V_MIN_DEG)
    horizontal_lines = []
    vertical_lines = []

    for line in segs:
        angle = _seg_angle_deg(line)
        if angle <= LSD_H_MAX_DEG or angle >= 180 - LSD_H_MAX_DEG:
            horizontal_lines.append(line)
        elif min_vertical_angle <= angle <= 180 - min_vertical_angle:
            vertical_lines.append(line)

    horizontal_lines.sort(key=_seg_length, reverse=True)
    vertical_lines.sort(key=_seg_length, reverse=True)
    return horizontal_lines, vertical_lines


def _pick_v_pair(v_segs, frame_width):
    max_candidates = int(_settings.get("lsd_v_candidates") or LSD_V_CANDIDATES)
    match_tolerance = float(_settings.get("lsd_v_match_tol") or LSD_V_MATCH_TOL)
    middle_x = frame_width / 2.0

    left_lines = [line for line in v_segs if (line[0] + line[2]) / 2.0 < middle_x][:max_candidates]
    right_lines = [line for line in v_segs if (line[0] + line[2]) / 2.0 >= middle_x][:max_candidates]

    if not left_lines or not right_lines:
        return None, None

    best_left_line = None
    best_right_line = None
    best_average_length = 0.0

    for left_line in left_lines:
        for right_line in right_lines:
            left_length = _seg_length(left_line)
            right_length = _seg_length(right_line)
            average_length = (left_length + right_length) / 2.0
            length_difference = abs(left_length - right_length) / max(average_length, 1e-6)

            if length_difference <= match_tolerance and average_length > best_average_length:
                best_left_line = left_line
                best_right_line = right_line
                best_average_length = average_length

    if best_left_line is None:
        best_left_line = max(left_lines, key=_seg_length)
        best_right_line = max(right_lines, key=_seg_length)

    return best_left_line, best_right_line


def lsd_lines_vis(edge_map):
    frame_width = edge_map.shape[1]
    segs = _run_lsd(edge_map)
    vis = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)

    for line in segs:
        x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
        cv2.line(vis, (x1, y1), (x2, y2), (55, 55, 55), 1)

    _, vertical_lines = _classify_segs(segs)
    left_line, right_line = _pick_v_pair(vertical_lines, frame_width)

    if left_line is not None:
        x1, y1, x2, y2 = int(left_line[0]), int(left_line[1]), int(left_line[2]), int(left_line[3])
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if right_line is not None:
        x1, y1, x2, y2 = int(right_line[0]), int(right_line[1]), int(right_line[2]), int(right_line[3])
        cv2.line(vis, (x1, y1), (x2, y2), (0, 200, 80), 2)
    return vis


def find_screen_quad(frame, edge_mask=None):
    frame_height, frame_width = frame.shape[:2]
    if edge_mask is None:
        edge_mask = compute_edges(frame)

    segs = _run_lsd(edge_mask)
    if len(segs) == 0:
        return None

    _, vertical_lines = _classify_segs(segs)
    left_line, right_line = _pick_v_pair(vertical_lines, frame_width)
    if left_line is None or right_line is None:
        return None

    def top_point(line):
        if line[1] <= line[3]:
            return (float(line[0]), float(line[1]))
        return (float(line[2]), float(line[3]))

    def bottom_point(line):
        if line[3] >= line[1]:
            return (float(line[2]), float(line[3]))
        return (float(line[0]), float(line[1]))

    top_left = top_point(left_line)
    top_right = top_point(right_line)
    bottom_left = bottom_point(left_line)
    bottom_right = bottom_point(right_line)

    def clamp_point(point):
        x = int(max(0, min(frame_width - 1, round(point[0]))))
        y = int(max(0, min(frame_height - 1, round(point[1]))))
        return (x, y)

    top_left = clamp_point(top_left)
    top_right = clamp_point(top_right)
    bottom_right = clamp_point(bottom_right)
    bottom_left = clamp_point(bottom_left)
    return [top_left, top_right, bottom_right, bottom_left]


def validate_quad(quad, frame_shape):
    if quad is None:
        return 'no_laptop'
    top_left, top_right, bottom_right, bottom_left = quad
    if top_right[0] - top_left[0] < frame_shape[1] * MIN_QUAD_W_FRAC:
        return 'no_laptop'
    if bottom_left[1] - top_left[1] < frame_shape[0] * MIN_QUAD_H_FRAC:
        return 'no_laptop'
    return 'ok'
