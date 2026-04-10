import cv2
import time
import numpy as np
from detector_logic import compute_edges, find_screen_quad, validate_quad
from transform import four_points_transform
from quad import make_quad_filter
import settings

RETRY_INTERVAL = 1.0
LOCK_ON_DURATION = 2.0

def run_screen_detector():
    camera = cv2.VideoCapture(settings.camera_index)
    previous_time = time.time()
    quad_filter = make_quad_filter()

    state_searching = True
    next_retry = 0.0
    lock_start_time = None

    while True:
        got_frame, frame = camera.read()
        if not got_frame:
            break

        current_time = time.time()
        fps = 1.0 / max(current_time - previous_time, 1e-9)
        previous_time = current_time

        edge_map = compute_edges(frame)

        if not state_searching:
            quad = find_screen_quad(frame, edge_mask=edge_map)
            status = validate_quad(quad, frame.shape)

            if status == 'ok':
                smoothed = quad_filter(quad)
                if smoothed is None:
                    state_searching = True
                    next_retry = current_time
                    quad_filter = make_quad_filter()
                    lock_start_time = None
            else:
                state_searching = True
                next_retry = current_time
                quad_filter = make_quad_filter()
                smoothed = None
                lock_start_time = None

        if state_searching:
            smoothed = None
            if current_time >= next_retry:
                quad = find_screen_quad(frame, edge_mask=edge_map)
                status = validate_quad(quad, frame.shape)
                if status == 'ok':
                    state_searching = False
                    quad_filter = make_quad_filter()
                    smoothed = quad_filter(quad)
                    lock_start_time = current_time
                else:
                    next_retry = current_time + RETRY_INTERVAL

        warped = None
        if smoothed is not None:
            warped = four_points_transform(frame, np.array(smoothed, dtype=np.float32))

        lock_remaining = max(0.0, LOCK_ON_DURATION - (current_time - lock_start_time)) if lock_start_time is not None else 0.0
        locked = lock_start_time is not None and lock_remaining == 0

        win_w, win_h = (720, 540)
        if warped is not None and locked:
            display = cv2.resize(warped, (win_w, win_h))
            edges_w = compute_edges(display)
            black_ratio = np.sum(edges_w < 20) / edges_w.size
            is_on = black_ratio <= 0.85
            state_label = "Screen ON" if is_on else "Screen OFF"
            state_color = (0, 255, 0) if is_on else (0, 0, 255)
            (tw, th), _ = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(display, state_label, ((win_w - tw) // 2, win_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2, cv2.LINE_AA)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Warped", display)
        else:
            placeholder = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            msg = f"Locking on... {lock_remaining:.1f}s" if lock_remaining > 0 else "No Screen Detected"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(placeholder, msg, ((win_w - tw) // 2, (win_h + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Warped", placeholder)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    camera.release()
    cv2.destroyAllWindows()
