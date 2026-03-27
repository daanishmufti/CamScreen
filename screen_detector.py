import cv2
import time
import numpy as np
from detector_logic import (
    compute_edges,
    lsd_lines_vis,
    find_screen_quad,
    validate_quad,
)
from transform import four_points_transform

from gui import show_windows
from quad import make_quad_filter
from screen_state import get_state_image
import settings

RETRY_INTERVAL = 1.0
LOCK_ON_DURATION = 2.0


def _draw_alert(frame, message):
    frame_height, frame_width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_width, 60), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, message, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


LOCKED_IN_DISPLAY = 1.0


def _draw_screen_status(img, msg):
    out = img.copy() if img is not None else np.zeros((360, 480, 3), dtype=np.uint8)
    if msg:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        h, w = out.shape[:2]
        (text_w, text_h), _ = cv2.getTextSize(msg, font, scale, thickness)
        x = (w - text_w) // 2
        y = text_h + 10
        cv2.putText(out, msg, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return out


def run_screen_detector():
    camera_index = settings.get("camera_index")
    camera = cv2.VideoCapture(camera_index)
    previous_time = time.time()
    quad_filter = make_quad_filter()

    state_searching = True  
    next_retry = 0.0
    alert_msg = 'Place laptop in view'
    lock_start_time = None

    while True:
        if settings.get("_reset_flag"):
            settings.set_val("_reset_flag", False)
            quad_filter = make_quad_filter()
            state_searching = True
            alert_msg = 'Place laptop in view'
            lock_start_time = None

        new_camera_index = settings.get("camera_index")
        if new_camera_index != camera_index:
            camera.release()
            camera_index = new_camera_index
            camera = cv2.VideoCapture(camera_index)
            quad_filter = make_quad_filter()
            state_searching = True
            alert_msg = 'Place laptop in view'
            lock_start_time = None

        if settings.get("paused"):
            is_window_open = show_windows(
                _last_frame if '_last_frame' in dir() else np.zeros((360, 480, 3), dtype=np.uint8),
                _last_lsd if '_last_lsd' in dir() else np.zeros((360, 480, 3), dtype=np.uint8),
                _last_warped if '_last_warped' in dir() else None,
                state_img=_last_state if '_last_state' in dir() else None,
                fps=0, det_status="PAUSED",
            )
            if not is_window_open:
                break
            time.sleep(0.03)
            continue

        got_frame, frame = camera.read()
        if not got_frame:
            break

        current_time = time.time()
        fps = 1.0 / max(current_time - previous_time, 1e-9)
        previous_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        edge_map = compute_edges(frame)
        lsd_vis = lsd_lines_vis(edge_map)

        if not state_searching:
            quad = find_screen_quad(frame, edge_mask=edge_map)
            status = validate_quad(quad, frame.shape)

            if status == 'ok':
                smoothed = quad_filter(quad)
                if smoothed is None:
                    state_searching = True
                    next_retry = current_time
                    alert_msg = 'Reacquiring...'
                    quad_filter = make_quad_filter()
                    lock_start_time = None
            else:
                state_searching = True
                next_retry = current_time
                alert_msg = 'Place laptop in view' if status == 'no_laptop' else 'Reposition laptop'
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
                    alert_msg = ''
                    quad_filter = make_quad_filter()
                    smoothed = quad_filter(quad)
                    lock_start_time = current_time
                else:
                    alert_msg = 'Place laptop in view' if status == 'no_laptop' else 'Reposition laptop'
                    next_retry = current_time + RETRY_INTERVAL

        if alert_msg:
            _draw_alert(frame, alert_msg)

        warped = None
        if smoothed is not None:
            quad_points = np.array(smoothed, dtype=np.int32)
            cv2.polylines(frame, [quad_points], isClosed=True, color=(0, 200, 80), thickness=2)
            for point in smoothed:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)
            warped = four_points_transform(frame, np.array(smoothed, dtype=np.float32))

        lock_remaining = max(0.0, LOCK_ON_DURATION - (current_time - lock_start_time)) if lock_start_time is not None else 0.0

        if lock_remaining > 0:
            lock_msg = f"Locking on... {lock_remaining:.1f}s"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(lock_msg, font, 0.8, 2)
            cv2.putText(frame, lock_msg, ((frame.shape[1] - tw) // 2, th + 10), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        locked = lock_start_time is not None and lock_remaining == 0
        locked_elapsed = (current_time - (lock_start_time + LOCK_ON_DURATION)) if locked else -1.0

        if lock_remaining > 0:
            warped_display = _draw_screen_status(None, "")
        elif locked and locked_elapsed < LOCKED_IN_DISPLAY:
            warped_display = _draw_screen_status(warped, "Locked In")
        elif warped is not None:
            warped_display = _draw_screen_status(warped, "Screen Detected")
        else:
            warped_display = _draw_screen_status(None, "No Screen Detected")

        state_img = get_state_image(warped) if (locked and warped is not None) else None

        if smoothed is not None:
            quad_pts = np.array(smoothed, dtype=np.int32)
            cv2.polylines(lsd_vis, [quad_pts], isClosed=True, color=(0, 200, 255), thickness=2)
            for point in smoothed:
                cv2.circle(lsd_vis, point, 5, (0, 0, 255), -1)

        _last_frame = frame
        _last_lsd = lsd_vis
        _last_warped = warped_display
        _last_state = state_img

        det_status = f"State: {'SEARCHING' if state_searching else 'TRACKING'}"
        if alert_msg:
            det_status += f"  —  {alert_msg}"

        lsd_panel = lsd_vis if settings.get("show_lsd") else None
        is_window_open = show_windows(frame, lsd_panel, warped_display, state_img=state_img, fps=fps, det_status=det_status)
        if not is_window_open:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_screen_detector()
