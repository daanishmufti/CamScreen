import cv2
import time
import numpy as np
from collections import deque
from image_enhnacer import filter_image
from detector_logic import (
    compute_edges,
    lsd_lines_vis,
    find_screen_quad,
    validate_quad,
)
from transform import four_points_transform
from state import State
from gui import show_windows
import settings

RETRY_INTERVAL = 1.0


def _draw_alert(frame, message):
    frame_height, frame_width = frame.shape[:2]
    overlay = frame.copy()  # copy so the rectangle blend does not compound per-frame
    cv2.rectangle(overlay, (0, 0), (frame_width, 60), (0, 0, 180), -1)  # draw solid blue banner on the copy
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)       # blend banner onto the original frame at 65% opacity
    cv2.putText(
        frame, message, (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,  # white text over the banner
    )


class QuadFilter:
    def __init__(self, alpha=0.18, median_len=5, max_move_frac=0.18):
        self.alpha = float(alpha)                     # EMA blend factor; lower = smoother but slower to react
        self.quad_buffer = deque(maxlen=int(median_len))     # rolling buffer of recent quads used for median initialisation
        self.prev = None                              # last smoothed quad (float32 array of 4 corners)
        self.max_move_frac = float(max_move_frac)    # max allowed corner movement per frame as fraction of quad width

    def _to_arr(self, quad):
        return np.array(quad, dtype=np.float32)

    def update(self, detected):
        if detected is None:
            if self.prev is None and len(self.quad_buffer) == self.quad_buffer.maxlen:
                self.prev = np.median(np.stack(self.quad_buffer), axis=0)  # bootstrap from buffer when first going None
            return None if self.prev is None else [tuple(p) for p in self.prev.astype(int)]  # hold last known position

        detected_quad = self._to_arr(detected)
        self.quad_buffer.append(detected_quad)  # always keep the buffer up to date

        if self.prev is None:  # first detection — initialise from median of buffer to avoid jump
            if len(self.quad_buffer) >= 2:
                self.prev = np.median(np.stack(self.quad_buffer), axis=0)
            else:
                self.prev = detected_quad.copy()
            return [tuple(p) for p in self.prev.astype(int)]

        quad_width = np.linalg.norm(self.prev[0] - self.prev[1])  # approximate quad width for scale-relative threshold
        max_move_pixels = max(4.0, quad_width * self.max_move_frac)       # maximum tolerated corner movement in pixels
        biggest_corner_jump = np.max(np.linalg.norm(detected_quad - self.prev, axis=1))  # largest single-corner jump this frame
        if biggest_corner_jump > max_move_pixels:  # jitter / false detection — hold previous position
            return [tuple(p) for p in self.prev.astype(int)]

        self.prev = (1.0 - self.alpha) * self.prev + self.alpha * detected_quad  # exponential moving average blend
        return [tuple(p) for p in self.prev.astype(int)]


def run_screen_detector():
    camera_index = settings.get("camera_index")
    camera = cv2.VideoCapture(camera_index)
    previous_time = time.time()
    quad_filter = QuadFilter(alpha=0.18, median_len=5, max_move_frac=0.18)

    state = State.SEARCHING
    next_retry = 0.0
    alert_msg = 'Place laptop in view'

    while True:
        if settings.get("_reset_flag"):
            settings.set_val("_reset_flag", False)
            quad_filter = QuadFilter()
            state = State.SEARCHING
            alert_msg = 'Place laptop in view'

        new_camera_index = settings.get("camera_index")
        if new_camera_index != camera_index:
            camera.release()
            camera_index = new_camera_index
            camera = cv2.VideoCapture(camera_index)
            quad_filter = QuadFilter()
            state = State.SEARCHING
            alert_msg = 'Place laptop in view'

        if settings.get("paused"):
            is_window_open = show_windows(
                _last_frame if '_last_frame' in dir() else np.zeros((360, 480, 3), dtype=np.uint8),
                _last_lsd if '_last_lsd' in dir() else np.zeros((360, 480, 3), dtype=np.uint8),
                _last_warped if '_last_warped' in dir() else None,
                _last_filtered if '_last_filtered' in dir() else None,
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
        fps = 1.0 / max(current_time - previous_time, 1e-9)  # instantaneous frame rate; clamped to avoid division by zero
        previous_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)  # green FPS overlay top-left

        edge_map = compute_edges(frame)   # Sobel magnitude map fed into LSD
        lsd_vis  = lsd_lines_vis(edge_map)  # BGR panel showing detected line segments


        if state == State.TRACKING:
            quad = find_screen_quad(frame, edge_mask=edge_map)
            status = validate_quad(quad, frame.shape)

            if status == 'ok':
                smoothed = quad_filter.update(quad)
            else:
                state = State.SEARCHING
                next_retry = current_time
                alert_msg = ('Place laptop in view'
                             if status == 'no_laptop'
                             else 'Reposition laptop')
                quad_filter = QuadFilter()
                smoothed = None

        if state == State.SEARCHING:
            smoothed = None
            if current_time >= next_retry:
                quad = find_screen_quad(frame, edge_mask=edge_map)
                status = validate_quad(quad, frame.shape)
                if status == 'ok':
                    state = State.TRACKING
                    alert_msg = ''
                    quad_filter = QuadFilter()
                    smoothed = quad_filter.update(quad)
                else:
                    alert_msg = ('Place laptop in view'
                                 if status == 'no_laptop'
                                 else 'Reposition laptop')
                    next_retry = current_time + RETRY_INTERVAL

        if alert_msg:
            _draw_alert(frame, alert_msg)

        warped = None
        filtered = None
        if smoothed is not None:
            quad_points = np.array(smoothed, dtype=np.int32)
            cv2.polylines(frame, [quad_points], isClosed=True,
                          color=(0, 200, 80), thickness=2)  # draw quad outline in green on the camera view
            for point in smoothed:
                cv2.circle(frame, point, 5, (0, 255, 0), -1)  # draw each corner as a filled green dot
            warped = four_points_transform(
                frame, np.array(smoothed, dtype=np.float32))  # perspective-warp the quad to a flat rectangle
            if warped is not None:
                filtered = filter_image(warped)  # apply CLAHE + sharpening to the warped image

        if settings.get("show_intersections") and smoothed is not None:
            for point in smoothed:
                cv2.circle(lsd_vis, point, 5, (0, 0, 255), -1)  # draw computed corner points as red dots on LSD view

        _last_frame = frame     # store for use during pause
        _last_lsd   = lsd_vis   # store for use during pause
        _last_warped = warped   # store for use during pause
        _last_filtered = filtered  # store for use during pause

        det_status = f"State: {state.value}"  # shown in the status bar
        if alert_msg:
            det_status += f"  —  {alert_msg}"

        lsd_panel = lsd_vis if settings.get("show_lsd") else None  # hide LSD panel if disabled
        is_window_open = show_windows(frame, lsd_panel, warped, filtered, fps=fps, det_status=det_status)
        if not is_window_open:  # window was closed
            break

    camera.release()  # release the camera when the loop exits


if __name__ == '__main__':
    run_screen_detector()
