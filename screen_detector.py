import cv2
import time
import numpy as np
from detector_logic import compute_edges, find_screen_quad, validate_quad
from transform import four_points_transform
from quad import make_quad_filter
import settings

RETRY_INTERVAL = 1.0 # retry seconds after failed detection and lockin broken
LOCK_ON_DURATION = 2.0 # seconds to wait after first detetion before Lock IN

def run_screen_detector():
    camera = cv2.VideoCapture(settings.camera_index)
    previous_time = time.time() # start time for FPS calculation 
    quad_filter = make_quad_filter() # returns 4 updated smoothed Quad points

    state_searching = True
    next_retry = 0.0
    lock_start_time = None

    while True: # main loop
        got_frame, frame = camera.read()
        if not got_frame:
            break

        current_time = time.time() # current time for FPS and timing logic
        fps = 1.0 / max(current_time - previous_time, 1e-9) # calculate FPS 
        previous_time = current_time # update previous time for next loop

        edge_map = compute_edges(frame) # returns an array egde map for given frame, where each pixel val show strength of an edge values (0-255)

        if not state_searching: # run when Lock IN to check if screen detected and valid
            quad = find_screen_quad(frame, edge_mask=edge_map) # returns 4 points of detected screen quad or None if not found
            status = validate_quad(quad, frame.shape) # returns 'ok' if quad is valid, else 'invalid' or 'none'

            if status == 'ok': # if valid quad found, update smoothed quad points using filter and continue
                smoothed = quad_filter(quad) # returns smoothed quad points or None if new quad is too different from previous (indicating possible detection loss)
                if smoothed is None: # if filter returns None, it means the new quad is too different from previous, so reset to searching state
                    state_searching = True
                    next_retry = current_time
                    quad_filter = make_quad_filter() 
            else: # if no valid quad found, reset to searching state to try again
                state_searching = True
                next_retry = current_time
                quad_filter = make_quad_filter()
                smoothed = None
                lock_start_time = None

        if state_searching: # run when searching for a screen
            smoothed = None
            if current_time >= next_retry: # check if it's time to retry detection after a failed attempt
                quad = find_screen_quad(frame, edge_mask=edge_map) # returns 4 points of detected screen quad or None if not found
                status = validate_quad(quad, frame.shape) # returns 'ok' if quad is valid, else 'invalid' or 'none'
                if status == 'ok': # if valid quad found, switch to lockin state and start timer
                    state_searching = False
                    quad_filter = make_quad_filter()
                    smoothed = quad_filter(quad)
                    lock_start_time = current_time
                else: # if no valid quad found, set next retry time to current time + retry interval
                    next_retry = current_time + RETRY_INTERVAL

        warped = None
        if smoothed is not None: # if we have smoothed quad points then apply transform to get warped screen image
            warped = four_points_transform(frame, np.array(smoothed, dtype=np.float32))

        lock_remaining = max(0.0, LOCK_ON_DURATION - (current_time - lock_start_time)) if lock_start_time is not None else 0.0
        locked = lock_start_time is not None and lock_remaining == 0

        win_w, win_h = (720, 540)
        if warped is not None and locked: # if we have a warped image and lockin is complete, analyze the warped image to determine if screen is on or off
            display = cv2.resize(warped, (win_w, win_h))
            edges_w = compute_edges(display)
            black_ratio = np.sum(edges_w < 20) / edges_w.size
            is_on = black_ratio <= 0.85 # percentage of black pixels in edge map (main determining factor for screen state)
            state_label = "Screen ON" if is_on else "Screen OFF"
            state_color = (0, 255, 0) if is_on else (0, 0, 255)
            (tw, th), _ = cv2.getTextSize(state_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(display, state_label, ((win_w - tw) // 2, win_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_color, 2, cv2.LINE_AA)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Warped", display)
        else: # if we don't have a warped image or lockin is not complete, show placeholder with lockin status
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
