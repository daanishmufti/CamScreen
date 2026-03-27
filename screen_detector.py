import cv2, time, numpy as np
from detector_logic import compute_edges, lsd_lines_vis, find_screen_quad, validate_quad
from transform import four_points_transform
def show_windows(frame, lsd=None, warped=None):
    cv2.imshow("Camera", frame)
    if lsd is not None:
        cv2.imshow("LSD", lsd)
    if warped is not None:
        cv2.imshow("Warped", cv2.resize(warped, (800, 600), interpolation=cv2.INTER_LINEAR))
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        cv2.destroyAllWindows()
        return False
    return True
from quad import new_filter, update_filter
import settings

def run_screen_detector():
    cam_idx = settings.get("camera_index")
    cam = cv2.VideoCapture(cam_idx)
    prev_t = time.time()
    qf = new_filter()
    lock_until = 0.0  
    locked_in = False  

    while True:
        if settings.get("_reset_flag"):
            settings.set_val("_reset_flag", False)
            qf = new_filter()
            lock_until = 0.0
            locked_in = False

        new_idx = settings.get("camera_index")
        if new_idx != cam_idx:
            cam.release()
            cam_idx = new_idx
            cam = cv2.VideoCapture(cam_idx)
            qf = new_filter()

        ok, frame = cam.read()
        if not ok:
            break

        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-9)
        prev_t = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        edges = compute_edges(frame)
        lsd_vis = lsd_lines_vis(edges)

        quad = find_screen_quad(frame, edge_mask=edges)
        valid = validate_quad(quad, frame.shape) == 'ok'

        if valid:
            prev_pts = qf.get("prev")
            exceeded = False
            if prev_pts is not None:
                det = np.array(quad, dtype=np.float32)
                w = np.linalg.norm(prev_pts[0] - prev_pts[1])
                max_move = settings.get("filter_max_move") or 0.18
                exceeded = np.max(np.linalg.norm(det - prev_pts, axis=1)) > max(4.0, w * max_move)

            if exceeded:
                lock_until = 0.0
                locked_in = False
                qf = new_filter()
                smoothed = None
                alert = 'No laptop detected'
            else:
                smoothed = update_filter(qf, quad)
                if not locked_in:
                    if lock_until == 0.0:
                        lock_until = now + 2.0
                    if now >= lock_until:
                        locked_in = True
                        alert = 'Screen detected'
                    else:
                        alert = f'Locking in ({int(lock_until - now) + 1}s)'
                else:
                    alert = 'Screen detected'
        else:
            smoothed = None
            lock_until = 0.0
            locked_in = False
            alert = 'No laptop detected'

        if alert:
            textsize = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            x = (frame.shape[1] - textsize[0]) // 2
            y = 40
            cv2.putText(frame, alert, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if smoothed:
            pts = np.array(smoothed, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 200, 80), 2)
            for p in smoothed:
                cv2.circle(frame, p, 5, (0, 255, 0), -1)
            warped = four_points_transform(frame, np.array(smoothed, dtype=np.float32))
            from screen_state import show_state_window
            show_state_window(warped)
            if not locked_in and lock_until > now:
                remaining = int(lock_until - now) + 1
                timer_msg = f'Locking in: {remaining}s'
                textsize = cv2.getTextSize(timer_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                x = (warped.shape[1] - textsize[0]) // 2
                y = (warped.shape[0] + textsize[1]) // 2
                cv2.putText(warped, timer_msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            warped = np.zeros((600, 800, 3), dtype=np.uint8)
            msg = 'No laptop detected'
            textsize = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            x = (warped.shape[1] - textsize[0]) // 2
            y = (warped.shape[0] + textsize[1]) // 2
            cv2.putText(warped, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if settings.get("show_intersections") and smoothed:
            for p in smoothed:
                cv2.circle(lsd_vis, p, 5, (0, 0, 255), -1)

        lsd_panel = lsd_vis if settings.get("show_lsd") else None
        if not show_windows(frame, lsd_panel, warped):
            break

    cam.release()
