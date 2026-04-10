import cv2
import numpy as np
from detector_logic import compute_edges, lsd_lines_vis

def show_state_window(warped):
    edge_map = compute_edges(warped)
    black_ratio = np.sum(edge_map < 20) / edge_map.size  
    state = 'OFF' if black_ratio > 0.85 else 'ON'
    color = (0, 255, 0) if state == 'ON' else (0, 0, 255)
    lsd_vis = lsd_lines_vis(edge_map)
    msg = f'Screen {state}'
    textsize = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
    x = (lsd_vis.shape[1] - textsize[0]) // 2
    y = (lsd_vis.shape[0] + textsize[1]) // 2
    cv2.putText(lsd_vis, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
    cv2.imshow("State", cv2.resize(lsd_vis, (800, 600), interpolation=cv2.INTER_LINEAR))
    cv2.waitKey(1)
