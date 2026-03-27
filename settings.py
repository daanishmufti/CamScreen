settings = {
    "camera_index": 0,
    "show_lsd": True,
    "show_intersections": False,
    "lsd_v_min_deg": 60,
    "lsd_v_candidates": 12,
    "lsd_v_match_tol": 0.30,
    "filter_max_move": 0.18,
}

def get(key): return settings.get(key)
def set_val(key, val): settings[key] = val
