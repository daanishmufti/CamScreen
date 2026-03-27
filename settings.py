_DEFAULTS = {
    "camera_index": 0,
    "show_lsd": True,
    "show_intersections": False,
    "lsd_v_min_deg": 60,
    "lsd_v_candidates": 8,
    "lsd_v_match_tol": 0.30,
    "paused": False,
}

_settings = dict(_DEFAULTS)


def get(key):
    return _settings.get(key, _DEFAULTS.get(key))


def set_val(key, value):
    _settings[key] = value


def reset():
    global _settings
    _settings = dict(_DEFAULTS)

