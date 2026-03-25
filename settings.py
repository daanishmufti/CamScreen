import json
import os

_FILE = os.path.join(os.path.dirname(__file__), "camscreen_settings.json")  

_DEFAULTS = {
    "camera_index": 0,          # index passed to cv2.VideoCapture
    "filter_enabled": True,     # whether to apply CLAHE + sharpening to the warped image
    "clahe_clip": 2.0,          # CLAHE contrast clip limit
    "clahe_tile": 8,            # CLAHE tile grid size (tile x tile)
    "sharpen_strength": 5,      # centre value of the sharpening kernel
    "show_lsd": True,           # show the LSD edge panel in the UI
    "show_intersections": False, # overlay computed quad corners on the LSD view
    "lsd_v_min_deg": 60,        # minimum angle from horizontal to classify a segment as vertical
    "lsd_v_candidates": 8,      # top-N longest V-segs considered from each half when matching
    "lsd_v_match_tol": 0.30,    # max fractional length difference allowed between the chosen pair
    "paused": False,            # whether the capture loop is paused
}

_settings = dict(_DEFAULTS)
_saved_settings = None


def load():
    global _settings
    settings_file = open(_FILE, "r")
    saved_values = json.load(settings_file)  
    for setting_name in _DEFAULTS:
        if setting_name in saved_values:
            _settings[setting_name] = type(_DEFAULTS[setting_name])(saved_values[setting_name])  
    return _settings


def save():
    settings_file = open(_FILE, "w")
    json.dump(_settings, settings_file, indent=2)
    global _saved_settings
    _saved_settings = dict(_settings)


def get(key):
    return _settings.get(key, _DEFAULTS.get(key)) 


def set_val(key, value):
    _settings[key] = value  


def reset():
    global _settings
    _settings = dict(_DEFAULTS)  
    save()  

def restore_saved():
    global _settings, _saved_settings
    if _saved_settings is None:
        load()
        _saved_settings = dict(_settings)
    _settings = dict(_saved_settings)
    return _settings


def get_all():
    return dict(_settings)


load()
