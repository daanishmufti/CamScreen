from enum import Enum

class State(Enum):
    SEARCHING = "Searching"    # no quad found yet; scanning every frame
    TRACKING = "Tracking"      # quad found and being smoothed frame-to-frame