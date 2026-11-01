
from collections import Counter, deque

class GestureCommandController:
    """Convert MediaPipe gestures into safe robot commands."""

    COMMANDS = {
        "Open_Palm": "STOP",
        "Closed_Fist": "MOVE BACKWARD",
        "Pointing_Up": "MOVE FORWARD",
        "Thumb_Down": "TURN LEFT",
        "Thumb_Up": "TURN RIGHT"
    }

    def __init__(self, threshold=0.50, window=5, votes_required=3):
        self.threshold = threshold
        self.history = deque(maxlen=window)
        self.votes_required = votes_required

    def update(self, gesture, confidence):
        # Fail-safe: uncertain or missing hand means STOP.
        if gesture not in self.COMMANDS or confidence < self.threshold:
            self.history.clear()
            return "STOP", "Safety stop"

        # Open palm always stops immediately.
        if gesture == "Open_Palm":
            self.history.clear()
            return "STOP", "Stop gesture"

        self.history.append(self.COMMANDS[gesture])

        command, votes = Counter(self.history).most_common(1)[0]

        if votes >= self.votes_required:
            return command, "Stable command"

        return "STOP", "Stabilizing"

    def emergency_stop(self):
        self.history.clear()
        return "STOP"
