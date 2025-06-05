import json
from datetime import datetime
import os

class KeyPressLogger:
    def __init__(self, log_file="key_press_logs.json"):
        self.log_file = log_file
        self.logs = []
        # Load existing logs if file exists
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.logs = json.load(f)
            except json.JSONDecodeError:
                self.logs = []

    def log_key_press(self, key_position, segment_corners):
        """
        Log a key press event with its position and segment corners
        :param key_position: tuple (x, y) of the key press position
        :param segment_corners: list of 4 tuples [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] representing corners
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "key_position": {
                "x": key_position[0],
                "y": key_position[1]
            },
            "segment_corners": [
                {"x": corner[0], "y": corner[1]} for corner in segment_corners
            ]
        }
        self.logs.append(log_entry)
        self._save_logs()

    def _save_logs(self):
        """Save logs to the JSON file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=4)

    def get_logs(self):
        """Return all logged data"""
        return self.logs

    def clear_logs(self):
        """Clear all logs"""
        self.logs = []
        self._save_logs()
