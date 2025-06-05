import cv2
import numpy as np

class KeyPressDetector:
    def __init__(self, movement_threshold=5, pressing_threshold=10):
        """Initialize the key press detector."""
        self.prev_fingertip_positions = []
        self.movement_threshold = movement_threshold  # pixels
        self.pressing_threshold = pressing_threshold  # pixels
        self.is_pressing = False

    def detect_press(self, frame, hand_landmarks, keyboard_bbox):
        """Detect if a finger is in the pressing state."""
        if not keyboard_bbox or not hand_landmarks:
            return False

        # Get keyboard coordinates
        kx1, ky1, kx2, ky2 = keyboard_bbox

        pressing_detected = False

        # Process each hand
        for landmarks in hand_landmarks:
            # Extract fingertip positions (index, middle, ring, pinky)
            fingertips = [
                (landmarks.landmark[8].x, landmarks.landmark[8].y),   # Index fingertip
                (landmarks.landmark[12].x, landmarks.landmark[12].y), # Middle fingertip
                (landmarks.landmark[16].x, landmarks.landmark[16].y), # Ring fingertip
                (landmarks.landmark[20].x, landmarks.landmark[20].y), # Pinky fingertip
            ]

            # Convert normalized coordinates to pixel coordinates
            h, w, _ = frame.shape
            fingertips_px = [(int(x * w), int(y * h)) for x, y in fingertips]

            # Check if we have previous positions to compare
            if not self.prev_fingertip_positions:
                self.prev_fingertip_positions = fingertips_px
                return False

            # Check if any fingertip is above the keyboard and moving downward
            for i, (curr_x, curr_y) in enumerate(fingertips_px):
                prev_x, prev_y = self.prev_fingertip_positions[i]

                # Check if fingertip is above the keyboard
                if kx1 <= curr_x <= kx2 and ky1 <= curr_y <= ky2:
                    # Calculate vertical movement
                    vertical_movement = curr_y - prev_y

                    # If movement is downward and exceeds the threshold
                    if vertical_movement > self.movement_threshold:
                        # Additional check: is the finger tip close to the keyboard surface?
                        # We can use the lower finger joints to determine if this is a pressing action
                        lower_joint = (landmarks.landmark[i*4+6].x, landmarks.landmark[i*4+6].y)  # Lower joint of the finger
                        lower_joint_px = (int(lower_joint[0] * w), int(lower_joint[1] * h))

                        # If the fingertip is significantly lower than the joint, it's likely pressing
                        if curr_y - lower_joint_px[1] > self.pressing_threshold:
                            pressing_detected = True
                            break

            # Update previous positions
            self.prev_fingertip_positions = fingertips_px

            # If pressing detected, no need to check other hands
            if pressing_detected:
                break

        # Update and return pressing state
        self.is_pressing = pressing_detected
        return self.is_pressing

    def draw_press_status(self, frame, is_pressing):
        """Draw press status on the frame."""
        if is_pressing:
            cv2.putText(frame, "KEY PRESS DETECTED!", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame
