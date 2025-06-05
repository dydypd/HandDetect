import cv2
import numpy as np
import time

class KeyPressDetector:
    def __init__(self):
        """Initialize the key press detector with simplified approach."""
        # Track previous finger positions for both hands
        self.prev_finger_positions = [None, None]  # Left and right hand

        # Parameters for detection - make more sensitive
        self.vertical_threshold = 10  # Minimum vertical movement to consider (pixels)
        self.horizontal_threshold = 15  # Maximum horizontal movement allowed (pixels)

        # State tracking for both hands
        self.is_pressing = False
        self.finger_pressing = [None, None]  # Which finger is pressing on each hand
        self.press_detected_time = None
        self.press_flash_duration = 0.2  # Show press indicator for only 0.2 seconds

        # Add position history to detect slower movements (for both hands)
        self.position_history = [[], []]  # Left and right hand
        self.history_length = 3

        # Debug info for both hands
        self.debug_info = [{}, {}]  # Left and right hand

    def detect_press(self, frame, hand_landmarks, keyboard_bbox):
        """Detect key press with simplified, robust algorithm for both hands."""
        # Reset debug info each frame
        self.debug_info = [{}, {}]

        # Reset pressing state if flash duration has passed
        if self.press_detected_time and time.time() - self.press_detected_time > self.press_flash_duration:
            self.is_pressing = False
            self.finger_pressing = [None, None]

        if not keyboard_bbox or not hand_landmarks or len(hand_landmarks) == 0:
            self.prev_finger_positions = [None, None]
            self.position_history = [[], []]
            self.is_pressing = False
            return False

        # Extract keyboard coordinates
        kx1, ky1, kx2, ky2 = keyboard_bbox
        keyboard_height = ky2 - ky1

        # Process each hand (up to 2 hands)
        for hand_idx in range(min(len(hand_landmarks), 2)):
            landmarks = hand_landmarks[hand_idx]

            # Extract fingertip positions (index, middle, ring, pinky)
            h, w, _ = frame.shape
            fingertips_idx = [8, 12, 16, 20]  # MediaPipe indices for fingertips
            fingertip_positions = []

            # Get current positions and convert to pixel coordinates
            for idx in fingertips_idx:
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                fingertip_positions.append((x, y))

            # Add to position history for this hand
            if not self.position_history[hand_idx]:
                self.position_history[hand_idx] = [fingertip_positions] * self.history_length
            else:
                self.position_history[hand_idx].append(fingertip_positions)
                if len(self.position_history[hand_idx]) > self.history_length:
                    self.position_history[hand_idx].pop(0)

            # If we don't have previous positions, store current and continue to next hand
            if self.prev_finger_positions[hand_idx] is None:
                self.prev_finger_positions[hand_idx] = fingertip_positions
                continue

            # Check each finger for pressing motion
            for i, curr_pos in enumerate(fingertip_positions):
                curr_x, curr_y = curr_pos

                # Skip if finger is not over keyboard
                if not (kx1-10 <= curr_x <= kx2+10 and ky1-10 <= curr_y <= ky2+10):  # Add small margin
                    continue

                # Get previous position (1 frame ago)
                prev_pos = self.prev_finger_positions[hand_idx][i]
                prev_x, prev_y = prev_pos

                # Calculate vertical and horizontal movement
                v_movement = curr_y - prev_y
                h_movement = abs(curr_x - prev_x)

                # Calculate cumulative movement if we have enough history
                cumulative_v_movement = 0
                if len(self.position_history[hand_idx]) >= 3:
                    oldest_pos = self.position_history[hand_idx][0][i]
                    cumulative_v_movement = curr_y - oldest_pos[1]

                # Debug info
                self.debug_info[hand_idx][f"finger_{i}"] = {
                    "position": curr_pos,
                    "v_movement": v_movement,
                    "cumulative": cumulative_v_movement,
                    "h_movement": h_movement,
                    "in_keyboard": "YES" if (kx1 <= curr_x <= kx2 and ky1 <= curr_y <= ky2) else "NO"
                }

                # Check for immediate downward movement OR cumulative movement
                immediate_press = (v_movement > self.vertical_threshold and
                                 h_movement < self.horizontal_threshold)

                cumulative_press = (cumulative_v_movement > self.vertical_threshold * 2 and
                                   abs(curr_x - self.position_history[hand_idx][0][i][0]) < self.horizontal_threshold * 2)

                if ((immediate_press or cumulative_press) and
                    curr_y > ky1 + keyboard_height * 0.2):  # At least 20% into keyboard

                    # Get finger position relative to keyboard depth
                    depth_ratio = (curr_y - ky1) / keyboard_height

                    # Only count as press if finger is deep enough into keyboard
                    if depth_ratio > 0.3:  # At least 30% into keyboard depth
                        self.is_pressing = True
                        self.finger_pressing[hand_idx] = i
                        self.press_detected_time = time.time()

            # Update previous positions for this hand
            self.prev_finger_positions[hand_idx] = fingertip_positions

        return self.is_pressing

    def draw_press_status(self, frame, is_pressing):
        """Draw press status on the frame with momentary flash."""
        # Define hand names and finger names
        hand_names = ["LEFT", "RIGHT"]
        finger_names = ["INDEX", "MIDDLE", "RING", "PINKY"]

        # Always draw debug info for both hands
        for hand_idx in range(2):
            for i in range(4):  # 4 fingers
                finger_key = f"finger_{i}"
                if hand_idx < len(self.debug_info) and finger_key in self.debug_info[hand_idx]:
                    info = self.debug_info[hand_idx][finger_key]
                    x, y = info["position"]

                    # Draw a circle at fingertip position - different color for each hand
                    color = (0, 255, 255) if hand_idx == 0 else (255, 255, 0)  # Yellow vs Cyan
                    cv2.circle(frame, (x, y), 5, color, -1)

                    # Draw movement data near fingertip
                    text = f"V:{info['v_movement']:.1f}"
                    if 'cumulative' in info:
                        text += f" C:{info['cumulative']:.1f}"
                    cv2.putText(frame, text, (x+10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw press notification if detected - only for a moment
        if is_pressing and (self.finger_pressing[0] is not None or self.finger_pressing[1] is not None):
            # Calculate intensity based on time since detection (fade out effect)
            time_since_press = time.time() - self.press_detected_time
            if time_since_press < self.press_flash_duration:
                # Draw press notification for each hand that's pressing
                for hand_idx in range(2):
                    if self.finger_pressing[hand_idx] is not None:
                        finger_idx = self.finger_pressing[hand_idx]
                        finger_text = finger_names[finger_idx]
                        hand_text = hand_names[hand_idx]

                        # Draw a notification that fades out
                        alpha = 1.0 - (time_since_press / self.press_flash_duration)
                        color_intensity = int(255 * alpha)
                        color = (0, 0, color_intensity)  # Fading red

                        # Position notification at top of screen for first hand, middle for second
                        y_pos = 50 if hand_idx == 0 else 90
                        cv2.putText(frame, f"KEY PRESS: {hand_text} {finger_text}", (50, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                        # Draw red circle on the pressing finger
                        finger_key = f"finger_{finger_idx}"
                        if finger_key in self.debug_info[hand_idx]:
                            x, y = self.debug_info[hand_idx][finger_key]["position"]
                            cv2.circle(frame, (x, y), 10, color, -1)

        return frame

