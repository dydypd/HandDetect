import cv2
import numpy as np
import math

class KeyboardSegmentAnalyzer:
    def __init__(self):
        """Initialize the keyboard segment analyzer."""
        self.keyboard_orientation = 0  # Degrees - 0 means standard orientation
        self.keyboard_contour = None
        self.keyboard_rect = None
        self.keyboard_center = None
        self.standard_aspect_ratio = 3.5  # Standard keyboard aspect ratio (width/height)
        self.is_correct_orientation = True  # Flag to indicate if keyboard is in the correct orientation
        self.hand_direction = None  # Store the direction of the hand relative to keyboard
        self.finger_direction = None  # Store the direction from finger base to fingertip
        self.analysis_method = "finger"  # Default analysis method: "angle", "hand", or "finger"

    def set_analysis_method(self, method):
        """Set the analysis method to use."""
        if method in ["angle", "hand", "finger"]:
            self.analysis_method = method
            return True
        return False

    def analyze_segment(self, frame, keyboard_bbox, keyboard_mask, hand_landmarks=None):
        """Analyze the keyboard segment and determine its orientation."""
        if keyboard_bbox is None or keyboard_mask is None:
            return None

        # Extract keyboard coordinates
        kx1, ky1, kx2, ky2 = keyboard_bbox

        # Create a binary mask for the keyboard
        binary_mask = self._prepare_mask(frame, keyboard_mask)
        if binary_mask is None:
            return None

        # Find contours in the mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (should be the keyboard)
        if not contours:
            return None

        self.keyboard_contour = max(contours, key=cv2.contourArea)

        # Find the minimum area rectangle that bounds the keyboard
        self.keyboard_rect = cv2.minAreaRect(self.keyboard_contour)
        self.keyboard_center, (width, height), angle = self.keyboard_rect

        # Determine if width and height need to be swapped
        if width < height:
            width, height = height, width
            angle += 90

        # Adjust angle to be between -90 and 90 degrees
        if angle > 90:
            angle -= 180

        # Store the orientation angle
        self.keyboard_orientation = angle

        # Choose analysis method based on selection
        if self.analysis_method == "angle":
            # Just use the angle of the keyboard for orientation detection
            self.is_correct_orientation = abs(angle) < 30  # Assuming correct if within 30 degrees
        elif self.analysis_method == "hand" and hand_landmarks:
            # Use hand position relative to keyboard
            self._analyze_orientation_with_hand(hand_landmarks)
        elif self.analysis_method == "finger" and hand_landmarks:
            # Use finger direction for orientation analysis
            self._analyze_orientation_with_finger(hand_landmarks)
        else:
            # Fallback to simple angle analysis
            self.is_correct_orientation = abs(angle) < 30

        # Extract the keyboard region
        keyboard_region = self._extract_keyboard_region(frame, binary_mask)

        # Create a normalized view
        normalized_view = self._create_normalized_view(frame, keyboard_region)

        return normalized_view

    def _analyze_orientation_with_hand(self, hand_landmarks):
        """Analyze if keyboard orientation is correct based on hand position."""
        if not hand_landmarks or self.keyboard_center is None:
            return

        # Get wrist position (base of hand)
        if len(hand_landmarks) > 0:
            # Extract hand information - MediaPipe landmarks are accessed by index
            first_hand = hand_landmarks[0]  # Get first detected hand
            # Wrist is typically landmark 0 in MediaPipe Hands
            # Access landmark directly using the landmark property
            wrist = first_hand.landmark[0]

            if wrist:
                # Convert normalized coordinates to pixel coordinates
                image_height, image_width = 720, 1280  # Default values, adjust if needed
                wrist_x = int(wrist.x * image_width)
                wrist_y = int(wrist.y * image_height)

                # Calculate hand direction relative to keyboard center
                keyboard_x, keyboard_y = self.keyboard_center
                dx = wrist_x - keyboard_x
                dy = wrist_y - keyboard_y

                # Calculate angle of hand relative to keyboard (in degrees)
                hand_angle = math.degrees(math.atan2(-dy, dx))  # Negative dy because y increases downward
                self.hand_direction = hand_angle

                # Determine if keyboard is correctly oriented based on hand position
                # For a correctly oriented keyboard:
                # - Hand should be below the keyboard (negative y direction)
                # - Keyboard angle should be aligned with the scene

                # Check if hand is positioned reasonably (should be below/above keyboard)
                is_hand_position_valid = dy != 0  # Avoid division by zero

                if is_hand_position_valid:
                    # Determine expected keyboard orientation based on hand position
                    expected_orientation = 0  # Default to standard orientation

                    # If hand is above the keyboard
                    if dy < 0:
                        # Hand is above keyboard, keyboard should be rotated 180 degrees
                        expected_orientation = 180

                    # Check if current orientation approximately matches expected
                    orientation_diff = abs((self.keyboard_orientation - expected_orientation + 180) % 360 - 180)
                    self.is_correct_orientation = orientation_diff < 30  # Within 30 degrees tolerance
                else:
                    # Can't determine orientation from hand position
                    self.is_correct_orientation = abs(self.keyboard_orientation) < 30

    def _analyze_orientation_with_finger(self, hand_landmarks):
        """Analyze keyboard orientation based on finger direction."""
        if not hand_landmarks or self.keyboard_center is None or self.keyboard_rect is None:
            return

        if len(hand_landmarks) > 0:
            first_hand = hand_landmarks[0]

            # Use index finger for direction (landmarks 5, 6, 7, 8)
            # 5: base of index finger, 8: tip of index finger
            if len(first_hand.landmark) >= 9:  # Make sure we have at least index finger landmarks
                finger_base = first_hand.landmark[5]
                finger_tip = first_hand.landmark[8]

                # Convert normalized coordinates to pixel coordinates
                image_height, image_width = 720, 1280  # Default values
                base_x = int(finger_base.x * image_width)
                base_y = int(finger_base.y * image_height)
                tip_x = int(finger_tip.x * image_width)
                tip_y = int(finger_tip.y * image_height)

                # Calculate finger direction
                dx = tip_x - base_x
                dy = tip_y - base_y

                # Skip if finger is not extended enough (increased sensitivity by lowering threshold)
                if abs(dx) < 5 and abs(dy) < 5:  # Lowered from 10 to 5 for higher sensitivity
                    self.is_correct_orientation = abs(self.keyboard_orientation) < 30
                    return

                # Calculate angle of finger direction (in degrees)
                finger_angle = math.degrees(math.atan2(-dy, dx))  # Negative dy because y increases downward
                self.finger_direction = finger_angle

                # Get keyboard rectangle information
                center, (width, height), angle = self.keyboard_rect

                # Ensure width is greater than height (landscape orientation)
                if width < height:
                    width, height = height, width
                    angle += 90

                # Normalize angle to be between -90 and 90 degrees
                if angle > 90:
                    angle -= 180

                # Calculate the angle of the keyboard's longer edge
                keyboard_long_edge_angle = angle

                # Calculate the angle of the keyboard's shorter edge (perpendicular to longer edge)
                keyboard_short_edge_angle = (keyboard_long_edge_angle + 90) % 180 - 90

                # Check if finger direction is perpendicular to the keyboard's longer edge
                # This means the finger should be parallel to the keyboard's shorter edge

                # Calculate difference between finger angle and keyboard's short edge angle
                short_edge_diff = abs((finger_angle - keyboard_short_edge_angle + 180) % 360 - 180)

                # Calculate difference between finger angle and keyboard's long edge angle
                long_edge_diff = abs((finger_angle - keyboard_long_edge_angle + 180) % 360 - 180)

                # The finger should be more aligned with the short edge (perpendicular to long edge)
                # for correct keyboard orientation - increased sensitivity by widening angle tolerance
                self.is_correct_orientation = short_edge_diff < 35  # Increased from 30 to 35 degrees for higher sensitivity

                # Debug information
                print(f"Keyboard long edge angle: {keyboard_long_edge_angle:.1f}°")
                print(f"Keyboard short edge angle: {keyboard_short_edge_angle:.1f}°")
                print(f"Finger angle: {finger_angle:.1f}°")
                print(f"Difference from short edge: {short_edge_diff:.1f}°")
                print(f"Difference from long edge: {long_edge_diff:.1f}°")
                print(f"Is correct orientation: {self.is_correct_orientation}")
            else:
                # Fallback to simple angle check if landmarks are not available
                self.is_correct_orientation = abs(self.keyboard_orientation) < 30

    def _prepare_mask(self, frame, mask):
        """Prepare and clean the keyboard mask."""
        try:
            # Convert mask to binary (0 or 1)
            if isinstance(mask, np.ndarray):
                if len(mask.shape) == 3 and mask.shape[0] == 1:
                    # Handle case where mask is [1, H, W]
                    binary_mask = (mask[0] > 0.5).astype(np.uint8) * 255
                else:
                    # Handle regular case
                    binary_mask = (mask > 0.5).astype(np.uint8) * 255

                # Resize mask if needed
                if binary_mask.shape[:2] != frame.shape[:2]:
                    binary_mask = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]))

                # Clean up the mask with morphological operations
                kernel = np.ones((5, 5), np.uint8)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

                return binary_mask
            else:
                print("Mask is not a numpy array")
                return None
        except Exception as e:
            print(f"Error preparing mask: {e}")
            return None

    def _extract_keyboard_region(self, frame, mask):
        """Extract the keyboard region using the mask."""
        # Create a black background image
        keyboard_region = np.zeros_like(frame)

        # Apply the mask to get only the keyboard pixels
        keyboard_region = cv2.bitwise_and(frame, frame, mask=mask)

        return keyboard_region

    def _create_normalized_view(self, frame, keyboard_region):
        """Create a normalized view of the keyboard (oriented correctly)."""
        if self.keyboard_rect is None:
            return keyboard_region

        # Get the box points
        box = cv2.boxPoints(self.keyboard_rect)
        box = np.int0(box)

        # Get width and height of the keyboard
        center, (width, height), angle = self.keyboard_rect

        # Ensure width is greater than height (landscape orientation)
        if width < height:
            width, height = height, width

        # Create a rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

        # Determine the new size after rotation
        rotated_width = int(width * 1.2)  # Add some padding
        rotated_height = int(height * 1.2)  # Add some padding

        # Rotate the image
        rotated_region = cv2.warpAffine(keyboard_region, rotation_matrix, (frame.shape[1], frame.shape[0]))

        # Crop the rotated image to get just the keyboard
        x_center, y_center = int(center[0]), int(center[1])
        x_start = max(0, x_center - rotated_width // 2)
        y_start = max(0, y_center - rotated_height // 2)
        x_end = min(frame.shape[1], x_center + rotated_width // 2)
        y_end = min(frame.shape[0], y_center + rotated_height // 2)

        normalized_view = rotated_region[y_start:y_end, x_start:x_end]

        # Check if we need to flip the image horizontally (if keyboard is upside down)
        # This is a simplification - in reality, we would need more analysis to determine this
        if abs(angle) > 45:
            normalized_view = cv2.flip(normalized_view, 1)

        return normalized_view

    def draw_analysis(self, frame):
        """Draw the keyboard analysis information on the frame."""
        if self.keyboard_contour is None or self.keyboard_rect is None:
            return frame

        result_frame = frame.copy()

        # Draw the contour
        cv2.drawContours(result_frame, [self.keyboard_contour], 0, (0, 255, 0), 2)

        # Draw the minimum area rectangle
        box = cv2.boxPoints(self.keyboard_rect)
        box = np.int0(box)
        cv2.drawContours(result_frame, [box], 0, (0, 0, 255), 2)

        # Draw the orientation angle
        center = (int(self.keyboard_center[0]), int(self.keyboard_center[1]))
        cv2.putText(result_frame, f"Angle: {self.keyboard_orientation:.1f} deg",
                   (center[0] - 50, center[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw arrow indicating keyboard orientation
        arrow_length = 50
        end_point = (
            int(center[0] + arrow_length * math.cos(math.radians(self.keyboard_orientation))),
            int(center[1] - arrow_length * math.sin(math.radians(self.keyboard_orientation)))
        )
        cv2.arrowedLine(result_frame, center, end_point, (255, 0, 0), 2)

        # Draw the orientation correctness
        correctness_text = "Correct" if self.is_correct_orientation else "Incorrect"
        cv2.putText(result_frame, f"Orientation: {correctness_text}",
                   (center[0] - 50, center[1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.is_correct_orientation else (0, 0, 255), 2)

        # Draw the analysis method being used
        method_text = f"Method: {self.analysis_method.capitalize()}"
        cv2.putText(result_frame, method_text,
                   (center[0] - 50, center[1] + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Draw finger direction if available
        if self.finger_direction is not None and self.analysis_method == "finger":
            # Draw text showing finger direction
            cv2.putText(result_frame, f"Finger: {self.finger_direction:.1f} deg",
                       (center[0] - 50, center[1] + 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

            # Draw arrow showing finger direction
            finger_end_point = (
                int(center[0] + arrow_length * math.cos(math.radians(self.finger_direction))),
                int(center[1] - arrow_length * math.sin(math.radians(self.finger_direction)))
            )
            cv2.arrowedLine(result_frame, center, finger_end_point, (255, 165, 0), 2)

        # Draw hand direction if available
        elif self.hand_direction is not None and self.analysis_method == "hand":
            # Draw text showing hand direction
            cv2.putText(result_frame, f"Hand: {self.hand_direction:.1f} deg",
                       (center[0] - 50, center[1] + 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # Draw arrow showing hand direction
            hand_end_point = (
                int(center[0] + arrow_length * math.cos(math.radians(self.hand_direction))),
                int(center[1] - arrow_length * math.sin(math.radians(self.hand_direction)))
            )
            cv2.arrowedLine(result_frame, center, hand_end_point, (0, 165, 255), 2)

        return result_frame
