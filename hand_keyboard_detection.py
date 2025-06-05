import cv2
from hand_detector import HandDetector
from keyboard_detector import KeyboardDetector
from key_press_detector import KeyPressDetector
from keyboard_segment_analyzer import KeyboardSegmentAnalyzer

class HandKeyboardSystem:
    def __init__(self, yolo_model_path="yolo11n-seg.pt"):
        """Initialize the hand and keyboard detection system."""
        # Initialize component detectors
        self.hand_detector = HandDetector()
        self.keyboard_detector = KeyboardDetector(model_path=yolo_model_path)
        self.key_press_detector = KeyPressDetector()
        self.keyboard_analyzer = KeyboardSegmentAnalyzer()

        # Available analysis methods
        self.analysis_methods = ["angle", "hand", "finger"]
        self.current_method_index = 0  # Start with angle method

    def process_frame(self, frame):
        """Process a frame for hand and keyboard detection."""
        # Make a copy of the frame for drawing
        display_frame = frame.copy()

        # Detect hands first so we can use hand data for keyboard orientation
        hand_landmarks = self.hand_detector.detect_hands(frame)

        # Detect keyboard
        keyboard_bbox, keyboard_mask = self.keyboard_detector.detect_keyboard(frame)

        # Analyze keyboard orientation if detected
        normalized_keyboard = None
        if keyboard_bbox and keyboard_mask is not None:
            # Pass hand landmarks to analyze_segment to determine correct orientation
            normalized_keyboard = self.keyboard_analyzer.analyze_segment(frame, keyboard_bbox, keyboard_mask, hand_landmarks)

            # Draw the keyboard analysis information
            display_frame = self.keyboard_analyzer.draw_analysis(display_frame)

            # Display keyboard orientation information
            angle = self.keyboard_analyzer.keyboard_orientation
            cv2.putText(display_frame, f"Keyboard angle: {angle:.1f}°",
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Add interpretation of the angle and orientation correctness
            orientation_text = self.get_orientation_description(angle)
            cv2.putText(display_frame, orientation_text,
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Add correctness based on selected analysis method
            correctness_text = "Correct orientation" if self.keyboard_analyzer.is_correct_orientation else "Incorrect orientation - please adjust keyboard"
            cv2.putText(display_frame, correctness_text,
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                      (0, 255, 0) if self.keyboard_analyzer.is_correct_orientation else (0, 0, 255), 2)

            # Show current analysis method
            method_text = f"Analysis Method: {self.keyboard_analyzer.analysis_method.capitalize()} (Press 'm' to change)"
            cv2.putText(display_frame, method_text,
                      (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Show keyboard usage hint if incorrect orientation
            if not self.keyboard_analyzer.is_correct_orientation:
                # Get suggestions based on the current analysis method
                if self.keyboard_analyzer.analysis_method == "hand" and self.keyboard_analyzer.hand_direction is not None:
                    suggestion = self.get_adjustment_suggestion(angle, self.keyboard_analyzer.hand_direction)
                    cv2.putText(display_frame, suggestion,
                              (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                elif self.keyboard_analyzer.analysis_method == "finger" and self.keyboard_analyzer.finger_direction is not None:
                    suggestion = self.get_finger_adjustment_suggestion(angle, self.keyboard_analyzer.finger_direction)
                    cv2.putText(display_frame, suggestion,
                              (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    # Simple suggestion based just on angle
                    if angle > 20:
                        suggestion = "Suggestion: Rotate keyboard counter-clockwise"
                    elif angle < -20:
                        suggestion = "Suggestion: Rotate keyboard clockwise"
                    else:
                        suggestion = "Suggestion: Slightly adjust keyboard orientation"
                    cv2.putText(display_frame, suggestion,
                              (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Check for finger pressing if both hand and keyboard are detected
        is_pressing = False
        if hand_landmarks and keyboard_bbox:
            is_pressing = self.key_press_detector.detect_press(frame, hand_landmarks, keyboard_bbox)

        # Draw detection results
        if keyboard_bbox:
            display_frame = self.keyboard_detector.draw_keyboard(display_frame, keyboard_bbox, keyboard_mask)

        if hand_landmarks:
            display_frame = self.hand_detector.draw_landmarks(display_frame, hand_landmarks)

        # Draw press status
        display_frame = self.key_press_detector.draw_press_status(display_frame, is_pressing)

        # Add debug info
        cv2.putText(display_frame, "Press 'q' to quit, 'm' to change analysis method", (10, display_frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return display_frame, is_pressing

    def cycle_analysis_method(self):
        """Cycle through available keyboard orientation analysis methods."""
        self.current_method_index = (self.current_method_index + 1) % len(self.analysis_methods)
        new_method = self.analysis_methods[self.current_method_index]
        self.keyboard_analyzer.set_analysis_method(new_method)
        return new_method

    def get_orientation_description(self, angle):
        """Get a human-readable description of the keyboard orientation."""
        if abs(angle) < 10:
            return "Orientation: Normal (straight)"
        elif angle > 0:
            if angle < 45:
                return f"Orientation: Slightly rotated clockwise ({angle:.1f}°)"
            else:
                return f"Orientation: Significantly rotated clockwise ({angle:.1f}°)"
        else:  # angle < 0
            if angle > -45:
                return f"Orientation: Slightly rotated counter-clockwise ({abs(angle):.1f}°)"
            else:
                return f"Orientation: Significantly rotated counter-clockwise ({abs(angle):.1f}°)"

    def get_adjustment_suggestion(self, keyboard_angle, hand_direction):
        """Get suggestion for adjusting keyboard based on hand direction."""
        # Calculate the difference between hand direction and keyboard angle
        # This helps determine if keyboard is upside down relative to hand
        angle_diff = (hand_direction - keyboard_angle + 180) % 360 - 180

        if abs(angle_diff) > 150:
            return "Suggestion: Rotate keyboard 180 degrees (it's upside down)"
        elif keyboard_angle > 20:
            return "Suggestion: Rotate keyboard counter-clockwise"
        elif keyboard_angle < -20:
            return "Suggestion: Rotate keyboard clockwise"
        else:
            return "Suggestion: Slightly adjust keyboard to align with hand position"

    def get_finger_adjustment_suggestion(self, keyboard_angle, finger_direction):
        """Get suggestion for adjusting keyboard based on finger direction."""
        # Calculate the difference between finger direction and keyboard angle
        angle_diff = abs((finger_direction - keyboard_angle + 180) % 360 - 180)
        opposite_diff = abs(angle_diff - 180)

        if min(angle_diff, opposite_diff) > 45:
            if keyboard_angle > 0:
                return "Suggestion: Rotate keyboard counter-clockwise to align with finger"
            else:
                return "Suggestion: Rotate keyboard clockwise to align with finger"
        elif keyboard_angle > 20:
            return "Suggestion: Slightly rotate keyboard counter-clockwise"
        elif keyboard_angle < -20:
            return "Suggestion: Slightly rotate keyboard clockwise"
        else:
            return "Suggestion: Keyboard is nearly aligned with finger direction"

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

    # Initialize the detector system
    system = HandKeyboardSystem()

    while cap.isOpened():
        # Read frame
        success, frame = cap.read()
        if not success:
            print("Failed to read video feed.")
            break

        # Process the frame
        display_frame, is_pressing = system.process_frame(frame)

        # Display the result
        cv2.imshow("Hand and Keyboard Detection", display_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            new_method = system.cycle_analysis_method()
            print(f"Switched to analysis method: {new_method}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
