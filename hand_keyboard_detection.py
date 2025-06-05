import cv2
from hand_detector import HandDetector
from keyboard_detector import KeyboardDetector
from key_press_detector import KeyPressDetector

class HandKeyboardSystem:
    def __init__(self, yolo_model_path="yolo11n-seg.pt"):
        """Initialize the hand and keyboard detection system."""
        # Initialize component detectors
        self.hand_detector = HandDetector()
        self.keyboard_detector = KeyboardDetector(model_path=yolo_model_path)
        self.key_press_detector = KeyPressDetector()

    def process_frame(self, frame):
        """Process a frame for hand and keyboard detection."""
        # Make a copy of the frame for drawing
        display_frame = frame.copy()

        # Detect keyboard
        keyboard_bbox, keyboard_mask = self.keyboard_detector.detect_keyboard(frame)

        # Detect hands
        hand_landmarks = self.hand_detector.detect_hands(frame)

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
        cv2.putText(display_frame, "Press 'q' to quit", (10, display_frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return display_frame, is_pressing

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

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
