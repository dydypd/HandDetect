import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import math

class HandKeyboardDetector:
    def __init__(self, yolo_model_path="yolo11n-seg.pt", confidence_threshold=0.25):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize YOLO for keyboard detection
        self.yolo_model = YOLO(yolo_model_path)
        self.confidence_threshold = confidence_threshold

        # Define keyboard class ID - we'll detect any object since it could be keyboard
        # This will be overridden by finding the keyboard in the class names
        self.keyboard_class_id = None

        # Get model class names and find keyboard class
        self.class_names = self.yolo_model.names
        for class_id, name in self.class_names.items():
            if 'keyboard' in name.lower():
                self.keyboard_class_id = class_id
                print(f"Found keyboard class ID: {class_id} with name: {name}")

        # If keyboard class not found, we'll try to detect any object as fallback
        if self.keyboard_class_id is None:
            print("Keyboard class not found in model. Will attempt to detect based on shape.")

        # Variables to store detection results
        self.hand_landmarks = []
        self.keyboard_bbox = None
        self.keyboard_mask = None

        # Finger pressing detection parameters
        self.prev_fingertip_positions = []
        self.movement_threshold = 5  # pixels
        self.pressing_threshold = 10  # pixels
        self.is_pressing = False

    def detect_hands(self, frame):
        """Detect hand landmarks using MediaPipe."""
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = self.hands.process(rgb_frame)

        # Store and return the hand landmarks
        self.hand_landmarks = []
        if results.multi_hand_landmarks:
            self.hand_landmarks = results.multi_hand_landmarks

        return self.hand_landmarks

    def detect_keyboard(self, frame):
        """Detect keyboard using YOLO model."""
        # Run YOLO detection
        results = self.yolo_model(frame)[0]

        # Reset keyboard detection results
        self.keyboard_bbox = None
        self.keyboard_mask = None

        # Debug: print all detected objects
        print("\nDetected objects:")
        for i, (box, score, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
            class_id = int(cls)
            class_name = self.class_names[class_id] if class_id in self.class_names else f"Unknown class {class_id}"
            print(f"Object {i+1}: {class_name} (ID: {class_id}), confidence: {score:.2f}")

        # Process YOLO results - try to find keyboard first
        if self.keyboard_class_id is not None:
            # Look for the keyboard class specifically
            for i, (box, score, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
                if score >= self.confidence_threshold and int(cls) == self.keyboard_class_id:
                    # Store keyboard bounding box
                    x1, y1, x2, y2 = map(int, box)
                    self.keyboard_bbox = (x1, y1, x2, y2)
                    print(f"Keyboard detected at {self.keyboard_bbox} with confidence {score:.2f}")

                    # If segmentation masks are available
                    if hasattr(results, 'masks') and results.masks is not None:
                        # Get the segmentation mask for the keyboard
                        self.keyboard_mask = results.masks.data[i].cpu().numpy()

                    # Only consider the first detected keyboard
                    break

        # If no keyboard detected, fallback to detecting any rectangular object
        if self.keyboard_bbox is None and len(results.boxes) > 0:
            # Try to detect based on shape - looking for rectangular objects that could be keyboards
            for i, (box, score, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
                if score >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    width, height = x2 - x1, y2 - y1
                    aspect_ratio = width / height if height > 0 else 0

                    # Typical keyboard has aspect ratio around 3:1 to 5:1
                    if 2.0 <= aspect_ratio <= 6.0 and width > 100:
                        self.keyboard_bbox = (x1, y1, x2, y2)
                        print(f"Possible keyboard (by shape) detected at {self.keyboard_bbox} with confidence {score:.2f}")

                        # If segmentation masks are available
                        if hasattr(results, 'masks') and results.masks is not None:
                            # Get the segmentation mask for the object
                            self.keyboard_mask = results.masks.data[i].cpu().numpy()

                        # Only consider the first detected possible keyboard
                        break

        return self.keyboard_bbox, self.keyboard_mask

    def is_finger_pressing(self, current_landmarks):
        """Detect if a finger is in the pressing state."""
        if not self.keyboard_bbox or not current_landmarks:
            return False

        # Get keyboard coordinates
        kx1, ky1, kx2, ky2 = self.keyboard_bbox

        # Extract fingertip positions (index, middle, ring, pinky)
        fingertips = [
            (current_landmarks[8].x, current_landmarks[8].y),  # Index fingertip
            (current_landmarks[12].x, current_landmarks[12].y),  # Middle fingertip
            (current_landmarks[16].x, current_landmarks[16].y),  # Ring fingertip
            (current_landmarks[20].x, current_landmarks[20].y),  # Pinky fingertip
        ]

        # Convert normalized coordinates to pixel coordinates
        h, w, _ = self.current_frame.shape
        fingertips_px = [(int(x * w), int(y * h)) for x, y in fingertips]

        # Check if we have previous positions to compare
        if not self.prev_fingertip_positions:
            self.prev_fingertip_positions = fingertips_px
            return False

        # Check if any fingertip is above the keyboard and moving downward
        pressing_detected = False
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
                    lower_joint = (current_landmarks[i*4+6].x, current_landmarks[i*4+6].y)  # Lower joint of the finger
                    lower_joint_px = (int(lower_joint[0] * w), int(lower_joint[1] * h))

                    # If the fingertip is significantly lower than the joint, it's likely pressing
                    if curr_y - lower_joint_px[1] > self.pressing_threshold:
                        pressing_detected = True
                        break

        # Update previous positions
        self.prev_fingertip_positions = fingertips_px

        # Update and return pressing state
        self.is_pressing = pressing_detected
        return self.is_pressing

    def process_frame(self, frame):
        """Process a frame for hand and keyboard detection."""
        self.current_frame = frame

        # Detect keyboard
        keyboard_bbox, keyboard_mask = self.detect_keyboard(frame)

        # Detect hands
        hand_landmarks = self.detect_hands(frame)

        # Check for finger pressing if both hand and keyboard are detected
        finger_pressing = False
        if hand_landmarks and keyboard_bbox:
            for landmarks in hand_landmarks:
                if self.is_finger_pressing(landmarks.landmark):
                    finger_pressing = True
                    break

        # Draw results on the frame
        result_frame = self.draw_results(frame.copy(), hand_landmarks, keyboard_bbox, keyboard_mask, finger_pressing)

        return result_frame, finger_pressing

    def draw_results(self, frame, hand_landmarks, keyboard_bbox, keyboard_mask, finger_pressing):
        """Draw detection results on the frame."""
        # Draw keyboard bounding box and mask
        if keyboard_bbox:
            x1, y1, x2, y2 = keyboard_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add text label
            cv2.putText(frame, "Keyboard", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Draw segmentation mask if available
            if keyboard_mask is not None:
                try:
                    # Convert mask to binary (0 or 1)
                    binary_mask = (keyboard_mask > 0.5).astype(np.uint8)

                    # Make sure the mask is the right shape for the frame
                    if binary_mask.shape[:2] != frame.shape[:2]:
                        if len(binary_mask.shape) == 2:
                            binary_mask = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]))
                        else:
                            print(f"Warning: Mask shape {binary_mask.shape} doesn't match frame shape {frame.shape}")
                            binary_mask = None

                    if binary_mask is not None:
                        # Create colored mask overlay
                        colored_mask = np.zeros_like(frame)
                        colored_mask[:, :, 1] = binary_mask * 255  # Green channel

                        # Blend mask with frame
                        alpha = 0.3  # Transparency factor
                        frame = cv2.addWeighted(frame, 1, colored_mask, alpha, 0)
                except Exception as e:
                    print(f"Error applying mask: {e}")

        # Draw hand landmarks
        if hand_landmarks:
            for landmarks in hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

        # Show finger pressing status
        if finger_pressing:
            cv2.putText(frame, "KEY PRESS DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Add debugging info
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

    # Initialize the detector
    detector = HandKeyboardDetector()

    while cap.isOpened():
        # Read frame
        success, frame = cap.read()
        if not success:
            print("Failed to read video feed.")
            break

        # Process the frame
        result_frame, is_pressing = detector.process_frame(frame)

        # Display the result
        cv2.imshow("Hand and Keyboard Detection", result_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
