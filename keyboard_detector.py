import cv2
import numpy as np
from ultralytics import YOLO

class KeyboardDetector:
    def __init__(self, model_path="yolo11n-seg.pt", confidence_threshold=0.25):
        """Initialize YOLO model for keyboard detection."""
        self.yolo_model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # Get model class names
        self.class_names = self.yolo_model.names

        # Find keyboard class ID
        self.keyboard_class_id = None
        for class_id, name in self.class_names.items():
            if 'keyboard' in name.lower():
                self.keyboard_class_id = class_id
                print(f"Found keyboard class ID: {class_id} with name: {name}")

        # If keyboard class not found, we'll try to detect any object as fallback
        if self.keyboard_class_id is None:
            print("Keyboard class not found in model. Will attempt to detect based on shape.")

    def detect_keyboard(self, frame):
        """Detect keyboard using YOLO model."""
        # Run YOLO detection
        results = self.yolo_model(frame)[0]

        # Initialize detection results
        keyboard_bbox = None
        keyboard_mask = None

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
                    keyboard_bbox = (x1, y1, x2, y2)
                    print(f"Keyboard detected at {keyboard_bbox} with confidence {score:.2f}")

                    # If segmentation masks are available
                    if hasattr(results, 'masks') and results.masks is not None:
                        # Get the segmentation mask for the keyboard
                        keyboard_mask = results.masks.data[i].cpu().numpy()

                    # Only consider the first detected keyboard
                    break

        # If no keyboard detected, fallback to detecting any rectangular object
        if keyboard_bbox is None and len(results.boxes) > 0:
            # Try to detect based on shape - looking for rectangular objects that could be keyboards
            for i, (box, score, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls)):
                if score >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    width, height = x2 - x1, y2 - y1
                    aspect_ratio = width / height if height > 0 else 0

                    # Typical keyboard has aspect ratio around 3:1 to 5:1
                    if 2.0 <= aspect_ratio <= 6.0 and width > 100:
                        keyboard_bbox = (x1, y1, x2, y2)
                        print(f"Possible keyboard (by shape) detected at {keyboard_bbox} with confidence {score:.2f}")

                        # If segmentation masks are available
                        if hasattr(results, 'masks') and results.masks is not None:
                            # Get the segmentation mask for the object
                            keyboard_mask = results.masks.data[i].cpu().numpy()

                        # Only consider the first detected possible keyboard
                        break

        return keyboard_bbox, keyboard_mask

    def draw_keyboard(self, frame, keyboard_bbox, keyboard_mask):
        """Draw keyboard detection results on the frame."""
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

        return frame
