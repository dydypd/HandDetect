import cv2
import numpy as np
import torch
from ultralytics import YOLO

class PhoneScreenDetector:
    def __init__(self, model_path='yolo11n-seg.pt', confidence_threshold=0.5):
        """
        Initialize the phone screen detector using YOLOv11 segmentation model.

        Args:
            model_path: Path to the YOLOv11 segmentation model
            confidence_threshold: Minimum confidence threshold for detections
        """
        self.confidence_threshold = confidence_threshold
        # Load YOLOv11 segmentation model
        self.model = YOLO(model_path)

        # Set class ID for phone screens if your model has been trained to detect them
        # Default COCO classes include cell phone (67), but we'll need to check if screen is separately defined
        self.phone_class_ids = [67]  # default COCO class ID for cell phone

    def detect_phone_screens(self, frame):
        """
        Detect and segment phone screens in the given frame.

        Args:
            frame: Input BGR image/frame

        Returns:
            detected_screens: List of dictionaries containing segmentation masks, bounding boxes, and confidence
            annotated_frame: Frame with visualization of the detected phone screens
        """
        # Make a copy of the frame for annotation
        annotated_frame = frame.copy()

        # Run YOLOv11 segmentation on the frame
        results = self.model(frame, verbose=False)

        # Process results
        detected_screens = []

        for result in results:
            if result.masks is not None:
                boxes = result.boxes
                masks = result.masks

                # Process each detection
                for i, (box, mask) in enumerate(zip(boxes, masks)):
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())

                    # If detection is a phone and confidence is above threshold
                    if cls_id in self.phone_class_ids and conf > self.confidence_threshold:
                        # Get bounding box coordinates (x1, y1, x2, y2)
                        x1, y1, x2, y2 = map(int, box.xyxy.squeeze().tolist())

                        # Get segmentation mask safely
                        try:
                            # Get the mask data and convert to numpy
                            seg_mask_data = mask.data.cpu().numpy()

                            # Handle different mask dimensions
                            if len(seg_mask_data.shape) == 3:
                                seg_mask_data = seg_mask_data[0]  # Take the first mask if multiple

                            # Convert to uint8
                            seg_mask_data = seg_mask_data.astype(np.uint8)

                            # Create a full frame mask
                            h, w = frame.shape[:2]
                            full_mask = np.zeros((h, w), dtype=np.uint8)

                            # If mask and frame dimensions are valid
                            if seg_mask_data.size > 0 and h > 0 and w > 0:
                                # Option 1: If mask is already properly sized, use it directly
                                if seg_mask_data.shape == (h, w):
                                    full_mask = seg_mask_data
                                # Option 2: Place a scaled version of the mask at the box location
                                else:
                                    try:
                                        # Ensure the box has valid dimensions
                                        box_h, box_w = max(1, y2-y1), max(1, x2-x1)

                                        # Scale the mask to fit the box
                                        # Only resize if mask has valid dimensions
                                        if seg_mask_data.shape[0] > 0 and seg_mask_data.shape[1] > 0:
                                            scaled_mask = cv2.resize(seg_mask_data, (box_w, box_h))
                                            # Place the scaled mask in the right position
                                            full_mask[y1:y2, x1:x2] = scaled_mask
                                        else:
                                            # If mask is invalid, just fill the box
                                            full_mask[y1:y2, x1:x2] = 255
                                    except Exception as e:
                                        print(f"Mask scaling error: {e}")
                                        # Fall back to simple box mask
                                        full_mask[y1:y2, x1:x2] = 255
                            else:
                                # If we can't process the mask properly, use the bounding box
                                full_mask[y1:y2, x1:x2] = 255

                            # Use the full mask
                            seg_mask = full_mask

                        except Exception as e:
                            print(f"Mask processing error: {e}")
                            # Create a fallback mask based on the bounding box
                            seg_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            seg_mask[y1:y2, x1:x2] = 255

                        # Store detection information
                        detected_screens.append({
                            'bbox': (x1, y1, x2, y2),
                            'mask': seg_mask,
                            'confidence': conf
                        })

                        # Annotate the frame (for visualization)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f'Phone: {conf:.2f}', (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Apply colored mask overlay for visualization
                        colored_mask = np.zeros_like(frame)
                        colored_mask[seg_mask > 0] = [0, 0, 255]  # Red color for mask
                        annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.5, 0)

        return detected_screens, annotated_frame

    def extract_screen_content(self, frame, detected_screen):
        """
        Extract and process the content of a detected phone screen.

        Args:
            frame: Input BGR image/frame
            detected_screen: Dictionary containing detection information

        Returns:
            screen_content: Extracted and potentially perspective-corrected screen content
        """
        # Get mask and bounding box
        mask = detected_screen['mask']
        x1, y1, x2, y2 = detected_screen['bbox']

        # Apply mask to extract only the screen pixels
        screen_region = frame.copy()
        screen_region[mask == 0] = 0

        # Crop to bounding box
        screen_content = screen_region[y1:y2, x1:x2]

        # Additional processing could be done here, such as:
        # - Perspective correction if the phone is at an angle
        # - Glare/reflection removal
        # - Contrast enhancement

        return screen_content

# Example usage:
if __name__ == "__main__":
    # Initialize detector
    detector = PhoneScreenDetector()

    # Open camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect phone screens
        detected_screens, annotated_frame = detector.detect_phone_screens(frame)

        # Process each detected screen if needed
        for i, screen in enumerate(detected_screens):
            screen_content = detector.extract_screen_content(frame, screen)

            # Display extracted screen content if available
            if screen_content.size > 0:
                # Resize for display if needed
                h, w = screen_content.shape[:2]
                if h > 0 and w > 0:
                    screen_display = cv2.resize(screen_content, (300, int(300 * h / w)))
                    cv2.imshow(f"Screen {i}", screen_display)

        # Display the annotated frame
        cv2.imshow("Phone Screen Detection", annotated_frame)

        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
