from ultralytics import YOLO
import cv2
import numpy as np
from keyboard_grid import KeyboardGrid

# Load the model
model = YOLO('best.pt')
print("Model loaded successfully.")

# Khởi tạo lưới bàn phím
keyboard_grid = KeyboardGrid()

# Open the webcam
cap = cv2.VideoCapture(0)

# Khởi tạo góc xoay
rotation_angle = 0

print("Điều khiển:")
print("'a' : Xoay ngược chiều kim đồng hồ 90 độ")
print("'d' : Xoay theo chiều kim đồng hồ 90 độ")
print("'q' : Thoát chương trình")

def rotate_image(image, angle):
    # Lấy kích thước của ảnh
    height, width = image.shape[:2]
    # Tính toán ma trận xoay
    center = (width/2, height/2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Thực hiện phép xoay
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated

while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()

    if success:
        # Xoay frame theo góc hiện tại
        rotated_frame = rotate_image(frame, rotation_angle)

        # Run YOLOv8 inference on the rotated frame
        results = model(rotated_frame, boxes=False)

        # Cập nhật vị trí các phím đã phát hiện và nội suy các phím chưa phát hiện
        keyboard_grid.update_detected_keys(results)

        # Vẽ visualization lên frame
        annotated_frame = results[0].plot(boxes=False)
        final_frame = keyboard_grid.draw_keyboard_visualization(annotated_frame)

        # Display the annotated frame
        cv2.imshow("Keyboard Detection", final_frame)

        # Xử lý phím bấm
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("a"):  # Xoay ngược chiều kim đồng hồ
            rotation_angle = (rotation_angle - 90) % 360
        elif key == ord("d"):  # Xoay theo chiều kim đồng hồ
            rotation_angle = (rotation_angle + 90) % 360
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
