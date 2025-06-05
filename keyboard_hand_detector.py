from ultralytics import YOLO
import cv2
import numpy as np
from keyboard_grid_updated import KeyboardGrid
from hand_detector import HandDetector

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width/2, height/2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated

def main():
    # Khởi tạo các thành phần
    model = YOLO('best.pt')
    keyboard_grid = KeyboardGrid()
    hand_detector = HandDetector()
    cap = cv2.VideoCapture(0)
    rotation_angle = 0

    print("Điều khiển:")
    print("'a' : Xoay ngược chiều kim đồng hồ 90 độ")
    print("'d' : Xoay theo chiều kim đồng hồ 90 độ")
    print("'q' : Thoát chương trình")

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Xoay frame theo góc hiện tại
            rotated_frame = rotate_image(frame, rotation_angle)

            # Phát hiện tay trong frame
            hand_landmarks = hand_detector.detect_hands(rotated_frame)

            # Chạy model YOLO để phát hiện bàn phím
            results = model(rotated_frame)

            # Cập nhật vị trí các phím
            keyboard_grid.update_detected_keys(results)

            # Vẽ kết quả phát hiện bàn phím
            annotated_frame = results[0].plot()

            # Vẽ các điểm mốc trên tay
            annotated_frame = hand_detector.draw_landmarks(annotated_frame, hand_landmarks)

            # Kiểm tra và hiển thị phím được bấm
            if hand_landmarks:
                for hand_landmark in hand_landmarks:
                    # Lấy vị trí đầu ngón trỏ (landmark số 8) và ngón giữa (landmark số 12)
                    index_finger = hand_landmark.landmark[8]
                    middle_finger = hand_landmark.landmark[12]

                    h, w, _ = annotated_frame.shape
                    index_x, index_y = int(index_finger.x * w), int(index_finger.y * h)
                    middle_x, middle_y = int(middle_finger.x * w), int(middle_finger.y * h)

                    # Vẽ điểm chạm của ngón tay
                    cv2.circle(annotated_frame, (index_x, index_y), 10, (0, 255, 0), -1)
                    cv2.circle(annotated_frame, (middle_x, middle_y), 10, (0, 0, 255), -1)

                    # Tính khoảng cách giữa hai ngón tay
                    finger_distance = np.sqrt((index_x - middle_x)**2 + (index_y - middle_y)**2)

                    # Nếu khoảng cách giữa hai ngón đủ gần (đang bấm)
                    if finger_distance < 30:  # Ngưỡng khoảng cách để xác định việc bấm phím
                        pressed_key = keyboard_grid.get_key_at_position(index_x, index_y)
                        if pressed_key:
                            cv2.putText(annotated_frame, f"Pressed: {pressed_key}", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            print(f"Phím được bấm: {pressed_key}")

            # Vẽ lưới bàn phím
            final_frame = keyboard_grid.draw_keyboard_visualization(annotated_frame)

            # Hiển thị frame
            cv2.imshow("Keyboard Hand Detection", final_frame)

            # Xử lý phím điều khiển
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                rotation_angle = (rotation_angle - 90) % 360
            elif key == ord("d"):
                rotation_angle = (rotation_angle + 90) % 360
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
