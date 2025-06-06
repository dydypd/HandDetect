import cv2
import numpy as np
from ultralytics import YOLO
from keyboard_grid import KeyboardGrid
from hand_detector import HandDetector
from finger_motion_analyzer import FingerMotionAnalyzer

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
    finger_motion_analyzer = FingerMotionAnalyzer(history_length=10, voting_window=5)
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

            # Sử dụng frame gốc thay vì vẽ kết quả YOLO
            annotated_frame = rotated_frame.copy()

            # Vẽ các điểm mốc trên tay
            annotated_frame = hand_detector.draw_landmarks(annotated_frame, hand_landmarks)

            # Tính toán bbox của bàn phím dựa trên các phím đã phát hiện
            keyboard_bbox = None
            detected_keys = [key for key, info in keyboard_grid.key_info.items() if info['detected']]
            
            if detected_keys:
                min_x, min_y = float('inf'), float('inf')
                max_x, max_y = 0, 0
                
                for key in detected_keys:
                    bbox = keyboard_grid.key_info[key]['bbox']
                    if bbox is not None:
                        min_x = min(min_x, bbox[0])
                        min_y = min(min_y, bbox[1])
                        max_x = max(max_x, bbox[2])
                        max_y = max(max_y, bbox[3])
                
                keyboard_bbox = (min_x, min_y, max_x, max_y)
                
                # Vẽ bbox của bàn phím
                cv2.rectangle(annotated_frame, 
                             (int(min_x), int(min_y)), 
                             (int(max_x), int(max_y)), 
                             (0, 255, 0), 2)

            # Sử dụng phân tích chuyển động ngón tay để phát hiện phím bấm
            is_pressing, pressing_finger = finger_motion_analyzer.analyze_motion(
                hand_landmarks, rotated_frame, keyboard_bbox
            )
            
            # Vẽ thông tin phân tích chuyển động
            annotated_frame = finger_motion_analyzer.draw_motion_analysis(annotated_frame)

            # Kiểm tra và hiển thị phím được bấm
            if is_pressing:
                pressed_key = finger_motion_analyzer.get_key_at_position(keyboard_grid)
                if pressed_key and pressing_finger:
                    hand_id, finger_id = pressing_finger
                    finger_name = finger_motion_analyzer.finger_names[finger_id]
                    hand_name = "LEFT" if hand_id == 0 else "RIGHT"
                    
                    # Hiển thị thông báo phím bấm với thông tin ngón tay
                    cv2.putText(annotated_frame, f"Bam: {pressed_key} - {hand_name} {finger_name}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    print(f"Phím được bấm: {pressed_key} bằng {hand_name} {finger_name}")
                elif pressing_finger:
                    hand_id, finger_id = pressing_finger
                    finger_name = finger_motion_analyzer.finger_names[finger_id]
                    hand_name = "LEFT" if hand_id == 0 else "RIGHT"
                    # cv2.putText(annotated_frame, f"Bam: {hand_name} {finger_name} (khong co phim)", (10, 30),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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
