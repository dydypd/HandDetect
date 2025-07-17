import cv2
import numpy as np
from ultralytics import YOLO
from keyboard_grid import KeyboardGrid
from key_input_tracker import KeyInputTracker
import time
from dotenv import load_dotenv
load_dotenv()

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width/2, height/2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated

def transform_coordinates(x, y, angle, image_width, image_height):
    """Chuyển đổi tọa độ từ frame gốc sang frame đã xoay"""
    # Điểm trung tâm của ảnh
    center_x, center_y = image_width / 2, image_height / 2

    # Chuyển đổi góc sang radian (đảo dấu để đổi chiều xoay)
    angle_rad = np.radians(-angle)

    # Tịnh tiến về gốc tọa độ
    x_centered = x - center_x
    y_centered = y - center_y

    # Xoay tọa độ
    x_rotated = x_centered * np.cos(angle_rad) - y_centered * np.sin(angle_rad)
    y_rotated = x_centered * np.sin(angle_rad) + y_centered * np.cos(angle_rad)

    # Tịnh tiến trở lại
    x_new = x_rotated + center_x
    y_new = y_rotated + center_y

    return x_new, y_new

def main():
    # Khởi tạo các thành phần
    keyboard_model = YOLO('best_v2.pt')  # Model phát hiện bàn phím
    keypress_model = YOLO('best (1).pt')  # Model phát hiện trạng thái bấm phím
    keyboard_grid = KeyboardGrid()
    # Khởi tạo KeyInputTracker với các tham số phù hợp
    key_input_tracker = KeyInputTracker(
        cooldown_time=0.1,          # Thời gian chờ giữa các lần bấm phím (0.5 giây)
        confidence_threshold=0.4,    # Ngưỡng tin cậy tối thiểu
        consistency_required=2       # Số frame liên tiếp cần phát hiện cùng phím
    )
    # Tạm thời bỏ hand_detector và finger_motion_analyzer
    # hand_detector = HandDetector()
    # finger_motion_analyzer = FingerMotionAnalyzer(history_length=10, voting_window=5)
    cap = cv2.VideoCapture(0)
    rotation_angle = 0

    # Các biến mới để quản lý việc phát hiện bàn phím
    keyboard_detection_interval = 30  # Số frame giữa các lần phát hiện bàn phím
    frame_count = 0
    keyboard_bbox = None
    keyboard_detection_active = True  # Trạng thái phát hiện bàn phím
    keyboard_stable_count = 0  # Đếm số frame bàn phím ổn định
    keyboard_stable_threshold = 5  # Số frame cần để xác nhận bàn phím ổn định

    # Biến để theo dõi trạng thái bấm phím
    pressing_detected = False
    pressing_confidence = 0

    print("Điều khiển:")
    print("'a' : Xoay ngược chiều kim đồng hồ 90 độ")
    print("'d' : Xoay theo chiều kim đồng hồ 90 độ")
    print("'r' : Phát hiện lại bàn phím")
    print("'s' : Tạm dừng/tiếp tục phát hiện bàn phím")
    print("'c' : Xóa văn bản đã nhập")
    print("'f' : Lưu văn bản đã nhập vào file")
    print("'q' : Thoát chương trình")

    print("Đang phát hiện bàn phím, vui lòng đợi...")

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Xoay frame theo góc hiện tại
            rotated_frame = rotate_image(frame, rotation_angle)

            # Sử dụng frame gốc
            annotated_frame = rotated_frame.copy()

            # Chỉ phát hiện bàn phím khi cần thiết
            need_keyboard_detection = (
                keyboard_detection_active and
                (frame_count % keyboard_detection_interval == 0 or keyboard_bbox is None)
            )

            if need_keyboard_detection:
                # Chạy model YOLO để phát hiện bàn phím
                start_time = time.time()
                results = keyboard_model(rotated_frame)
                detection_time = time.time() - start_time

                # Hiển thị thời gian phát hiện
                cv2.putText(annotated_frame, f"Detection time: {detection_time:.2f}s",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Cập nhật vị trí các phím
                keyboard_grid.update_detected_keys(results)

                # Tính toán bbox của bàn phím dựa trên các phím đã phát hiện
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

                    new_keyboard_bbox = (min_x, min_y, max_x, max_y)

                    # Nếu đã có bbox trước đó, kiểm tra sự ổn định
                    if keyboard_bbox is not None:
                        # Tính toán sự thay đổi giữa bbox cũ và mới
                        bbox_change = sum([abs(a-b) for a, b in zip(keyboard_bbox, new_keyboard_bbox)])
                        if bbox_change < 50:  # Ngưỡng thay đổi nhỏ
                            keyboard_stable_count += 1
                        else:
                            keyboard_stable_count = 0

                    keyboard_bbox = new_keyboard_bbox

                    # Khi bàn phím ổn định, giảm tần suất phát hiện
                    if keyboard_stable_count >= keyboard_stable_threshold:
                        keyboard_detection_interval = 100  # Tăng khoảng thời gian giữa các lần phát hiện
                        print("Bàn phím đã ổn định, giảm tần suất phát hiện")
            else:
                # Sử dụng kết quả phát hiện đã lưu từ trước
                pass

            # Chạy model phát hiện trạng thái bấm
            keypress_results = keypress_model(frame)

            # Xử lý kết quả phát hiện trạng thái bấm
            pressing_detected = False
            pressing_confidence = 0
            press_bbox = None  # Lưu bounding box của phát hiện bấm phím

            # Kiểm tra các kết quả từ model
            for result in keypress_results:
                # Xử lý kết quả theo định dạng dành cho model detection
                for det in result.boxes.data:
                    # Det là một tensor với format [x1, y1, x2, y2, confidence, class_id]
                    confidence = float(det[4])
                    class_id = int(det[5])
                    class_name = result.names[class_id]

                    if class_name == "press" and confidence > 0.3:  # Điều chỉnh ngưỡng tin cậy nếu cần
                        pressing_detected = True
                        pressing_confidence = confidence
                        # Lưu bounding box của phát hiện
                        press_bbox = [float(det[0]), float(det[1]), float(det[2]), float(det[3])]
                        break

            # Hiển thị kết quả phát hiện trạng thái bấm
            pressed_key = None
            if pressing_detected and press_bbox:
                # Lấy kích thước của frame
                height, width = frame.shape[:2]

                # Chuyển đổi các điểm của bounding box từ frame gốc sang frame đã xoay
                x1, y1 = transform_coordinates(press_bbox[0], press_bbox[1], rotation_angle, width, height)
                x2, y2 = transform_coordinates(press_bbox[2], press_bbox[3], rotation_angle, width, height)

                # Đảm bảo thứ tự các điểm sau khi xoay là đúng (x1,y1 là góc trên bên trái, x2,y2 là góc dưới bên phải)
                xmin = min(x1, x2)
                ymin = min(y1, y2)
                xmax = max(x1, x2)
                ymax = max(y1, y2)

                # Vẽ bounding box của vùng bấm đã được xoay
                cv2.rectangle(annotated_frame,
                             (int(xmin), int(ymin)),
                             (int(xmax), int(ymax)),
                             (0, 0, 255), 2)

                # Tính toán điểm trung tâm của bounding box đã xoay
                center_x, center_y = transform_coordinates(
                    (press_bbox[0] + press_bbox[2]) / 2,
                    (press_bbox[1] + press_bbox[3]) / 2,
                    rotation_angle, width, height
                )

                # Vẽ điểm trung tâm
                cv2.circle(annotated_frame, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

                # Xác định phím được bấm dựa trên tọa độ trung tâm đã xoay
                pressed_key = keyboard_grid.get_key_at_position(center_x, center_y)

                # Hiển thị thông tin phím được bấm
                if pressed_key:
                    cv2.putText(annotated_frame, f"PRESSING: {pressed_key} ({pressing_confidence:.2f})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Cập nhật bộ theo dõi nhập liệu
                    text_changed = key_input_tracker.update(pressed_key, pressing_confidence, True)
                    if text_changed:
                        print(f"Chuỗi văn bản hiện tại: {key_input_tracker.get_text()}")
                else:
                    cv2.putText(annotated_frame, f"PRESSING DETECTED! ({pressing_confidence:.2f})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Nếu không phát hiện thấy phím bấm, cập nhật bộ theo dõi với trạng thái không bấm
                key_input_tracker.update(None, 0, False)

            # Hiển thị văn bản đã nhập
            current_text = key_input_tracker.get_text()
            if current_text:
                # Tạo một vùng hiển thị văn bản màu đen với độ trong suốt
                text_overlay = annotated_frame.copy()
                cv2.rectangle(text_overlay, (10, annotated_frame.shape[0] - 60),
                             (annotated_frame.shape[1] - 10, annotated_frame.shape[0] - 10),
                             (0, 0, 0), -1)

                # Pha trộn hình ảnh gốc với vùng đen để tạo hiệu ứng trong suốt
                alpha = 0.7
                cv2.addWeighted(text_overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

                # Hiển thị văn bản đã nhập
                cv2.putText(annotated_frame, f"Text: {current_text}",
                           (20, annotated_frame.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Hiển thị trạng thái phát hiện bàn phím
            status_text = "Keyboard detection: ON" if keyboard_detection_active else "Keyboard detection: OFF"
            cv2.putText(annotated_frame, status_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Vẽ bbox của bàn phím nếu có
            if keyboard_bbox:
                min_x, min_y, max_x, max_y = keyboard_bbox
                cv2.rectangle(annotated_frame,
                             (int(min_x), int(min_y)), 
                             (int(max_x), int(max_y)), 
                             (0, 255, 0), 2)

            # Vẽ lưới bàn phím
            final_frame = keyboard_grid.draw_keyboard_visualization(annotated_frame)

            # Hiển thị frame
            cv2.imshow("Keyboard Hand Detection", final_frame)

            # Tăng bộ đếm frame
            frame_count += 1

            # Xử lý phím điều khiển
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("a"):
                rotation_angle = (rotation_angle - 90) % 360
                # Đặt lại phát hiện bàn phím khi xoay
                keyboard_bbox = None
                keyboard_stable_count = 0
                keyboard_detection_interval = 30
            elif key == ord("d"):
                rotation_angle = (rotation_angle + 90) % 360
                # Đặt lại phát hiện bàn phím khi xoay
                keyboard_bbox = None
                keyboard_stable_count = 0
                keyboard_detection_interval = 30
            elif key == ord("r"):
                # Phát hiện lại bàn phím
                keyboard_bbox = None
                keyboard_stable_count = 0
                keyboard_detection_interval = 30
                print("Đang phát hiện lại bàn phím...")
            elif key == ord("s"):
                # Bật/tắt phát hiện bàn phím
                keyboard_detection_active = not keyboard_detection_active
                print(f"Phát hiện bàn phím: {'Bật' if keyboard_detection_active else 'Tắt'}")
            elif key == ord("c"):
                # Xóa văn bản đã nhập
                key_input_tracker.clear_text()
                print("Đã xóa văn bản đã nhập.")
            elif key == ord("f"):
                # Lưu văn bản đã nhập vào file
                with open("output.txt", "a", encoding="utf-8") as f:
                    f.write(key_input_tracker.get_text() + "\n")
                print("Đã lưu văn bản vào file output.txt")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
