# Tham số có thể điều chỉnh trong Hệ thống Phân tích Hướng Bàn phím

Tài liệu này mô tả các tham số có thể điều chỉnh trong hệ thống phân tích hướng bàn phím và tay người dùng. Việc hiểu và tinh chỉnh các tham số này sẽ giúp hệ thống hoạt động chính xác hơn trong các môi trường và tình huống khác nhau.

## 1. Tham số phát hiện bàn phím

### Trong lớp `KeyboardDetector`

| Tham số | Mô tả | Giá trị mặc định | Ảnh hưởng khi điều chỉnh |
|---------|-------|-----------------|-------------------------|
| `model_path` | Đường dẫn đến mô hình YOLO được sử dụng để phát hiện bàn phím | "yolo11n-seg.pt" | Thay đổi mô hình có thể ảnh hưởng đến tốc độ và độ chính xác |
| `confidence_threshold` | Ngưỡng tin cậy tối thiểu để xác nhận phát hiện bàn phím | 0.25 | Tăng giá trị này sẽ giảm phát hiện sai, nhưng có thể bỏ lỡ các bàn phím thật |

## 2. Tham số phân tích hướng bàn phím

### Trong lớp `KeyboardSegmentAnalyzer`

| Tham số | Mô tả | Giá trị hiện tại | Ảnh hưởng khi điều chỉnh |
|---------|-------|-----------------|-------------------------|
| `analysis_method` | Phương pháp phân tích hướng bàn phím (angle, hand, finger) | "finger" | Thay đổi phương pháp phân tích hướng bàn phím |
| `finger_extension_threshold` | Ngưỡng tối thiểu để xác định ngón tay đã duỗi đủ | 5 pixel | Giảm giá trị sẽ tăng độ nhạy, tăng giá trị sẽ yêu cầu ngón tay duỗi rõ ràng hơn |
| `finger_angle_tolerance` | Dung sai góc khi xác định liệu ngón tay có vuông góc với cạnh dài | 35 độ | Tăng giá trị tăng độ dung sai, giảm giá trị tăng độ chính xác |
| `keyboard_angle_tolerance` | Dung sai góc khi xác định liệu bàn phím có thẳng | 30 độ | Tăng giá trị cho phép bàn phím nghiêng nhiều hơn vẫn được coi là đúng |

## 3. Tham số phát hiện tay

### Trong lớp `HandDetector`

| Tham số | Mô tả | Giá trị mặc định | Ảnh hưởng khi điều chỉnh |
|---------|-------|-----------------|-------------------------|
| `static_image_mode` | Chế độ phát hiện trong ảnh tĩnh hay video | False | True sẽ phát hiện chính xác hơn nhưng chậm hơn |
| `max_num_hands` | Số lượng tay tối đa có thể phát hiện | 2 | Tăng giá trị cho phép phát hiện nhiều tay hơn, nhưng tốn nhiều tài nguyên |
| `min_detection_confidence` | Ngưỡng tin cậy tối thiểu để phát hiện tay | 0.7 | Giảm giá trị sẽ phát hiện được nhiều tay hơn, nhưng có thể có nhiều phát hiện sai |
| `min_tracking_confidence` | Ngưỡng tin cậy tối thiểu để theo dõi tay | 0.5 | Tăng giá trị cho độ ổn định tốt hơn, giảm giá trị cho độ nhạy tốt hơn |

## 4. Cách điều chỉnh các tham số

### Điều chỉnh phương pháp phân tích hướng bàn phím
Trong giao diện người dùng, bạn có thể nhấn phím 'm' để chuyển đổi giữa các phương pháp:
- `angle`: Chỉ dựa vào góc nghiêng của bàn phím
- `hand`: Dựa vào vị trí của tay so với bàn phím
- `finger`: Dựa vào hướng của ngón tay (vuông góc với cạnh dài của bàn phím)

### Điều chỉnh thông qua code
Để điều chỉnh các tham số khác, bạn cần chỉnh sửa trực tiếp trong code:

```python
# Điều chỉnh độ nhạy phát hiện ngón tay trong keyboard_segment_analyzer.py
if abs(dx) < 5 and abs(dy) < 5:  # Có thể điều chỉnh giá trị 5 này
    self.is_correct_orientation = abs(self.keyboard_orientation) < 30
    return

# Điều chỉnh dung sai góc
self.is_correct_orientation = short_edge_diff < 35  # Có thể điều chỉnh giá trị 35 này
```

### Điều chỉnh tham số phát hiện tay
Trong file `hand_detector.py`, bạn có thể điều chỉnh các tham số MediaPipe Hands:

```python
self.hands = self.mp_hands.Hands(
    static_image_mode=static_image_mode,  # False cho video, True cho ảnh tĩnh
    max_num_hands=max_num_hands,  # Số lượng tay tối đa (thường là 2)
    min_detection_confidence=min_detection_confidence,  # Ngưỡng phát hiện (0.5-0.7)
    min_tracking_confidence=min_tracking_confidence  # Ngưỡng theo dõi (0.5-0.7)
)
```

## 5. Ví dụ điều chỉnh cho các tình huống cụ thể

### Tăng độ nhạy (phát hiện chuyển động nhỏ)
- Giảm `finger_extension_threshold` xuống 3-5 pixel
- Tăng `finger_angle_tolerance` lên 35-40 độ
- Giảm `min_detection_confidence` xuống 0.5-0.6

### Tăng độ chính xác (giảm phát hiện sai)
- Tăng `finger_extension_threshold` lên 10-15 pixel
- Giảm `finger_angle_tolerance` xuống 20-25 độ
- Tăng `min_detection_confidence` lên 0.7-0.8

### Cân bằng hiệu suất và độ chính xác
- `finger_extension_threshold` = 7-8 pixel
- `finger_angle_tolerance` = 30 độ
- `min_detection_confidence` = 0.65
- `min_tracking_confidence` = 0.55
