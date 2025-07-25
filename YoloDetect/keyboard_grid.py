import numpy as np
import cv2


class KeyboardGrid:
    def __init__(self):
        # Định nghĩa layout chuẩn của bàn phím
        self.keyboard_layout = [
            # Hàng 1
            ['esc', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12'],
            # Hàng 2
            ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'backspace'],
            # Hàng 3
            ['tab', 'q', 'w', 'e', 'r', 't', 'z', 'u', 'i', 'o', 'p', '[', ']', '\\'],
            # Hàng 4
            ['caps', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'", 'enter'],
            # Hàng 5
            ['shift-left', 'y', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 'shift-right'],
            # Hàng 6
            ['strg-left', 'win', 'alt-left', 'space', 'altgr-right', 'menu', 'strg-right']
        ]

        # Khởi tạo dictionary lưu trữ thông tin về các phím
        self.key_info = {}
        self._initialize_key_info()

    def _initialize_key_info(self):
        """Khởi tạo thông tin cho tất cả các phím"""
        for row in self.keyboard_layout:
            for key in row:
                self.key_info[key] = {
                    'detected': False,
                    'bbox': None,
                    'confidence': 0.0,
                    'interpolated': False
                }

    def update_detected_keys(self, results):
        """Cập nhật thông tin về các phím được phát hiện từ model"""
        # Reset tất cả các phím về trạng thái chưa phát hiện
        self._initialize_key_info()

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls = int(box.cls[0])
                name = results[0].names[cls]
                conf = float(box.conf[0])

                if conf > 0.5:  # Chỉ lấy các phím có độ tin cậy > 0.5
                    bbox = box.xyxy[0].cpu().numpy()

                    # Cập nhật thông tin cho phím được phát hiện
                    if name in self.key_info:
                        self.key_info[name].update({
                            'detected': True,
                            'bbox': bbox,
                            'confidence': conf,
                            'interpolated': False
                        })

        # Sau khi cập nhật các phím được phát hiện, thực hiện nội suy
        self._interpolate_missing_keys()

    def _interpolate_missing_keys(self):
        """Nội suy vị trí của các phím chưa được phát hiện"""
        detected_keys = [k for k, v in self.key_info.items() if v['detected']]

        if len(detected_keys) < 2:
            return  # Cần ít nhất 2 phím để nội suy

        # Tìm vị trí của các phím đã phát hiện trong layout
        for row_idx, row in enumerate(self.keyboard_layout):
            for col_idx, key in enumerate(row):
                if key in detected_keys:
                    self._interpolate_neighbors(row_idx, col_idx)

    def _interpolate_neighbors(self, row_idx, col_idx):
        """Nội suy vị trí cho các phím lân cận của một phím đã phát hiện"""
        current_key = self.keyboard_layout[row_idx][col_idx]
        current_info = self.key_info[current_key]

        if not current_info['detected']:
            return

        bbox = current_info['bbox']
        key_width = bbox[2] - bbox[0]
        key_height = bbox[3] - bbox[1]

        # Nội suy theo chiều ngang
        for dcol in [-1, 1]:
            if 0 <= col_idx + dcol < len(self.keyboard_layout[row_idx]):
                neighbor_key = self.keyboard_layout[row_idx][col_idx + dcol]
                if not self.key_info[neighbor_key]['detected']:
                    new_bbox = np.array([
                        bbox[0] + dcol * key_width,
                        bbox[1],
                        bbox[2] + dcol * key_width,
                        bbox[3]
                    ])
                    self.key_info[neighbor_key].update({
                        'bbox': new_bbox,
                        'detected': False,
                        'interpolated': True,
                        'confidence': current_info['confidence'] * 0.8
                    })

    def draw_keyboard_visualization(self, frame):
        """Vẽ visualization của bàn phím lên frame"""
        for key, info in self.key_info.items():
            if info['bbox'] is not None:
                bbox = info['bbox'].astype(int)
                if info['detected'] or info['interpolated']:
                    # Chỉ vẽ tên phím
                    color = (0, 255, 0) if info['detected'] else (0, 255, 255)
                    cv2.putText(frame, key, (bbox[0], bbox[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def get_key_at_position(self, x, y):
        """
        Xác định phím tại vị trí (x, y)
        Args:
            x: tọa độ x của điểm cần kiểm tra
            y: tọa độ y của điểm cần kiểm tra
        Returns:
            Tên của phím nếu tìm thấy, None nếu không có phím nào tại vị trí đó
        """
        for key, info in self.key_info.items():
            if info['bbox'] is not None:
                bbox = info['bbox']
                if (bbox[0] <= x <= bbox[2]) and (bbox[1] <= y <= bbox[3]):
                    return key
        return None
