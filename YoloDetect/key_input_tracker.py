import time

class KeyInputTracker:
    def __init__(self, cooldown_time=0.5, confidence_threshold=0.3, consistency_required=3):
        """
        Khởi tạo bộ theo dõi phím bấm

        Tham số:
        - cooldown_time: Thời gian (giây) phải chờ trước khi coi một phím có thể được bấm lại
        - confidence_threshold: Ngưỡng tin cậy tối thiểu để coi một phím là được bấm
        - consistency_required: Số frame liên tiếp phải phát hiện cùng một phím để xác nhận việc bấm
        """
        self.input_text = ""  # Chuỗi đã nhập
        self.last_key = None  # Phím được bấm cuối cùng
        self.last_press_time = {}  # Thời gian bấm cuối cùng của mỗi phím
        self.cooldown_time = cooldown_time  # Thời gian chờ giữa các lần bấm
        self.confidence_threshold = confidence_threshold  # Ngưỡng tin cậy

        # Theo dõi số frame liên tiếp mà một phím được phát hiện
        self.current_key_streak = 0
        self.current_key = None
        self.consistency_required = consistency_required

        # Lưu trạng thái hiện tại của phím
        self.current_pressing = False

        # Ánh xạ đặc biệt cho các phím không phải ký tự thông thường
        self.special_key_mapping = {
            'backspace': lambda: self._handle_backspace(),
            'space': lambda: self._append_text(' '),
            'enter': lambda: self._append_text('\n'),
            'shift-left': lambda: None,  # Sẽ triển khai sau nếu cần
            'shift-right': lambda: None,  # Sẽ triển khai sau nếu cần
            'caps': lambda: None,  # Sẽ triển khai sau nếu cần
            'tab': lambda: self._append_text('\t')
        }

    def update(self, pressed_key, confidence, is_pressing):
        """
        Cập nhật trạng thái dựa trên phím đang được bấm

        Tham số:
        - pressed_key: Phím được phát hiện là đang bấm (None nếu không có)
        - confidence: Độ tin cậy của việc phát hiện
        - is_pressing: Có đang trong trạng thái bấm hay không

        Trả về:
        - True nếu có sự thay đổi trong chuỗi văn bản, False nếu không
        """
        text_changed = False

        # Kiểm tra nếu có phím được bấm với độ tin cậy đủ cao
        if pressed_key and confidence >= self.confidence_threshold and is_pressing:
            # Kiểm tra nếu đây là phím mới hoặc phím đã hết thời gian chờ
            current_time = time.time()
            last_time = self.last_press_time.get(pressed_key, 0)

            # Nếu đây là phím đang được theo dõi, tăng số đếm
            if pressed_key == self.current_key:
                self.current_key_streak += 1
            else:
                # Nếu là phím mới, bắt đầu đếm mới
                self.current_key = pressed_key
                self.current_key_streak = 1

            # Nếu phím đã được phát hiện đủ số frame liên tiếp và đã hết thời gian chờ
            if (self.current_key_streak >= self.consistency_required and
                current_time - last_time > self.cooldown_time):

                # Xử lý phím đặc biệt
                if pressed_key in self.special_key_mapping:
                    action = self.special_key_mapping[pressed_key]
                    action()
                else:
                    # Xử lý các phím thông thường (ký tự)
                    self._append_text(pressed_key)

                # Cập nhật thời gian bấm cuối cùng
                self.last_press_time[pressed_key] = current_time
                self.last_key = pressed_key
                text_changed = True

                # Reset streak sau khi đã xử lý
                self.current_key_streak = 0

        # Nếu không có phím nào được bấm, reset streak
        elif not is_pressing:
            self.current_key_streak = 0
            self.current_key = None

        # Cập nhật trạng thái bấm hiện tại
        self.current_pressing = is_pressing

        return text_changed

    def _append_text(self, char):
        """Thêm ký tự vào chuỗi văn bản"""
        self.input_text += char

    def _handle_backspace(self):
        """Xử lý phím backspace - xóa ký tự cuối cùng"""
        if self.input_text:
            self.input_text = self.input_text[:-1]

    def get_text(self):
        """Lấy chuỗi văn bản hiện tại"""
        return self.input_text

    def clear_text(self):
        """Xóa toàn bộ chuỗi văn bản"""
        self.input_text = ""
        return True
