import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import json
import os
from datetime import datetime
from key_classifier import KeyClassifier  # Thêm import
import threading
import time
import queue
import torch

class VideoLabeler:
    def __init__(self, model_path='best_v2.pt'):
        """Initialize the Video Labeler with YOLO model."""
        # Check GPU availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model with GPU support
        self.model = YOLO(model_path)
        if self.device == 'cuda':
            self.model.to('cuda')
        
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.current_frame_idx = 0
        self.total_frames = 0
        
        # Thread safety
        self.frame_lock = threading.Lock()
        self.gui_ready = threading.Event()
        self.video_thread = None
        self.video_queue = queue.Queue(maxsize=2)  # Giảm queue size hơn nữa
        self.stop_video_thread = threading.Event()
        
        # Key tracking - Chuyển sang dạng dictionary để theo dõi nhiều phím
        self.target_keys = ['i', 'o', 't', 'e', 'u']  # Danh sách các phím cần theo dõi
        self.active_key = 't'  # Phím đang được chọn để labeling
        self.key_data = {}  # Dictionary lưu thông tin của từng phím
        for key in self.target_keys:
            self.key_data[key] = {
                'bbox': None,
                'expanded_bbox': None,
                'roi': None,
                'labels': {},  # frame_idx: is_pressed (0 or 1)
                'color': self._get_key_color(key)  # Màu riêng cho mỗi phím
            }
        
        # Video display
        self.video_window_name = "Video Display"
        self.display_frame = None
        self.needs_update = False
        self.last_frame_time = 0  # Thêm tracking thời gian
        
        # Performance optimization
        self.frame_skip = 1  # Process every Nth frame
        self.last_processed_frame = -1
        
        # Classifiers
        self.available_models = KeyClassifier.get_available_models()
        self.classifiers = {}
        self.current_model = None
        self.has_classifier = len(self.available_models) > 0
        
        if self.has_classifier:
            try:
                self.current_model = self.available_models[0]
                self.classifiers[self.current_model] = KeyClassifier(model_type=self.current_model)
            except Exception as e:
                print(f"Không thể khởi tạo classifier: {str(e)}")
                self.has_classifier = False
        
        # Output
        self.output_dir = "labeled_data"
        self.image_dir = os.path.join(self.output_dir, "key_images")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Prediction settings
        self.auto_predict = False
        self.prediction_results = {}  # Lưu kết quả dự đoán cho từng phím
        
        # Playback control - Di chuyển lên trước setup_gui
        self.is_playing = False
        self.play_speed = 20  # Tăng FPS lên 20
        
        # GUI setup
        self.setup_gui()
        
        # Start GUI thread
        self.gui_ready.set()

    def _get_key_color(self, key):
        """Tạo màu riêng cho mỗi phím."""
        colors = {
            'i': (255, 0, 0),    # Đỏ
            'o': (0, 255, 0),    # Xanh lá
            't': (0, 0, 255),    # Xanh dương
            'y': (255, 255, 0),  # Vàng
            'u': (255, 0, 255)   # Tím
        }
        return colors.get(key, (128, 128, 128))  # Màu xám cho các phím khác

    def setup_gui(self):
        """Setup the GUI interface."""
        self.root = tk.Tk()
        self.root.title("Video Key Labeler")
        self.root.geometry("800x600")
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Video Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Select Video", command=self.select_video).pack(side=tk.LEFT, padx=(0, 10))
        self.video_label = ttk.Label(file_frame, text="No video selected")
        self.video_label.pack(side=tk.LEFT)
        
        # Key selection frame
        key_frame = ttk.LabelFrame(main_frame, text="Key Selection", padding=10)
        key_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Tạo radio buttons cho từng phím
        self.active_key_var = tk.StringVar(value=self.active_key)
        for key in self.target_keys:
            ttk.Radiobutton(key_frame, text=f"Key {key.upper()}", 
                          variable=self.active_key_var, value=key,
                          command=self.on_key_selected).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(key_frame, text="Detect All Keys", 
                  command=self.detect_all_keys).pack(side=tk.LEFT, padx=(20, 0))
        
        # Key status frame
        self.key_status_frame = ttk.LabelFrame(main_frame, text="Key Status", padding=10)
        self.key_status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Tạo labels cho từng phím
        self.key_status_labels = {}
        for key in self.target_keys:
            label_frame = ttk.Frame(self.key_status_frame)
            label_frame.pack(fill=tk.X, pady=2)
            ttk.Label(label_frame, text=f"Key {key.upper()}:").pack(side=tk.LEFT, padx=(0, 10))
            self.key_status_labels[key] = ttk.Label(label_frame, text="Not detected")
            self.key_status_labels[key].pack(side=tk.LEFT)
        
        # Video control frame
        control_frame = ttk.LabelFrame(main_frame, text="Video Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Frame navigation
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(nav_frame, text="<<", command=self.prev_frame).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Play/Pause", command=self.toggle_play).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text=">>", command=self.next_frame).pack(side=tk.LEFT, padx=(0, 5))
        
        # Video window controls
        window_frame = ttk.Frame(nav_frame)
        window_frame.pack(side=tk.RIGHT)
        
        ttk.Button(window_frame, text="Restart Window", 
                  command=self.restart_video_window).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Frame info
        self.frame_info = ttk.Label(nav_frame, text="Frame: 0 / 0")
        self.frame_info.pack(side=tk.LEFT, padx=(20, 0))
        
        # Performance settings
        perf_frame = ttk.Frame(control_frame)
        perf_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(perf_frame, text="FPS:").pack(side=tk.LEFT, padx=(0, 5))
        self.fps_var = tk.IntVar(value=self.play_speed)
        fps_scale = ttk.Scale(perf_frame, from_=5, to=30, orient=tk.HORIZONTAL, 
                             variable=self.fps_var, command=self.update_fps)
        fps_scale.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        ttk.Label(perf_frame, text="Device:").pack(side=tk.LEFT, padx=(10, 5))
        self.device_label = ttk.Label(perf_frame, text=self.device.upper())
        self.device_label.pack(side=tk.LEFT)
        
        # Video info frame
        info_frame = ttk.LabelFrame(main_frame, text="Video Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Video properties
        self.video_info_text = tk.StringVar(value="No video loaded")
        self.video_info_label = ttk.Label(info_frame, textvariable=self.video_info_text, 
                                        font=("Courier", 9))
        self.video_info_label.pack(fill=tk.X)
        
        # Optimization controls
        opt_frame = ttk.Frame(info_frame)
        opt_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Frame skip controls
        skip_control_frame = ttk.Frame(opt_frame)
        skip_control_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.enable_frame_skip_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(skip_control_frame, text="Enable Frame Skip", 
                       variable=self.enable_frame_skip_var,
                       command=self.toggle_frame_skip).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(skip_control_frame, text="Frame Skip:").pack(side=tk.LEFT, padx=(0, 5))
        self.frame_skip_var = tk.IntVar(value=self.frame_skip)
        self.skip_scale = ttk.Scale(skip_control_frame, from_=1, to=5, orient=tk.HORIZONTAL, 
                                   variable=self.frame_skip_var, command=self.update_frame_skip)
        self.skip_scale.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        # Full resolution controls
        res_control_frame = ttk.Frame(opt_frame)
        res_control_frame.pack(fill=tk.X)
        
        self.full_resolution_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(res_control_frame, text="Full Resolution Processing", 
                       variable=self.full_resolution_var,
                       command=self.toggle_full_resolution).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(opt_frame, text="Re-optimize", command=self.re_optimize_video).pack(side=tk.RIGHT)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                     variable=self.progress_var, command=self.seek_frame)
        self.progress_bar.pack(fill=tk.X, pady=(10, 0))
        
        # Labeling frame
        label_frame = ttk.LabelFrame(main_frame, text="Labeling", padding=10)
        label_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Current frame label status
        self.current_label_var = tk.StringVar(value="Not Labeled")
        current_label_frame = ttk.Frame(label_frame)
        current_label_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(current_label_frame, text="Current Frame:").pack(side=tk.LEFT)
        self.current_label_display = ttk.Label(current_label_frame, textvariable=self.current_label_var, 
                                               background="lightgray")
        self.current_label_display.pack(side=tk.LEFT, padx=(10, 0))
        
        # Labeling buttons
        button_frame = ttk.Frame(label_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Key NOT Pressed (0)", 
                  command=lambda: self.label_frame(0)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Key PRESSED (1)", 
                  command=lambda: self.label_frame(1)).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Remove Label", 
                  command=self.remove_label).pack(side=tk.LEFT, padx=(0, 10))
        
        # Statistics
        stats_frame = ttk.Frame(label_frame)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.stats_label = ttk.Label(stats_frame, text="Labels: 0 total, 0 pressed, 0 not pressed")
        self.stats_label.pack(side=tk.LEFT)
        
        # Export frame
        export_frame = ttk.LabelFrame(main_frame, text="Export", padding=10)
        export_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(export_frame, text="Save Labels", command=self.save_labels).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(export_frame, text="Export for CNN+LSTM", command=self.export_for_training).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(export_frame, text="Load Labels", command=self.load_labels).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(export_frame, text="Convert Old Format", command=self.convert_old_label_file).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(export_frame, text="Export Current Key Image", command=self.export_current_key_image).pack(side=tk.LEFT, padx=(0, 10))
        
        # Prediction frame (new)
        if self.has_classifier:
            predict_frame = ttk.LabelFrame(main_frame, text="Prediction", padding=10)
            predict_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Model selection
            model_frame = ttk.Frame(predict_frame)
            model_frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(model_frame, text="Select Model:").pack(side=tk.LEFT, padx=(0, 10))
            self.model_var = tk.StringVar(value=self.current_model)
            self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                          values=self.available_models, state="readonly")
            self.model_combo.pack(side=tk.LEFT, padx=(0, 10))
            self.model_combo.bind('<<ComboboxSelected>>', self.on_model_selected)
            
            # Auto predict checkbox
            self.auto_predict_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(model_frame, text="Auto Predict", 
                          variable=self.auto_predict_var,
                          command=self.toggle_auto_predict).pack(side=tk.LEFT, padx=(10, 0))
            
            # Model info
            self.model_info_label = ttk.Label(model_frame, text="")
            self.model_info_label.pack(side=tk.LEFT, padx=(10, 0))
            self.update_model_info()
            
            # Prediction controls
            control_frame = ttk.Frame(predict_frame)
            control_frame.pack(fill=tk.X)
            
            ttk.Button(control_frame, text="Predict Current Frame", 
                      command=self.predict_current_frame).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Button(control_frame, text="Predict All Models", 
                      command=self.predict_all_models).pack(side=tk.LEFT, padx=(0, 10))
            
            # Prediction results
            self.prediction_label = ttk.Label(control_frame, text="No prediction yet")
            self.prediction_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Key bindings
        self.root.bind('<Key-0>', lambda e: self.label_frame(0))
        self.root.bind('<Key-1>', lambda e: self.label_frame(1))
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('<space>', lambda e: self.toggle_play())
        self.root.bind('<Delete>', lambda e: self.remove_label())
        self.root.bind('<Escape>', lambda e: self.cleanup())
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)
        self.root.focus_set()

    def cleanup(self, event=None):
        """Cleanup resources before closing."""
        self.is_playing = False  # Stop video playback
        
        # Stop video thread
        self.stop_video_display()
        
        if self.cap:
            with self.frame_lock:
                self.cap.release()
        
        # Send stop signal to video queue
        try:
            self.video_queue.put(None, block=False)
        except:
            pass
            
        self.root.quit()
        self.root.destroy()

    def select_video(self):
        """Select video file."""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        
        if file_path:
            self.video_path = file_path
            self.video_label.config(text=f"Video: {os.path.basename(file_path)}")
            self.load_video()
            
    def load_video(self):
        """Load the selected video."""
        if not self.video_path:
            return
            
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress_bar.config(to=self.total_frames-1)
        
        # Detect video properties and optimize
        self._optimize_video_loading()
        
        # Update video info display
        self._update_video_info()
        
        # Start video thread
        self.start_video_thread()
        
        # Load first frame
        self.current_frame_idx = 0
        self.update_frame()
        
        messagebox.showinfo("Success", f"Video loaded successfully!\nTotal frames: {self.total_frames}")
        
    def _optimize_video_loading(self):
        """Optimize video loading based on encoding and properties."""
        if self.cap is None:
            return
            
        # Get video properties
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        print(f"Video properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Codec: {codec}")
        
        # Check if full resolution mode is enabled
        if self.full_resolution_var.get():
            print("  Full resolution mode - skipping optimizations")
            self.frame_skip = 1
            self.play_speed = 30  # Max FPS for full resolution
            return
        
        # Optimize based on codec
        if codec in ['H264', 'AVC1', 'XVID']:
            # Hardware acceleration for common codecs
            self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            print("  Using hardware acceleration")
        elif codec in ['MJPG', 'MJPEG']:
            # MJPEG is usually fast to decode
            self.frame_skip = 1
            print("  MJPEG detected - using standard processing")
        elif codec in ['HEVC', 'H265']:
            # HEVC is more complex, may need optimization
            self.frame_skip = 2  # Process every 2nd frame
            print("  HEVC detected - using frame skipping")
        else:
            print(f"  Unknown codec: {codec} - using default settings")
            
        # Optimize based on resolution
        if width * height > 1920 * 1080:  # 1080p or higher
            self.frame_skip = max(self.frame_skip, 2)
            print("  High resolution detected - using frame skipping")
            
        # Optimize based on FPS
        if fps > 30:
            self.play_speed = min(self.play_speed, 15)  # Cap FPS for high-speed videos
            print("  High FPS detected - capping playback speed")
            
        # Set buffer size for faster seeking
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"  Final settings: FPS={self.play_speed}, Frame skip={self.frame_skip}")

    def detect_all_keys(self):
        """Detect tất cả các phím trong frame hiện tại."""
        if self.cap is None:
            messagebox.showerror("Error", "Please select a video first!")
            return
            
        # Go to current frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        
        if not ret:
            messagebox.showerror("Error", "Could not read current frame!")
            return
            
        # Run YOLO detection with GPU optimization
        try:
            with torch.no_grad():  # Disable gradient computation for inference
                results = self.model(frame, verbose=False)  # Disable verbose output
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            return
        
        # Reset detection results
        for key in self.target_keys:
            self.key_data[key]['bbox'] = None
            self.key_data[key]['expanded_bbox'] = None
        
        # Find all keys
        keys_found = []
        for result in results:
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    key_name = result.names[int(box.cls)]
                    if key_name in self.target_keys:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = (int(x1), int(y1), int(x2), int(y2))
                        
                        # Expand bounding box by 20%
                        width = x2 - x1
                        height = y2 - y1
                        expand_x = width * 0.2
                        expand_y = height * 0.2
                        
                        exp_x1 = max(0, int(x1 - expand_x))
                        exp_y1 = max(0, int(y1 - expand_y))
                        exp_x2 = min(frame.shape[1], int(x2 + expand_x))
                        exp_y2 = min(frame.shape[0], int(y2 + expand_y))
                        
                        expanded_bbox = (exp_x1, exp_y1, exp_x2, exp_y2)
                        
                        # Save detection results
                        self.key_data[key_name]['bbox'] = bbox
                        self.key_data[key_name]['expanded_bbox'] = expanded_bbox
                        keys_found.append(key_name)
                        
                        # Update status label
                        self.key_status_labels[key_name].config(
                            text=f"Detected! Bbox: {expanded_bbox}")
        
        if keys_found:
            messagebox.showinfo("Success", 
                f"Detected keys: {', '.join(keys_found)}\n"
                f"Total keys found: {len(keys_found)}")
        else:
            messagebox.showwarning("Warning", "No keys found!")

    def on_key_selected(self):
        """Xử lý khi người dùng chọn phím khác."""
        self.active_key = self.active_key_var.get()
        self.root.title(f"Video Labeler - Key '{self.active_key.upper()}'")
        self.update_frame()  # Cập nhật frame để hiển thị thông tin phím mới

    def detect_key_t(self):
        """Detect key T in the first frame and expand bounding box by 10%."""
        if self.cap is None:
            messagebox.showerror("Error", "Please select a video first!")
            return
            
        # Go to first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        
        if not ret:
            messagebox.showerror("Error", "Could not read first frame!")
            return
            
        # Run YOLO detection
        results = self.model(frame)
        
        # Find key T
        key_t_found = False
        for result in results:
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    if result.names[int(box.cls)] == 'i':
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        self.key_bbox = (int(x1), int(y1), int(x2), int(y2))
                        
                        # Expand bounding box by 20%
                        width = x2 - x1
                        height = y2 - y1
                        expand_x = width * 0.2
                        expand_y = height * 0.2
                        
                        exp_x1 = max(0, int(x1 - expand_x))
                        exp_y1 = max(0, int(y1 - expand_y))
                        exp_x2 = min(frame.shape[1], int(x2 + expand_x))
                        exp_y2 = min(frame.shape[0], int(y2 + expand_y))
                        
                        self.expanded_bbox = (exp_x1, exp_y1, exp_x2, exp_y2)
                        key_t_found = True
                        break
                        
            if key_t_found:
                break
                
        if key_t_found:
            self.key_status_label.config(text=f"Key T detected! Bbox: {self.expanded_bbox}")
            messagebox.showinfo("Success", 
                f"Key T detected and expanded by 10%!\n"
                f"Original: {self.key_bbox}\n"
                f"Expanded: {self.expanded_bbox}")
        else:
            messagebox.showerror("Error", "Key T not found in the first frame!")
            
    def toggle_auto_predict(self):
        """Bật/tắt chế độ dự đoán tự động."""
        self.auto_predict = self.auto_predict_var.get()
        if self.auto_predict and not self.has_classifier:
            messagebox.showerror("Error", "Không có model nào khả dụng!")
            self.auto_predict_var.set(False)
            self.auto_predict = False
            return
            
        if self.auto_predict and self.expanded_bbox is None:
            messagebox.showerror("Error", "Vui lòng detect phím T trước!")
            self.auto_predict_var.set(False)
            self.auto_predict = False
            return

    def predict_frame(self):
        """Dự đoán frame hiện tại cho tất cả các phím đã detect."""
        if not self.has_classifier:
            return None
            
        predictions = {}
        for key in self.target_keys:
            if self.key_data[key]['expanded_bbox'] is not None:
                try:
                    # Extract và tiền xử lý vùng phím
                    x1, y1, x2, y2 = self.key_data[key]['expanded_bbox']
                    key_region = self.current_frame[y1:y2, x1:x2]
                    key_region = cv2.resize(key_region, (64, 64))
                    
                    # Dự đoán
                    classifier = self.classifiers[self.current_model]
                    result = classifier.predict(key_region)
                    predictions[key] = result
                except Exception as e:
                    print(f"Lỗi khi dự đoán phím {key}: {str(e)}")
                    
        return predictions

    def update_frame(self):
        """Update the current frame display."""
        if self.cap is None:
            return
            
        # Throttle frame updates
        current_time = time.time()
        if current_time - self.last_frame_time < 1.0 / self.play_speed:
            return
        self.last_frame_time = current_time
            
        with self.frame_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            
            if ret:
                self.current_frame = frame.copy()
                display_frame = frame.copy()
                
                # Dự đoán nếu đang ở chế độ auto predict và frame mới
                if self.auto_predict and self.current_frame_idx != self.last_processed_frame:
                    try:
                        with torch.no_grad():
                            self.prediction_results = self.predict_frame()
                        self.last_processed_frame = self.current_frame_idx
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        self.prediction_results = {}
                
                # Draw bounding boxes for all detected keys
                for key in self.target_keys:
                    if self.key_data[key]['expanded_bbox'] is not None:
                        x1, y1, x2, y2 = self.key_data[key]['expanded_bbox']
                        base_color = self.key_data[key]['color']
                        
                        # Vẽ bbox với màu phụ thuộc vào kết quả dự đoán
                        if self.auto_predict and key in self.prediction_results:
                            result = self.prediction_results[key]
                            confidence = result['confidence']
                            predicted_class = result['class_id']
                            
                            # Chỉ hiển thị màu sáng khi confidence > 50% và dự đoán là pressed
                            if confidence > 50 and predicted_class == 1:
                                color = base_color  # Sử dụng màu gốc với độ sáng cao
                            else:
                                # Giảm độ sáng của màu gốc
                                color = tuple(int(c * 0.5) for c in base_color)
                            
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display_frame, 
                                      f"Key {key.upper()}: {result['class_name']} ({confidence:.1f}%)", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        else:
                            # Nếu không có dự đoán, sử dụng màu gốc với độ mờ
                            color = tuple(int(c * 0.7) for c in base_color)
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display_frame, f"Key {key.upper()}", (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Show current label status for active key
                label_text = "Not Labeled"
                label_color = (128, 128, 128)
                if self.current_frame_idx in self.key_data[self.active_key]['labels']:
                    if self.key_data[self.active_key]['labels'][self.current_frame_idx] == 1:
                        label_text = f"Key {self.active_key.upper()} PRESSED"
                        label_color = (0, 255, 0)
                    else:
                        label_text = f"Key {self.active_key.upper()} NOT PRESSED"
                        label_color = (0, 0, 255)
                
                cv2.putText(display_frame, label_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
                
                # Clear queue if it's getting full
                while not self.video_queue.empty():
                    try:
                        self.video_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Send frame to video thread via queue
                try:
                    self.video_queue.put((display_frame, None), block=False)
                except queue.Full:
                    pass  # Skip frame if queue is full
                
                # Update UI in main thread
                self.root.after(0, self._update_ui)

    def _update_ui(self):
        """Update UI elements in main thread."""
        # Update frame info
        self.frame_info.config(text=f"Frame: {self.current_frame_idx} / {self.total_frames-1}")
        self.progress_var.set(self.current_frame_idx)
        self.update_current_label_display()
        self.update_stats()
        
        # Update prediction labels if auto predict is on
        if self.auto_predict and self.prediction_results:
            for key, result in self.prediction_results.items():
                confidence = result['confidence']
                confidence_status = "High" if confidence > 90 else "Low"
                prediction_text = (
                    f"Key {key.upper()}: {result['class_name']} "
                    f"({confidence:.1f}% - {confidence_status} Confidence)"
                )
                if hasattr(self, f'prediction_label_{key}'):
                    getattr(self, f'prediction_label_{key}').config(text=prediction_text)

    def _switch_key(self, key_char):
        """Switch to different key in main thread."""
        if key_char in self.target_keys:
            self.active_key_var.set(key_char)
            self.on_key_selected()
            
    def prev_frame(self):
        """Go to previous frame."""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.update_frame()
            
    def next_frame(self):
        """Go to next frame."""
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.update_frame()
            
    def update_fps(self, value):
        """Update FPS setting."""
        self.play_speed = int(float(value))
        
    def update_frame_skip(self, value):
        """Update frame skip setting."""
        if self.enable_frame_skip_var.get():
            self.frame_skip = int(float(value))
        else:
            self.frame_skip = 1  # No frame skip
            
    def toggle_frame_skip(self):
        """Toggle frame skip on/off."""
        if self.enable_frame_skip_var.get():
            self.frame_skip = self.frame_skip_var.get()
            self.skip_scale.config(state="normal")
        else:
            self.frame_skip = 1  # No frame skip
            self.skip_scale.config(state="disabled")
        self._update_video_info()
        
    def toggle_full_resolution(self):
        """Toggle full resolution processing."""
        if self.full_resolution_var.get():
            # Disable all optimizations for full resolution
            self.frame_skip = 1
            self.frame_skip_var.set(1)
            self.enable_frame_skip_var.set(False)
            self.skip_scale.config(state="disabled")
            print("Full resolution mode enabled - all optimizations disabled")
        else:
            # Re-enable optimizations
            self.enable_frame_skip_var.set(True)
            self.skip_scale.config(state="normal")
            self._optimize_video_loading()
            print("Optimization mode enabled")
        self._update_video_info()
        
    def re_optimize_video(self):
        """Re-optimize video loading."""
        if self.cap is not None:
            if not self.full_resolution_var.get():
                self._optimize_video_loading()
            self._update_video_info()
            
    def _update_video_info(self):
        """Update video information display."""
        if self.cap is None:
            self.video_info_text.set("No video loaded")
            return
            
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        info = f"Resolution: {width}x{height} | FPS: {fps:.1f} | Codec: {codec}\n"
        info += f"Playback: {self.play_speed} FPS | Frame Skip: {self.frame_skip} | Device: {self.device.upper()}"
        
        self.video_info_text.set(info)
        
    def seek_frame(self, value):
        """Seek to specific frame."""
        self.current_frame_idx = int(float(value))
        self.update_frame()
        
    def toggle_play(self):
        """Toggle play/pause."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_video()
            
    def play_video(self):
        """Play video continuously."""
        if self.is_playing and self.current_frame_idx < self.total_frames - 1:
            self.next_frame()
            self.root.after(int(1000/self.play_speed), self.play_video)
        else:
            self.is_playing = False
            
    def label_frame(self, label):
        """Label current frame for active key."""
        key_data = self.key_data[self.active_key]
        if key_data['expanded_bbox'] is None:
            messagebox.showerror("Error", f"Please detect key {self.active_key.upper()} first!")
            return
            
        key_data['labels'][self.current_frame_idx] = label
        self.update_current_label_display()
        self.update_stats()
        
        # Auto advance to next frame
        if self.current_frame_idx < self.total_frames - 1:
            self.next_frame()
            
    def remove_label(self):
        """Remove label from current frame for active key."""
        if self.current_frame_idx in self.key_data[self.active_key]['labels']:
            del self.key_data[self.active_key]['labels'][self.current_frame_idx]
            self.update_current_label_display()
            self.update_stats()
            
    def update_current_label_display(self):
        """Update current frame label display for active key."""
        if self.current_frame_idx in self.key_data[self.active_key]['labels']:
            label_val = self.key_data[self.active_key]['labels'][self.current_frame_idx]
            if label_val == 1:
                self.current_label_var.set(f"Key {self.active_key.upper()} PRESSED")
                self.current_label_display.config(background="lightgreen")
            else:
                self.current_label_var.set(f"Key {self.active_key.upper()} NOT PRESSED")
                self.current_label_display.config(background="lightcoral")
        else:
            self.current_label_var.set("Not Labeled")
            self.current_label_display.config(background="lightgray")
            
    def update_stats(self):
        """Update labeling statistics for active key."""
        labels = self.key_data[self.active_key]['labels']
        total = len(labels)
        pressed = sum(1 for label in labels.values() if label == 1)
        not_pressed = total - pressed
        
        self.stats_label.config(
            text=f"Key {self.active_key.upper()} - Labels: {total} total, {pressed} pressed, {not_pressed} not pressed")
        
    def save_labels(self):
        """Save labels to JSON file."""
        if not any(len(key_data['labels']) > 0 for key_data in self.key_data.values()):
            messagebox.showwarning("Warning", "No labels to save!")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"labels_all_keys_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "keys": {}
        }
        
        for key in self.target_keys:
            key_data = self.key_data[key]
            if len(key_data['labels']) > 0:
                data["keys"][key] = {
                    "bbox": key_data['bbox'],
                    "expanded_bbox": key_data['expanded_bbox'],
                    "labels": key_data['labels']
                }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        messagebox.showinfo("Success", f"Labels saved to {filepath}")
        
    def load_labels(self):
        """Load labels from JSON file."""
        filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
        
        file_path = filedialog.askopenfilename(
            title="Select Label File",
            filetypes=filetypes,
            initialdir=self.output_dir
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Check if it's old format (single key) or new format (multiple keys)
                if "keys" in data:
                    # New format - multiple keys
                    for key, key_data in data["keys"].items():
                        if key in self.key_data:
                            self.key_data[key]['bbox'] = key_data.get("bbox")
                            self.key_data[key]['expanded_bbox'] = key_data.get("expanded_bbox")
                            self.key_data[key]['labels'] = {
                                int(k): v for k, v in key_data.get("labels", {}).items()
                            }
                            
                            if self.key_data[key]['expanded_bbox']:
                                self.key_status_labels[key].config(
                                    text=f"Loaded! Bbox: {self.key_data[key]['expanded_bbox']}")
                    
                elif "target_key" in data:
                    # Old format - single key
                    target_key = data["target_key"]
                    if target_key in self.key_data:
                        self.key_data[target_key]['bbox'] = data.get("key_bbox")
                        self.key_data[target_key]['expanded_bbox'] = data.get("expanded_bbox")
                        self.key_data[target_key]['labels'] = {
                            int(k): v for k, v in data.get("labels", {}).items()
                        }
                        
                        if self.key_data[target_key]['expanded_bbox']:
                            self.key_status_labels[target_key].config(
                                text=f"Loaded! Bbox: {self.key_data[target_key]['expanded_bbox']}")
                            
                        # Set active key to the loaded key
                        self.active_key = target_key
                        self.active_key_var.set(target_key)
                        
                        messagebox.showinfo("Backward Compatibility", 
                            f"Loaded old format file for key '{target_key.upper()}'.\n"
                            f"Active key set to '{target_key.upper()}'.")
                else:
                    messagebox.showerror("Error", "Unknown label file format!")
                    return
                    
                self.update_current_label_display()
                self.update_stats()
                self.update_frame()
                
                messagebox.showinfo("Success", f"Labels loaded from {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading labels: {str(e)}")
                
    def export_for_training(self):
        """Export data in format suitable for CNN+LSTM training."""
        if not any(len(key_data['labels']) > 0 for key_data in self.key_data.values()):
            messagebox.showwarning("Warning", "Need labels and key detection first!")
            return
            
        # Create sequence data
        sequence_data = []
        
        for key in self.target_keys:
            key_data = self.key_data[key]
            for frame_idx in sorted(key_data['labels'].keys()):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                
                if ret:
                    # Extract key region
                    x1, y1, x2, y2 = key_data['expanded_bbox']
                    key_region = frame[y1:y2, x1:x2]
                    
                    # Resize to standard size for CNN
                    key_region = cv2.resize(key_region, (64, 64))
                    
                    # Normalize
                    key_region = key_region.astype(np.float32) / 255.0
                    
                    label = key_data['labels'][frame_idx]
                    
                    sequence_data.append({
                        "frame_idx": frame_idx,
                        "image": key_region.tolist(),
                        "label": label
                    })
                
        # Save training data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_data_all_keys_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        export_data = {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "image_size": [64, 64],
            "sequence_data": sequence_data,
            "timestamp": timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        messagebox.showinfo("Success", 
            f"Training data exported to {filepath}\n"
            f"Total samples: {len(sequence_data)}")
            
    def export_current_key_image(self):
        """Export current key region as image."""
        if self.key_data[self.active_key]['expanded_bbox'] is None or self.current_frame is None:
            messagebox.showerror("Error", "Please detect key T and load a frame first!")
            return
            
        # Extract key region
        x1, y1, x2, y2 = self.key_data[self.active_key]['expanded_bbox']
        key_region = self.current_frame[y1:y2, x1:x2]
        
        # Resize to standard size
        key_region = cv2.resize(key_region, (64, 64))
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"key_T_frame_{self.current_frame_idx}_{timestamp}.jpg"
        filepath = os.path.join(self.image_dir, filename)
        
        cv2.imwrite(filepath, key_region)
        
        # Get current label if exists
        label_text = "unlabeled"
        if self.current_frame_idx in self.key_data[self.active_key]['labels']:
            label_text = "pressed" if self.key_data[self.active_key]['labels'][self.current_frame_idx] == 1 else "not_pressed"
            
        # Rename file to include label
        new_filename = f"key_T_frame_{self.current_frame_idx}_{label_text}_{timestamp}.jpg"
        new_filepath = os.path.join(self.image_dir, new_filename)
        os.rename(filepath, new_filepath)
        
        messagebox.showinfo("Success", f"Key image saved to {new_filepath}")
        
    def on_model_selected(self, event=None):
        """Xử lý khi người dùng chọn model khác."""
        selected_model = self.model_var.get()
        if selected_model != self.current_model:
            if selected_model not in self.classifiers:
                try:
                    self.classifiers[selected_model] = KeyClassifier(model_type=selected_model)
                except Exception as e:
                    messagebox.showerror("Error", f"Không thể load model {selected_model}: {str(e)}")
                    self.model_var.set(self.current_model)
                    return
                    
            self.current_model = selected_model
            self.update_model_info()
            
    def update_model_info(self):
        """Cập nhật thông tin về model hiện tại."""
        if self.current_model and self.current_model in self.classifiers:
            classifier = self.classifiers[self.current_model]
            accuracy = classifier.get_model_accuracy()
            if accuracy is not None:
                self.model_info_label.config(text=f"Accuracy: {accuracy:.2f}%")
            else:
                self.model_info_label.config(text="")
                
    def predict_current_frame(self):
        """Dự đoán trạng thái phím của frame hiện tại."""
        if not self.has_classifier:
            messagebox.showerror("Error", "Không có model nào khả dụng!")
            return
            
        if self.key_data[self.active_key]['expanded_bbox'] is None or self.current_frame is None:
            messagebox.showerror("Error", "Vui lòng detect phím T và load frame trước!")
            return
            
        # Extract và tiền xử lý vùng phím
        x1, y1, x2, y2 = self.key_data[self.active_key]['expanded_bbox']
        key_region = self.current_frame[y1:y2, x1:x2]
        key_region = cv2.resize(key_region, (64, 64))
        
        # Dự đoán
        try:
            classifier = self.classifiers[self.current_model]
            result = classifier.predict(key_region)
            
            # Hiển thị kết quả
            prediction_text = (
                f"Model: {result['model_type']} "
                f"(Acc: {result['model_accuracy']:.2f}%) | "
                f"Predicted: {result['class_name']} "
                f"({result['confidence']:.2f}%)"
            )
            self.prediction_label.config(text=prediction_text)
            
            # So sánh với nhãn thực tế nếu có
            if self.current_frame_idx in self.key_data[self.active_key]['labels']:
                actual_label = "Key Press" if self.key_data[self.active_key]['labels'][self.current_frame_idx] == 1 else "No Press"
                is_correct = actual_label == result['class_name']
                status = "✓" if is_correct else "✗"
                messagebox.showinfo("Kết quả dự đoán", 
                    f"Model: {result['model_type']}\n"
                    f"Độ chính xác model: {result['model_accuracy']:.2f}%\n"
                    f"Dự đoán: {result['class_name']} ({result['confidence']:.2f}%)\n"
                    f"Thực tế: {actual_label}\n"
                    f"Kết quả: {status}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Lỗi khi dự đoán: {str(e)}")
            
    def predict_all_models(self):
        """Dự đoán sử dụng tất cả các model có sẵn."""
        if not self.has_classifier:
            messagebox.showerror("Error", "Không có model nào khả dụng!")
            return
            
        if self.key_data[self.active_key]['expanded_bbox'] is None or self.current_frame is None:
            messagebox.showerror("Error", "Vui lòng detect phím T và load frame trước!")
            return
            
        # Extract và tiền xử lý vùng phím
        x1, y1, x2, y2 = self.key_data[self.active_key]['expanded_bbox']
        key_region = self.current_frame[y1:y2, x1:x2]
        key_region = cv2.resize(key_region, (64, 64))
        
        # Dự đoán với tất cả model
        results = []
        for model_type in self.available_models:
            try:
                if model_type not in self.classifiers:
                    self.classifiers[model_type] = KeyClassifier(model_type=model_type)
                    
                result = self.classifiers[model_type].predict(key_region)
                results.append(result)
            except Exception as e:
                print(f"Lỗi với model {model_type}: {str(e)}")
                
        # Hiển thị kết quả
        if results:
            # Lấy nhãn thực tế nếu có
            actual_label = None
            if self.current_frame_idx in self.key_data[self.active_key]['labels']:
                actual_label = "Key Press" if self.key_data[self.active_key]['labels'][self.current_frame_idx] == 1 else "No Press"
                
            # Tạo message
            message = "Kết quả dự đoán từ tất cả model:\n\n"
            for result in results:
                status = ""
                if actual_label:
                    is_correct = actual_label == result['class_name']
                    status = " ✓" if is_correct else " ✗"
                    
                message += (
                    f"Model: {result['model_type']}\n"
                    f"Độ chính xác model: {result['model_accuracy']:.2f}%\n"
                    f"Dự đoán: {result['class_name']} ({result['confidence']:.2f}%){status}\n"
                    f"{'=' * 40}\n"
                )
                
            if actual_label:
                message += f"\nNhãn thực tế: {actual_label}"
                
            messagebox.showinfo("Kết quả từ tất cả model", message)
            
    def start_video_thread(self):
        """Start the video display thread."""
        if self.video_thread is None or not self.video_thread.is_alive():
            self.stop_video_thread.clear()
            self.video_thread = threading.Thread(target=self._video_display_loop, daemon=True)
            self.video_thread.start()
            
    def restart_video_window(self):
        """Restart video window if it's not responding."""
        try:
            cv2.destroyAllWindows()
            time.sleep(0.1)  # Small delay
            self.start_video_thread()
            print("Video window restarted")
        except Exception as e:
            print(f"Failed to restart video window: {e}")
            
    def stop_video_display(self):
        """Stop the video display thread."""
        self.stop_video_thread.set()
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
            
    def _video_display_loop(self):
        """Video display loop in separate thread."""
        try:
            cv2.namedWindow(self.video_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.video_window_name, 800, 600)
            
            # Set window properties for better responsiveness
            cv2.setWindowProperty(self.video_window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
            cv2.setWindowProperty(self.video_window_name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
            
            while not self.stop_video_thread.is_set():
                try:
                    # Get frame from queue with timeout
                    frame_data = self.video_queue.get(timeout=0.05)  # Giảm timeout
                    if frame_data is None:  # Stop signal
                        break
                        
                    display_frame, key_events = frame_data
                    
                    # Display frame
                    cv2.imshow(self.video_window_name, display_frame)
                    
                    # Process key events with shorter wait and handle window events
                    key = cv2.waitKey(1) & 0xFF
                    
                    # Handle window close
                    try:
                        if cv2.getWindowProperty(self.video_window_name, cv2.WND_PROP_VISIBLE) < 1:
                            self.root.after(0, self.cleanup)
                            break
                    except:
                        # Window might be closed
                        self.root.after(0, self.cleanup)
                        break
                        
                    if key == 27:  # ESC key
                        self.root.after(0, self.cleanup)
                        break
                    elif key == ord('0'):
                        self.root.after(0, lambda: self.label_frame(0))
                    elif key == ord('1'):
                        self.root.after(0, lambda: self.label_frame(1))
                    elif key == ord(' '):
                        self.root.after(0, self.toggle_play)
                    elif key == 83 or key == ord('d'):  # Right arrow or 'd'
                        self.root.after(0, self.next_frame)
                    elif key == 81 or key == ord('a'):  # Left arrow or 'a'
                        self.root.after(0, self.prev_frame)
                    elif key == ord('r'):
                        self.root.after(0, self.remove_label)
                    elif key in [ord(k) for k in self.target_keys]:
                        self.root.after(0, lambda k=key: self._switch_key(chr(k)))
                        
                except queue.Empty:
                    # Handle window events even when no frame
                    cv2.waitKey(1)
                    continue
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    continue
                    
        except Exception as e:
            print(f"Video window error: {e}")
        finally:
            try:
                cv2.destroyAllWindows()
            except:
                pass

    def run(self):
        """Start the application."""
        self.root.mainloop()
        
        # Cleanup
        if self.cap:
            self.cap.release()

    def convert_old_label_file(self):
        """Convert old format label file to new format."""
        filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
        
        file_path = filedialog.askopenfilename(
            title="Select Old Format Label File",
            filetypes=filetypes,
            initialdir=self.output_dir
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                if "target_key" in data:
                    # Convert old format to new format
                    target_key = data["target_key"]
                    new_data = {
                        "video_path": data.get("video_path"),
                        "total_frames": data.get("total_frames"),
                        "keys": {
                            target_key: {
                                "bbox": data.get("key_bbox"),
                                "expanded_bbox": data.get("expanded_bbox"),
                                "labels": data.get("labels", {})
                            }
                        }
                    }
                    
                    # Save as new format
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_filename = f"converted_labels_{target_key}_{timestamp}.json"
                    new_filepath = os.path.join(self.output_dir, new_filename)
                    
                    with open(new_filepath, 'w') as f:
                        json.dump(new_data, f, indent=2)
                        
                    messagebox.showinfo("Success", 
                        f"Converted old format file to new format!\n"
                        f"Saved as: {new_filename}")
                else:
                    messagebox.showerror("Error", "File is not in old format!")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Error converting file: {str(e)}")

if __name__ == "__main__":
    # Make sure required packages are installed
    try:
        import PIL
    except ImportError:
        print("Installing required packages...")
        os.system("pip install Pillow")
        import PIL
    
    app = VideoLabeler()
    app.run()