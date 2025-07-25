import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import threading
import time
import os
from datetime import datetime
import sounddevice as sd
import json
import queue
import math

class KeyboardTracker:
    def __init__(self, model_path='best_v2.pt'):
        """Initialize the Keyboard Tracker with YOLO model."""
        self.model = YOLO(model_path)
        self.cap = None
        self.video_file = None
        self.use_camera = True
        self.recording = False
        self.tracking_active = False
        self.keyboard_bboxes = {}  # Store all keyboard key bounding boxes
        
        # Create output directories first
        self.output_dir = "keyboard_data"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "sequences"), exist_ok=True)
        
        # Now we can use output_dir for other file paths
        self.keyboard_layout_built = False  # Track if layout is already built
        self.keyboard_layout_file = os.path.join(self.output_dir, "keyboard_layout.json")
        self.audio_queue = queue.Queue()
        self.audio_threshold = 0.000016  # Calibrated threshold based on audio test
        self.key_press_events = []  # Store detected key press events
        
        # Audio settings
        self.audio_sample_rate = 44100
        self.audio_block_size = 1024
        self.audio_monitoring = False
        self.last_audio_time = 0
        self.audio_debounce_time = 0.3  # 300ms debounce
        self.audio_input_device = None  # Will be set by user
        
        # Video processing mode
        self.video_processing_mode = "manual"  # "manual" or "auto"
        
        # Manual trigger for key press detection
        self.manual_trigger_requested = False
        
        # Data collection settings
        self.data_collection_active = False
        self.current_key_label = None
        self.collected_data = []
        self.video_paused = False
        self.current_frame = None
        self.current_frame_number = 0
        self.pause_for_labeling = False
        
        # Load existing data if available
        self.data_file = os.path.join(self.output_dir, "collected_data.json")
        self.load_existing_data()
        
        # Load keyboard layout if available
        self.load_keyboard_layout()
        
        # GUI elements
        self.root = tk.Tk()
        self.root.title("Keyboard Tracker & Data Collector")
        self.root.geometry("800x600")
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = tk.Label(main_frame, text="Keyboard Tracker & Data Collector", 
                              font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Status section
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.status_label = tk.Label(status_frame, text="Status: Ready", font=("Arial", 10))
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.tracking_label = tk.Label(status_frame, text="Tracking: Not Started", font=("Arial", 10))
        self.tracking_label.grid(row=1, column=0, sticky=tk.W)
        
        self.audio_label = tk.Label(status_frame, text="Audio: Not Monitoring", font=("Arial", 10))
        self.audio_label.grid(row=2, column=0, sticky=tk.W)
        
        # Control buttons
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Video source selection
        source_frame = ttk.Frame(control_frame)
        source_frame.grid(row=0, column=0, columnspan=3, pady=5)
        
        self.video_source_var = tk.StringVar(value="camera")
        tk.Radiobutton(source_frame, text="Camera", variable=self.video_source_var, 
                      value="camera", command=self.on_source_change).grid(row=0, column=0, padx=5)
        tk.Radiobutton(source_frame, text="Video File", variable=self.video_source_var, 
                      value="file", command=self.on_source_change).grid(row=0, column=1, padx=5)
        
        self.select_video_btn = tk.Button(source_frame, text="Select Video File", 
                                        command=self.select_video_file,
                                        font=("Arial", 9), bg="#607D8B", fg="white",
                                        state=tk.DISABLED)
        self.select_video_btn.grid(row=0, column=2, padx=5)
        
        # Video file info
        self.video_file_label = tk.Label(source_frame, text="No video file selected", 
                                       font=("Arial", 8), fg="gray")
        self.video_file_label.grid(row=1, column=0, columnspan=3, pady=2)
        
        # Main control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
        self.start_tracking_btn = tk.Button(button_frame, text="Start Video Processing", 
                                          command=self.start_video_processing,
                                          font=("Arial", 10), bg="#4CAF50", fg="white")
        self.start_tracking_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.build_layout_btn = tk.Button(button_frame, text="Build Layout", 
                                        command=self.build_keyboard_layout,
                                        font=("Arial", 10), bg="#673AB7", fg="white")
        self.build_layout_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.show_layout_btn = tk.Button(button_frame, text="Show Layout", 
                                       command=self.show_keyboard_layout,
                                       font=("Arial", 10), bg="#FF5722", fg="white")
        self.show_layout_btn.grid(row=0, column=2, padx=5, pady=5)
        
        self.start_audio_btn = tk.Button(button_frame, text="Start Audio Monitoring", 
                                       command=self.toggle_audio_monitoring,
                                       font=("Arial", 10), bg="#2196F3", fg="white")
        self.start_audio_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # Audio threshold control
        threshold_frame = ttk.Frame(control_frame)
        threshold_frame.grid(row=2, column=0, columnspan=3, pady=5)
        
        tk.Label(threshold_frame, text="Audio Threshold:").grid(row=0, column=0, padx=5)
        self.threshold_var = tk.StringVar(value=str(self.audio_threshold))
        self.threshold_entry = tk.Entry(threshold_frame, textvariable=self.threshold_var, width=10)
        self.threshold_entry.grid(row=0, column=1, padx=5)
        
        tk.Button(threshold_frame, text="Update", command=self.update_audio_threshold,
                 font=("Arial", 8), bg="#9C27B0", fg="white").grid(row=0, column=2, padx=5)
        
        # Audio level display
        self.audio_level_label = tk.Label(threshold_frame, text="Audio Level: 0.000000", 
                                        font=("Arial", 8), fg="blue")
        self.audio_level_label.grid(row=0, column=3, padx=10)
        
        # Audio device selection
        device_frame = ttk.Frame(control_frame)
        device_frame.grid(row=3, column=0, columnspan=3, pady=5)
        
        tk.Label(device_frame, text="Audio Device:").grid(row=0, column=0, padx=5)
        self.audio_device_var = tk.StringVar(value="Default")
        self.audio_device_combo = ttk.Combobox(device_frame, textvariable=self.audio_device_var, 
                                              width=30, state="readonly")
        self.audio_device_combo.grid(row=0, column=1, padx=5)
        
        tk.Button(device_frame, text="Refresh Devices", command=self.refresh_audio_devices,
                 font=("Arial", 8), bg="#607D8B", fg="white").grid(row=0, column=2, padx=5)
        
        # Manual trigger button
        self.manual_trigger_btn = tk.Button(device_frame, text="Manual Key Press", 
                                          command=self.trigger_manual_key_press,
                                          font=("Arial", 10), bg="#E91E63", fg="white")
        self.manual_trigger_btn.grid(row=0, column=3, padx=5)
        
        # Initialize audio devices
        self.refresh_audio_devices()
        
        self.stop_btn = tk.Button(button_frame, text="Stop All", 
                                command=self.stop_all,
                                font=("Arial", 10), bg="#f44336", fg="white")
        self.stop_btn.grid(row=0, column=4, padx=5, pady=5)
        
        # Key press detection section
        keypress_frame = ttk.LabelFrame(main_frame, text="Key Press Detection", padding="10")
        keypress_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.keypress_list = tk.Listbox(keypress_frame, height=8)
        self.keypress_list.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        scrollbar = ttk.Scrollbar(keypress_frame, orient="vertical", command=self.keypress_list.yview)
        scrollbar.grid(row=0, column=2, sticky=(tk.N, tk.S))
        self.keypress_list.configure(yscrollcommand=scrollbar.set)
        
        # Key labeling section
        label_frame = ttk.LabelFrame(main_frame, text="Key Labeling", padding="10")
        label_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        tk.Label(label_frame, text="Key Label:").grid(row=0, column=0, sticky=tk.W)
        self.key_label_entry = tk.Entry(label_frame, width=20)
        self.key_label_entry.grid(row=0, column=1, padx=5)
        
        self.label_key_btn = tk.Button(label_frame, text="Label Key & Continue", 
                                     command=self.label_key_and_continue,
                                     font=("Arial", 10), bg="#FF9800", fg="white")
        self.label_key_btn.grid(row=0, column=2, padx=5)
        
        self.skip_frame_btn = tk.Button(label_frame, text="Skip Frame", 
                                      command=self.skip_current_frame,
                                      font=("Arial", 10), bg="#9E9E9E", fg="white")
        self.skip_frame_btn.grid(row=0, column=3, padx=5)
        
        # Video control section
        video_control_frame = ttk.LabelFrame(main_frame, text="Video Control", padding="10")
        video_control_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.frame_info_label = tk.Label(video_control_frame, text="Frame: 0/0", font=("Arial", 10))
        self.frame_info_label.grid(row=0, column=0, sticky=tk.W)
        
        self.pause_status_label = tk.Label(video_control_frame, text="Status: Ready", font=("Arial", 10))
        self.pause_status_label.grid(row=0, column=1, sticky=tk.W, padx=20)
        
        # Data collection section
        data_frame = ttk.LabelFrame(main_frame, text="Data Collection", padding="10")
        data_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        self.data_count_label = tk.Label(data_frame, text="Collected Samples: 0")
        self.data_count_label.grid(row=0, column=0, sticky=tk.W)
        
        self.save_data_btn = tk.Button(data_frame, text="Save Data", 
                                     command=self.save_collected_data,
                                     font=("Arial", 10), bg="#9C27B0", fg="white")
        self.save_data_btn.grid(row=0, column=1, padx=5)
        
        self.clear_data_btn = tk.Button(data_frame, text="Clear Data", 
                                      command=self.clear_collected_data,
                                      font=("Arial", 10), bg="#795548", fg="white")
        self.clear_data_btn.grid(row=0, column=2, padx=5)
        
        # Instructions
        instructions = tk.Text(main_frame, height=10, width=80)
        instructions.grid(row=7, column=0, columnspan=3, pady=10)
        instructions.insert(tk.END, "Instructions:\n")
        instructions.insert(tk.END, "1. Build keyboard layout first if not already done\n")
        instructions.insert(tk.END, "2. Choose video source and select video file\n")
        instructions.insert(tk.END, "3. Select audio device:\n")
        instructions.insert(tk.END, "   - For video audio: Use 'Stereo Mix' if available\n")
        instructions.insert(tk.END, "   - For live audio: Use microphone device\n")
        instructions.insert(tk.END, "4. Start Audio Monitoring (optional) or use Manual Key Press\n")
        instructions.insert(tk.END, "5. Start Video Processing\n")
        instructions.insert(tk.END, "6. When you hear/see a key press:\n")
        instructions.insert(tk.END, "   - Auto: Video pauses automatically with audio detection\n")
        instructions.insert(tk.END, "   - Manual: Click 'Manual Key Press' to pause video\n")
        instructions.insert(tk.END, "7. Enter the key label and click 'Label Key & Continue'\n")
        instructions.insert(tk.END, "8. Or click 'Skip Frame' to ignore this detection\n")
        instructions.insert(tk.END, "9. Process continues until video ends\n")
        instructions.config(state=tk.DISABLED)
        
    def load_existing_data(self):
        """Load existing collected data."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    self.collected_data = json.load(f)
                print(f"Loaded {len(self.collected_data)} existing data samples")
            except Exception as e:
                print(f"Error loading existing data: {e}")
                self.collected_data = []
        else:
            self.collected_data = []
            
    def load_keyboard_layout(self):
        """Load existing keyboard layout if available."""
        if os.path.exists(self.keyboard_layout_file):
            try:
                with open(self.keyboard_layout_file, 'r') as f:
                    layout_data = json.load(f)
                    
                keyboard_bboxes_json = layout_data.get('keyboard_bboxes', {})
                
                # Convert lists back to numpy arrays
                self.keyboard_bboxes = {}
                for key, info in keyboard_bboxes_json.items():
                    self.keyboard_bboxes[key] = {
                        'bbox': np.array(info['bbox']),  # Convert list back to numpy array
                        'confidence': float(info['confidence'])
                    }
                
                self.keyboard_layout_built = len(self.keyboard_bboxes) > 0
                if self.keyboard_layout_built:
                    print(f"Loaded keyboard layout with {len(self.keyboard_bboxes)} keys")
                    self.update_layout_status()
            except Exception as e:
                print(f"Error loading keyboard layout: {e}")
                self.keyboard_layout_built = False
        else:
            self.keyboard_layout_built = False
            
    def save_keyboard_layout(self):
        """Save current keyboard layout to file."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            keyboard_bboxes_json = {}
            for key, info in self.keyboard_bboxes.items():
                keyboard_bboxes_json[key] = {
                    'bbox': info['bbox'].tolist(),  # Convert numpy array to list
                    'confidence': float(info['confidence'])  # Ensure float
                }
            
            layout_data = {
                'keyboard_bboxes': keyboard_bboxes_json,
                'created_timestamp': datetime.now().isoformat(),
                'total_keys': len(self.keyboard_bboxes)
            }
            
            with open(self.keyboard_layout_file, 'w') as f:
                json.dump(layout_data, f, indent=2)
            
            print(f"Keyboard layout saved with {len(self.keyboard_bboxes)} keys")
            
        except Exception as e:
            print(f"Error saving keyboard layout: {e}")
            
    def update_layout_status(self):
        """Update GUI to show layout status."""
        if self.keyboard_layout_built:
            key_count = len(self.keyboard_bboxes)
            self.status_label.config(text=f"Status: Keyboard Layout Ready ({key_count} keys)")
            self.build_layout_btn.config(text="Rebuild Layout", bg="#FF9800")
            self.show_layout_btn.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="Status: No Keyboard Layout")
            self.build_layout_btn.config(text="Build Layout", bg="#673AB7")
            self.show_layout_btn.config(state=tk.DISABLED)
            
    def start_tracking(self):
        """Legacy method - redirect to build layout."""
        self.build_keyboard_layout()
        
    def track_keyboard(self):
        """Legacy method - redirect to build layout."""
        self.build_layout_process()
        
    def toggle_audio_monitoring(self):
        """Toggle audio monitoring."""
        if not self.audio_monitoring:
            self.start_audio_monitoring()
        else:
            self.stop_audio_monitoring()
            
    def start_audio_monitoring(self):
        """Start monitoring audio for key presses."""
        self.audio_monitoring = True
        self.start_audio_btn.config(text="Stop Audio Monitoring", bg="#f44336")
        self.audio_label.config(text="Audio: Monitoring")
        
        # Start audio monitoring thread
        audio_thread = threading.Thread(target=self.monitor_audio)
        audio_thread.daemon = True
        audio_thread.start()
        
    def stop_audio_monitoring(self):
        """Stop audio monitoring."""
        self.audio_monitoring = False
        self.start_audio_btn.config(text="Start Audio Monitoring", bg="#2196F3")
        self.audio_label.config(text="Audio: Not Monitoring")
        
    def monitor_audio(self):
        """Monitor audio for key press detection."""
        try:
            device_id = self.get_selected_audio_device()
            device_name = self.audio_device_var.get()
            
            print(f"Starting audio monitoring with device: {device_name}")
            
            with sd.InputStream(callback=self.audio_callback, 
                              samplerate=self.audio_sample_rate,
                              blocksize=self.audio_block_size,
                              channels=1,
                              device=device_id):
                while self.audio_monitoring:
                    time.sleep(0.1)
                    
                    # Process audio queue
                    try:
                        while True:
                            audio_data = self.audio_queue.get_nowait()
                            self.process_audio_data(audio_data)
                    except queue.Empty:
                        pass
                        
        except Exception as e:
            print(f"Audio monitoring error: {e}")
            messagebox.showerror("Audio Error", f"Audio monitoring failed: {e}\n\nTry selecting a different audio device or use manual trigger.")
            self.audio_monitoring = False
            self.start_audio_btn.config(text="Start Audio Monitoring", bg="#2196F3")
            self.audio_label.config(text="Audio: Error")
            
    def audio_callback(self, indata, frames, time, status):
        """Audio callback function."""
        if self.audio_monitoring:
            self.audio_queue.put(indata.copy())
            
    def process_audio_data(self, audio_data):
        """Process audio data to detect key presses."""
        # Calculate RMS (Root Mean Square) for volume detection
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Update audio level display
        self.root.after(0, lambda: self.update_audio_level_display(rms))
        
        # Debug: Print current audio level periodically
        current_time = time.time()
        if current_time - getattr(self, 'last_debug_time', 0) > 2:  # Every 2 seconds
            print(f"Audio level: {rms:.6f}, Threshold: {self.audio_threshold:.6f}")
            self.last_debug_time = current_time
        
        # If volume exceeds threshold, consider it a key press
        if rms > self.audio_threshold:
            # Check debounce time
            if current_time - self.last_audio_time < self.audio_debounce_time:
                return  # Too soon, ignore
                
            self.last_audio_time = current_time
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            print(f"🎵 KEY PRESS DETECTED! Volume: {rms:.6f} at {timestamp}")
            
            # Add to key press events
            event = {
                'timestamp': timestamp,
                'volume': float(rms),
                'labeled': False,
                'key': None,
                'bbox_data': None,
                'frame_number': self.current_frame_number
            }
            
            self.key_press_events.append(event)
            
            # If video processing is active, trigger pause
            if self.tracking_active and not self.video_paused:
                self.pause_for_labeling = True
                print(f"📹 PAUSING VIDEO at frame {self.current_frame_number}")
            
            # Update GUI
            self.root.after(0, self.update_keypress_list)
            self.update_audio_level_display(rms)
            
    def update_keypress_list(self):
        """Update the key press list in GUI."""
        self.keypress_list.delete(0, tk.END)
        for i, event in enumerate(self.key_press_events):
            status = "✓" if event['labeled'] else "○"
            key_label = event['key'] if event['key'] else "Unknown"
            frame_info = f"Frame {event.get('frame_number', 0)}" if event.get('frame_number') else ""
            
            # Check if it's a manual trigger
            if event.get('manual_trigger', False):
                trigger_type = "MANUAL"
                volume_info = ""
            else:
                trigger_type = "AUDIO"
                volume_info = f" (Vol: {event['volume']:.3f})"
            
            if event['key'] == 'SKIPPED':
                display_text = f"{status} {event['timestamp']} - SKIPPED {frame_info} [{trigger_type}]{volume_info}"
            else:
                display_text = f"{status} {event['timestamp']} - {key_label} {frame_info} [{trigger_type}]{volume_info}"
            
            self.keypress_list.insert(tk.END, display_text)
            
    def label_key_press(self):
        """Legacy method - redirect to new workflow."""
        if self.video_paused:
            self.label_key_and_continue()
        else:
            messagebox.showinfo("Info", "This function is now integrated into the video processing workflow.\n"
                                       "Please use 'Start Video Processing' for the new workflow.")
            
    def expand_bbox_by_percentage(self, bbox, percentage):
        """Expand bounding box by percentage."""
        x1, y1, x2, y2 = bbox
        
        # Calculate current dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Calculate expansion amounts
        width_expansion = width * percentage / 2
        height_expansion = height * percentage / 2
        
        # Expand the bounding box
        new_x1 = x1 - width_expansion
        new_y1 = y1 - height_expansion
        new_x2 = x2 + width_expansion
        new_y2 = y2 + height_expansion
        
        # Ensure coordinates are not negative
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        
        return np.array([new_x1, new_y1, new_x2, new_y2])
        
    def update_data_count(self):
        """Update the data count label."""
        count = len(self.collected_data)
        self.data_count_label.config(text=f"Collected Samples: {count}")
        
    def save_collected_data(self):
        """Save collected data to JSON file."""
        if not self.collected_data:
            messagebox.showwarning("Warning", "No data to save")
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"keyboard_data_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(self.collected_data, f, indent=2)
                
            # Also save to the main data file
            with open(self.data_file, 'w') as f:
                json.dump(self.collected_data, f, indent=2)
                
            messagebox.showinfo("Success", f"Data saved successfully!\nFile: {filename}\nSamples: {len(self.collected_data)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {e}")
            
    def clear_collected_data(self):
        """Clear all collected data."""
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all collected data?"):
            self.collected_data = []
            self.key_press_events = []
            self.update_keypress_list()
            self.update_data_count()
            messagebox.showinfo("Success", "All data cleared")
            
    def stop_all(self):
        """Stop all operations and close application."""
        self.tracking_active = False
        self.audio_monitoring = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        
        self.root.quit()
        
    def run(self):
        """Run the application."""
        self.update_data_count()
        self.update_layout_status()
        self.root.protocol("WM_DELETE_WINDOW", self.stop_all)
        self.root.mainloop()
        
    def on_source_change(self):
        """Handle video source change (camera/file)."""
        if self.video_source_var.get() == "file":
            self.select_video_btn.config(state=tk.NORMAL)
            self.use_camera = False
        else:
            self.select_video_btn.config(state=tk.DISABLED)
            self.use_camera = True
            self.video_file = None
            self.video_file_label.config(text="No video file selected")
            
    def select_video_file(self):
        """Select video file for processing."""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.video_file = file_path
            filename = os.path.basename(file_path)
            self.video_file_label.config(text=f"Selected: {filename}")
            print(f"Selected video file: {file_path}")
            
    def on_video_ended(self):
        """Handle video file ending."""
        result = messagebox.askyesno("Video Ended", 
                                   "Video playback has ended. Do you want to replay from the beginning?")
        if result:
            # Restart video from beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # Continue tracking
            tracking_thread = threading.Thread(target=self.track_keyboard)
            tracking_thread.daemon = True
            tracking_thread.start()
        else:
            # Stop tracking
            self.tracking_active = False
            self.start_tracking_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Video Ended")
            self.tracking_label.config(text="Tracking: Stopped")
            
    def build_keyboard_layout(self):
        """Build keyboard layout by detecting all keys."""
        if self.use_camera:
            if self.cap is None:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Cannot open camera")
                    return
        else:
            if self.video_file is None:
                messagebox.showerror("Error", "Please select a video file first")
                return
            
            self.cap = cv2.VideoCapture(self.video_file)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Cannot open video file: {self.video_file}")
                return
        
        self.tracking_active = True
        self.start_tracking_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Building Keyboard Layout...")
        self.tracking_label.config(text="Layout Building: Active")
        
        # Start layout building in a separate thread
        layout_thread = threading.Thread(target=self.build_layout_process)
        layout_thread.daemon = True
        layout_thread.start()
        
    def build_layout_process(self):
        """Process multiple frames to build comprehensive keyboard layout."""
        detected_keys = {}
        frame_count = 0
        max_frames = 30  # Analyze 30 frames to get comprehensive layout
        
        print("Building keyboard layout from multiple frames...")
        
        while self.cap.isOpened() and self.tracking_active and frame_count < max_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Run YOLO detection
            results = self.model(frame)
            
            # Process detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        if confidence > 0.5:  # Confidence threshold
                            bbox = box.xyxy[0].cpu().numpy()
                            
                            # If this key hasn't been detected or has lower confidence, update it
                            if class_name not in detected_keys or detected_keys[class_name]['confidence'] < confidence:
                                detected_keys[class_name] = {
                                    'bbox': bbox,
                                    'confidence': confidence
                                }
            
            # Show progress
            if frame_count % 5 == 0:
                print(f"Processed {frame_count}/{max_frames} frames, found {len(detected_keys)} unique keys")
                
            # Optional: Show current frame with detections
            if frame_count <= 10:  # Show first 10 frames
                annotated_frame = results[0].plot()
                cv2.putText(annotated_frame, f'Building Layout: {frame_count}/{max_frames}', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'Keys found: {len(detected_keys)}', 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated_frame, 'Press Q to stop early', 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Building Keyboard Layout', annotated_frame)
                
                # Allow early termination
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        cv2.destroyAllWindows()
        
        # Update keyboard layout
        self.keyboard_bboxes = detected_keys
        self.keyboard_layout_built = len(detected_keys) > 0
        
        # Save layout
        if self.keyboard_layout_built:
            self.save_keyboard_layout()
            
        # Update GUI
        self.root.after(0, self.on_layout_build_complete)
        
    def on_layout_build_complete(self):
        """Handle layout building completion."""
        self.tracking_active = False
        self.start_tracking_btn.config(state=tk.NORMAL)
        self.tracking_label.config(text="Layout Building: Complete")
        
        if self.keyboard_layout_built:
            key_count = len(self.keyboard_bboxes)
            self.update_layout_status()
            messagebox.showinfo("Success", f"Keyboard layout built successfully!\nDetected {key_count} keys\nLayout saved for future use.")
            
            # Print detected keys for debugging
            print("Detected keys:")
            for key, info in self.keyboard_bboxes.items():
                print(f"  {key}: confidence={info['confidence']:.3f}")
        else:
            messagebox.showwarning("Warning", "No keyboard keys detected. Please try again with better lighting or camera position.")
            
    def show_keyboard_layout(self):
        """Show the current keyboard layout visually."""
        if not self.keyboard_layout_built:
            messagebox.showwarning("Warning", "No keyboard layout available. Please build layout first.")
            return
            
        # Create a visualization of the keyboard layout
        if self.use_camera:
            if self.cap is None:
                self.cap = cv2.VideoCapture(0)
        else:
            if self.video_file:
                self.cap = cv2.VideoCapture(self.video_file)
        
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Draw all detected keys on the frame
                for key_name, key_info in self.keyboard_bboxes.items():
                    bbox = key_info['bbox']
                    confidence = key_info['confidence']
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = bbox.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw key label
                    cv2.putText(frame, f'{key_name}', (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Draw confidence
                    cv2.putText(frame, f'{confidence:.2f}', (x1, y2+15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                
                # Add title
                cv2.putText(frame, f'Keyboard Layout ({len(self.keyboard_bboxes)} keys)', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, 'Press any key to close', 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Keyboard Layout', frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
    def start_video_processing(self):
        """Start video processing with audio-triggered pausing."""
        if not self.keyboard_layout_built:
            messagebox.showwarning("Warning", "Please build keyboard layout first!")
            return
            
        if not self.audio_monitoring:
            messagebox.showwarning("Warning", "Please start audio monitoring first!")
            return
            
        if self.use_camera:
            messagebox.showinfo("Info", "Camera mode not supported for this workflow. Please use video file.")
            return
        else:
            if self.video_file is None:
                messagebox.showerror("Error", "Please select a video file first")
                return
            
            self.cap = cv2.VideoCapture(self.video_file)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Cannot open video file: {self.video_file}")
                return
        
        self.tracking_active = True
        self.video_paused = False
        self.current_frame_number = 0
        self.start_tracking_btn.config(state=tk.DISABLED)
        self.pause_status_label.config(text="Status: Processing Video...")
        
        # Clear previous key press events for this session
        self.key_press_events = []
        self.update_keypress_list()
        
        # Start video processing in a separate thread
        video_thread = threading.Thread(target=self.process_video_with_audio)
        video_thread.daemon = True
        video_thread.start()
        
    def process_video_with_audio(self):
        """Process video and pause when audio detects key press."""
        if not self.cap or not self.cap.isOpened():
            print("Error: Video capture is not initialized or opened")
            return
            
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Starting video processing: {total_frames} frames at {fps} FPS")
        
        while self.cap.isOpened() and self.tracking_active:
            if not self.video_paused:
                ret, frame = self.cap.read()
                if not ret:
                    # Video ended
                    self.root.after(0, self.on_video_processing_complete)
                    break
                    
                self.current_frame_number += 1
                self.current_frame = frame.copy()
                
                # Update frame info
                self.root.after(0, lambda: self.frame_info_label.config(
                    text=f"Frame: {self.current_frame_number}/{total_frames}"))
                
                # Check if we should pause due to audio detection
                if self.pause_for_labeling:
                    self.video_paused = True
                    self.pause_for_labeling = False
                    self.root.after(0, self.on_video_paused)
                    
                # Display current frame
                self.display_current_frame()
                
                # Control playback speed
                if fps > 0:
                    time.sleep(1.0 / fps)
            else:
                # Video is paused, wait
                time.sleep(0.1)
                
        cv2.destroyAllWindows()
        
    def display_current_frame(self):
        """Display current frame with keyboard layout overlay."""
        if self.current_frame is None:
            return
            
        display_frame = self.current_frame.copy()
        
        # Draw keyboard layout
        for key_name, key_info in self.keyboard_bboxes.items():
            bbox = key_info['bbox']
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(display_frame, key_name, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Add frame info
        cv2.putText(display_frame, f'Frame: {self.current_frame_number}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if self.video_paused:
            cv2.putText(display_frame, 'PAUSED - Label the key that was pressed', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, 'Check GUI to enter key label', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, 'Processing... Audio monitoring active', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('Video Processing', display_frame)
        cv2.waitKey(1)
        
    def on_video_paused(self):
        """Handle when video is paused for key labeling."""
        self.pause_status_label.config(text="Status: PAUSED - Please label the key")
        self.key_label_entry.focus_set()  # Focus on entry field
        messagebox.showinfo("Key Press Detected", 
                           f"Key press detected at frame {self.current_frame_number}!\n"
                           f"Please enter the key that was pressed and click 'Label Key & Continue'")
        
    def label_key_and_continue(self):
        """Label the current key press and continue video processing."""
        if not self.video_paused:
            messagebox.showwarning("Warning", "Video is not paused. No key to label.")
            return
            
        key_label = self.key_label_entry.get().strip().upper()
        if not key_label:
            messagebox.showwarning("Warning", "Please enter a key label")
            return
            
        # Find the most recent key press event
        if not self.key_press_events:
            messagebox.showerror("Error", "No key press event found")
            return
            
        # Get the most recent unlabeled event
        recent_event = None
        for event in reversed(self.key_press_events):
            if not event['labeled']:
                recent_event = event
                break
                
        if recent_event is None:
            messagebox.showerror("Error", "No unlabeled key press event found")
            return
            
        # Find bounding box for the key
        if key_label in self.keyboard_bboxes:
            bbox_info = self.keyboard_bboxes[key_label]
            original_bbox = bbox_info['bbox']
            
            # Expand bounding box by 20%
            expanded_bbox = self.expand_bbox_by_percentage(original_bbox, 0.2)
            
            # Update the event
            recent_event['labeled'] = True
            recent_event['key'] = key_label
            recent_event['frame_number'] = self.current_frame_number
            recent_event['bbox_data'] = {
                'original_bbox': original_bbox.tolist(),
                'expanded_bbox': expanded_bbox.tolist(),
                'expansion_percentage': 0.2
            }
            
            # Add to collected data
            data_sample = {
                'timestamp': recent_event['timestamp'],
                'frame_number': self.current_frame_number,
                'key': key_label,
                'volume': recent_event['volume'],
                'original_bbox': original_bbox.tolist(),
                'expanded_bbox': expanded_bbox.tolist(),
                'confidence': bbox_info['confidence']
            }
            
            self.collected_data.append(data_sample)
            
            # Update GUI
            self.update_keypress_list()
            self.update_data_count()
            self.key_label_entry.delete(0, tk.END)
            
            # Continue video processing
            self.video_paused = False
            self.pause_status_label.config(text="Status: Processing Video...")
            
            print(f"Labeled key '{key_label}' at frame {self.current_frame_number}")
            
        else:
            messagebox.showerror("Error", f"Key '{key_label}' not found in keyboard layout.")
            
    def skip_current_frame(self):
        """Skip current frame and continue video processing."""
        if not self.video_paused:
            messagebox.showwarning("Warning", "Video is not paused.")
            return
            
        # Mark the most recent event as skipped
        if self.key_press_events:
            for event in reversed(self.key_press_events):
                if not event['labeled']:
                    event['labeled'] = True
                    event['key'] = 'SKIPPED'
                    event['frame_number'] = self.current_frame_number
                    break
                    
        self.update_keypress_list()
        
        # Continue video processing
        self.video_paused = False
        self.pause_status_label.config(text="Status: Processing Video...")
        self.key_label_entry.delete(0, tk.END)
        
        print(f"Skipped frame {self.current_frame_number}")
        
    def on_video_processing_complete(self):
        """Handle completion of video processing."""
        self.tracking_active = False
        self.video_paused = False
        self.start_tracking_btn.config(state=tk.NORMAL)
        self.pause_status_label.config(text="Status: Video Processing Complete")
        
        completed_samples = len([d for d in self.collected_data if d.get('frame_number')])
        messagebox.showinfo("Video Processing Complete", 
                           f"Video processing finished!\n"
                           f"Collected {completed_samples} labeled samples\n"
                           f"Don't forget to save your data!")
        
    def update_audio_threshold(self):
        """Update audio threshold from GUI input."""
        try:
            new_threshold = float(self.threshold_var.get())
            if new_threshold > 0:
                self.audio_threshold = new_threshold
                print(f"Audio threshold updated to: {new_threshold}")
                messagebox.showinfo("Success", f"Audio threshold set to {new_threshold}")
            else:
                messagebox.showerror("Error", "Threshold must be greater than 0")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
            
    def update_audio_level_display(self, rms):
        """Update audio level display in GUI."""
        self.audio_level_label.config(text=f"Audio Level: {rms:.6f}")
        
        # Color coding based on threshold
        if rms > self.audio_threshold:
            self.audio_level_label.config(fg="red")  # Above threshold
        elif rms > self.audio_threshold * 0.5:
            self.audio_level_label.config(fg="orange")  # Close to threshold
        else:
            self.audio_level_label.config(fg="blue")  # Normal level
            
    def refresh_audio_devices(self):
        """Refresh the list of available audio input devices."""
        try:
            devices = sd.query_devices()
            input_devices = []
            
            # Find input devices
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    device_name = f"{i}: {device['name']}"
                    input_devices.append(device_name)
            
            # Update combobox
            self.audio_device_combo['values'] = input_devices
            
            # Set default device
            if input_devices:
                self.audio_device_combo.set(input_devices[0])
            
            print(f"Found {len(input_devices)} audio input devices")
            
        except Exception as e:
            print(f"Error refreshing audio devices: {e}")
            self.audio_device_combo['values'] = ["Default"]
            self.audio_device_combo.set("Default")
    
    def get_selected_audio_device(self):
        """Get the selected audio device ID."""
        try:
            device_str = self.audio_device_var.get()
            if device_str and ":" in device_str:
                device_id = int(device_str.split(":")[0])
                return device_id
            return None
        except:
            return None
    
    def trigger_manual_key_press(self):
        """Manually trigger a key press event."""
        if not self.tracking_active:
            messagebox.showwarning("Warning", "Video processing is not active")
            return
        
        # Create a manual key press event
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        event = {
            'timestamp': timestamp,
            'volume': 0.0,  # No audio for manual trigger
            'labeled': False,
            'key': None,
            'bbox_data': None,
            'frame_number': self.current_frame_number,
            'manual_trigger': True
        }
        
        self.key_press_events.append(event)
        
        # If video processing is active, trigger pause
        if self.tracking_active and not self.video_paused:
            self.pause_for_labeling = True
            self.video_paused = True
            self.root.after(0, self.on_video_paused)
            print(f"📹 MANUAL TRIGGER: Paused video at frame {self.current_frame_number}")
        
        # Update GUI
        self.update_keypress_list()
        
        
def main():
    """Main function to run the Keyboard Tracker application."""
    tracker = KeyboardTracker()
    tracker.run()

main()