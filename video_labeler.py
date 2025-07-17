import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import json
import os
from datetime import datetime

class VideoLabeler:
    def __init__(self, model_path='best_v2.pt'):
        """Initialize the Video Labeler with YOLO model."""
        self.model = YOLO(model_path)
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.current_frame_idx = 0
        self.total_frames = 0
        
        # Key tracking
        self.target_key = 't'
        self.key_bbox = None
        self.expanded_bbox = None
        self.key_roi = None
        
        # Labeling data
        self.labels = {}  # frame_idx: is_pressed (0 or 1)
        self.sequence_data = []  # For CNN+LSTM: [frame_features, label]
        
        # Output
        self.output_dir = "labeled_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # GUI setup
        self.setup_gui()
        
        # Playback control
        self.is_playing = False
        self.play_speed = 30  # FPS
        
    def setup_gui(self):
        """Setup the GUI interface."""
        self.root = tk.Tk()
        self.root.title(f"Video Labeler - Key '{self.target_key}'")
        self.root.geometry("1200x800")
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Video Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Select Video", command=self.select_video).pack(side=tk.LEFT, padx=(0, 10))
        self.video_label = ttk.Label(file_frame, text="No video selected")
        self.video_label.pack(side=tk.LEFT)
        
        # Key detection frame
        key_frame = ttk.LabelFrame(main_frame, text="Key Detection", padding=10)
        key_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(key_frame, text="Detect Key T", command=self.detect_key_t).pack(side=tk.LEFT, padx=(0, 10))
        self.key_status_label = ttk.Label(key_frame, text="Key not detected")
        self.key_status_label.pack(side=tk.LEFT)
        
        # Video control frame
        control_frame = ttk.LabelFrame(main_frame, text="Video Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Frame navigation
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(nav_frame, text="<<", command=self.prev_frame).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="Play/Pause", command=self.toggle_play).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text=">>", command=self.next_frame).pack(side=tk.LEFT, padx=(0, 5))
        
        # Frame info
        self.frame_info = ttk.Label(nav_frame, text="Frame: 0 / 0")
        self.frame_info.pack(side=tk.LEFT, padx=(20, 0))
        
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
        
        # Canvas for video display
        self.canvas = tk.Canvas(main_frame, bg="black", height=400)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Key bindings
        self.root.bind('<Key-0>', lambda e: self.label_frame(0))
        self.root.bind('<Key-1>', lambda e: self.label_frame(1))
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('<space>', lambda e: self.toggle_play())
        self.root.bind('<Delete>', lambda e: self.remove_label())
        self.root.focus_set()
        
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
        
        # Load first frame
        self.current_frame_idx = 0
        self.update_frame()
        
        messagebox.showinfo("Success", f"Video loaded successfully!\nTotal frames: {self.total_frames}")
        
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
                    if result.names[int(box.cls)] == 't':
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        self.key_bbox = (int(x1), int(y1), int(x2), int(y2))
                        
                        # Expand bounding box by 10%
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
            
    def update_frame(self):
        """Update the current frame display."""
        if self.cap is None:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame.copy()
            
            # Draw key bounding box if detected
            if self.expanded_bbox:
                x1, y1, x2, y2 = self.expanded_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Key {self.target_key}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show current label status
            label_text = "Not Labeled"
            label_color = (128, 128, 128)
            if self.current_frame_idx in self.labels:
                if self.labels[self.current_frame_idx] == 1:
                    label_text = "PRESSED"
                    label_color = (0, 255, 0)
                else:
                    label_text = "NOT PRESSED"
                    label_color = (0, 0, 255)
            
            cv2.putText(frame, label_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
            
            # Display frame
            self.display_frame(frame)
            
            # Update UI
            self.frame_info.config(text=f"Frame: {self.current_frame_idx} / {self.total_frames-1}")
            self.progress_var.set(self.current_frame_idx)
            self.update_current_label_display()
            self.update_stats()
            
    def display_frame(self, frame):
        """Display frame on canvas."""
        # Resize frame to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Calculate scale to fit
            scale_x = canvas_width / frame.shape[1]
            scale_y = canvas_height / frame.shape[0]
            scale = min(scale_x, scale_y)
            
            new_width = int(frame.shape[1] * scale)
            new_height = int(frame.shape[0] * scale)
            
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to RGB for tkinter
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PhotoImage
            from PIL import Image, ImageTk
            pil_image = Image.fromarray(rgb_frame)
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and display image
            self.canvas.delete("all")
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
            
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
        """Label current frame."""
        if self.expanded_bbox is None:
            messagebox.showerror("Error", "Please detect key T first!")
            return
            
        self.labels[self.current_frame_idx] = label
        self.update_current_label_display()
        self.update_stats()
        
        # Auto advance to next frame
        if self.current_frame_idx < self.total_frames - 1:
            self.next_frame()
            
    def remove_label(self):
        """Remove label from current frame."""
        if self.current_frame_idx in self.labels:
            del self.labels[self.current_frame_idx]
            self.update_current_label_display()
            self.update_stats()
            
    def update_current_label_display(self):
        """Update current frame label display."""
        if self.current_frame_idx in self.labels:
            label_val = self.labels[self.current_frame_idx]
            if label_val == 1:
                self.current_label_var.set("PRESSED")
                self.current_label_display.config(background="lightgreen")
            else:
                self.current_label_var.set("NOT PRESSED")
                self.current_label_display.config(background="lightcoral")
        else:
            self.current_label_var.set("Not Labeled")
            self.current_label_display.config(background="lightgray")
            
    def update_stats(self):
        """Update labeling statistics."""
        total = len(self.labels)
        pressed = sum(1 for label in self.labels.values() if label == 1)
        not_pressed = total - pressed
        
        self.stats_label.config(text=f"Labels: {total} total, {pressed} pressed, {not_pressed} not pressed")
        
    def save_labels(self):
        """Save labels to JSON file."""
        if not self.labels:
            messagebox.showwarning("Warning", "No labels to save!")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"labels_{self.target_key}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            "video_path": self.video_path,
            "target_key": self.target_key,
            "key_bbox": self.key_bbox,
            "expanded_bbox": self.expanded_bbox,
            "total_frames": self.total_frames,
            "labels": self.labels,
            "timestamp": timestamp
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
                    
                self.labels = {int(k): v for k, v in data["labels"].items()}
                self.target_key = data.get("target_key", "T")
                self.key_bbox = data.get("key_bbox")
                self.expanded_bbox = data.get("expanded_bbox")
                
                if self.expanded_bbox:
                    self.key_status_label.config(text=f"Key {self.target_key} loaded! Bbox: {self.expanded_bbox}")
                    
                self.update_current_label_display()
                self.update_stats()
                self.update_frame()
                
                messagebox.showinfo("Success", f"Labels loaded from {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading labels: {str(e)}")
                
    def export_for_training(self):
        """Export data in format suitable for CNN+LSTM training."""
        if not self.labels or not self.expanded_bbox:
            messagebox.showwarning("Warning", "Need labels and key detection first!")
            return
            
        # Create sequence data
        sequence_data = []
        
        for frame_idx in sorted(self.labels.keys()):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if ret:
                # Extract key region
                x1, y1, x2, y2 = self.expanded_bbox
                key_region = frame[y1:y2, x1:x2]
                
                # Resize to standard size for CNN
                key_region = cv2.resize(key_region, (64, 64))
                
                # Normalize
                key_region = key_region.astype(np.float32) / 255.0
                
                label = self.labels[frame_idx]
                
                sequence_data.append({
                    "frame_idx": frame_idx,
                    "image": key_region.tolist(),
                    "label": label
                })
                
        # Save training data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_data_{self.target_key}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        export_data = {
            "video_path": self.video_path,
            "target_key": self.target_key,
            "expanded_bbox": self.expanded_bbox,
            "image_size": [64, 64],
            "sequence_data": sequence_data,
            "timestamp": timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        messagebox.showinfo("Success", 
            f"Training data exported to {filepath}\n"
            f"Total samples: {len(sequence_data)}")
            
    def run(self):
        """Start the application."""
        self.root.mainloop()
        
        # Cleanup
        if self.cap:
            self.cap.release()

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