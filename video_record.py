import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox, ttk
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
        """Initialize the Key E recorder with YOLO model."""
        self.model = YOLO(model_path)
        self.cap = None
        self.recording = False
        self.tracking_complete = False
        self.key_e_bbox = None
        self.output_video = None
        self.frame_count = 0
        self.fps = 30
        self.tracking_active = False
        
        # Bounding box expansion settings
        self.bbox_padding = 20  # Pixels to expand around bounding box
        self.bbox_scale = 1.3   # Scale factor to expand bounding box
        
        # Create output directory
        self.output_dir = "recorded_videos"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # GUI elements
        self.root = tk.Tk()
        self.root.title("Key E Recorder")
        self.root.geometry("500x400")
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI interface."""
        # Title
        title_label = tk.Label(self.root, text="Key E Recorder", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Status labels
        self.status_label = tk.Label(self.root, text="Status: Ready", 
                                   font=("Arial", 12))
        self.status_label.pack(pady=5)
        
        self.tracking_label = tk.Label(self.root, text="Tracking: Not Started", 
                                     font=("Arial", 12))
        self.tracking_label.pack(pady=5)
        
        # Buttons
        self.start_tracking_btn = tk.Button(self.root, text="Start Tracking Key E", 
                                          command=self.start_tracking,
                                          font=("Arial", 12), bg="#4CAF50", fg="white")
        self.start_tracking_btn.pack(pady=10)
        
        self.record_btn = tk.Button(self.root, text="Start Recording", 
                                   command=self.toggle_recording,
                                   font=("Arial", 12), bg="#2196F3", fg="white",
                                   state=tk.DISABLED)
        self.record_btn.pack(pady=10)
        
        self.close_tracking_btn = tk.Button(self.root, text="Close Tracking Window", 
                                          command=self.close_tracking_window,
                                          font=("Arial", 12), bg="#FF9800", fg="white",
                                          state=tk.DISABLED)
        self.close_tracking_btn.pack(pady=5)
        
        self.stop_btn = tk.Button(self.root, text="Stop All", 
                                command=self.stop_all,
                                font=("Arial", 12), bg="#f44336", fg="white")
        self.stop_btn.pack(pady=10)
        
        # Info text
        info_text = tk.Text(self.root, height=10, width=50)
        info_text.pack(pady=10)
        info_text.insert(tk.END, "Instructions:\n")
        info_text.insert(tk.END, "1. Click 'Start Tracking Key E' to find key E\n")
        info_text.insert(tk.END, "2. Tracking window will stay open after finding key E\n")
        info_text.insert(tk.END, "3. Green box = expanded recording area\n")
        info_text.insert(tk.END, "4. Red box = original detection area\n")
        info_text.insert(tk.END, "5. Press '+/-' to adjust bounding box size\n")
        info_text.insert(tk.END, "6. Press 'Q' in tracking window to close it\n")
        info_text.insert(tk.END, "7. Press 'R' in tracking window to reset tracking\n")
        info_text.insert(tk.END, "8. Click 'Start Recording' to record video\n")
        info_text.insert(tk.END, "9. Press 'q' in recording window to stop recording")
        info_text.config(state=tk.DISABLED)
        
    def start_tracking(self):
        """Start tracking to find key E."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open camera")
                return
        
        self.tracking_complete = False
        self.tracking_active = True
        self.start_tracking_btn.config(state=tk.DISABLED)
        self.close_tracking_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Tracking Key E...")
        self.tracking_label.config(text="Tracking: Searching for Key E")
        
        # Start tracking in a separate thread
        tracking_thread = threading.Thread(target=self.track_key_e)
        tracking_thread.daemon = True
        tracking_thread.start()
        
    def close_tracking_window(self):
        """Close tracking window and reset tracking state."""
        self.tracking_active = False
        cv2.destroyAllWindows()
        self.start_tracking_btn.config(state=tk.NORMAL)
        self.close_tracking_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Tracking Window Closed")
        self.tracking_label.config(text="Tracking: Stopped")
        
    def track_key_e(self):
        """Track and find key E using YOLO model."""
        while self.cap.isOpened() and self.tracking_active:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Run YOLO detection
            results = self.model(frame)
            
            # Look for key E
            key_e_found_this_frame = False
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        if class_name.upper() == 'E':
                            # Found key E!
                            original_bbox = box.xyxy[0].cpu().numpy()
                            
                            # Expand the bounding box
                            self.key_e_bbox = self.expand_bbox(original_bbox, frame.shape)
                            key_e_found_this_frame = True
                            
                            # Update GUI only once
                            if not self.tracking_complete:
                                self.tracking_complete = True
                                self.root.after(0, self.on_tracking_complete)
                            
                            # Draw original bounding box (red)
                            orig_x1, orig_y1, orig_x2, orig_y2 = original_bbox.astype(int)
                            cv2.rectangle(frame, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 0, 255), 2)
                            cv2.putText(frame, 'Original', (orig_x1, orig_y1-30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            
                            # Draw expanded bounding box (green)
                            x1, y1, x2, y2 = self.key_e_bbox.astype(int)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(frame, f'Key E Found! (Expanded) Press Q to close', (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Show cropped area in separate window
                            cropped_frame = frame[y1:y2, x1:x2]
                            if cropped_frame.size > 0:
                                # Resize cropped frame for better visibility
                                cropped_resized = cv2.resize(cropped_frame, (250, 250))
                                cv2.imshow('Key E Area Preview (Expanded)', cropped_resized)
                            break
            
            # Draw all detections
            annotated_frame = results[0].plot()
            
            # Add status text
            if self.tracking_complete:
                cv2.putText(annotated_frame, 'Key E Tracking Complete - Ready to Record', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated_frame, 'Press Q to close | R to reset | +/- to adjust bbox', 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f'BBox Scale: {self.bbox_scale:.1f} | Padding: {self.bbox_padding}px', 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, 'Searching for Key E...', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(annotated_frame, 'Press +/- to adjust bounding box size', 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Key E Tracking', annotated_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and self.tracking_complete:
                # Reset tracking if R is pressed
                self.tracking_complete = False
                self.key_e_bbox = None
                self.root.after(0, self.on_tracking_reset)
                cv2.destroyWindow('Key E Area Preview (Expanded)')
            elif key == ord('+') or key == ord('='):
                # Increase bounding box size
                self.bbox_scale = min(2.0, self.bbox_scale + 0.1)
                self.bbox_padding = min(50, self.bbox_padding + 5)
                print(f"Bounding box expanded: scale={self.bbox_scale:.1f}, padding={self.bbox_padding}px")
            elif key == ord('-') or key == ord('_'):
                # Decrease bounding box size
                self.bbox_scale = max(1.0, self.bbox_scale - 0.1)
                self.bbox_padding = max(0, self.bbox_padding - 5)
                print(f"Bounding box reduced: scale={self.bbox_scale:.1f}, padding={self.bbox_padding}px")
                
        cv2.destroyAllWindows()
        if not self.tracking_complete:
            self.root.after(0, self.on_tracking_failed)
            
    def on_tracking_complete(self):
        """Called when key E is found."""
        self.status_label.config(text="Status: Key E Found!")
        self.tracking_label.config(text="Tracking: Complete - Window still open")
        self.record_btn.config(state=tk.NORMAL)
        messagebox.showinfo("Success", "Key E found!\nTracking window is still open for you to monitor.\nPress Q in the tracking window to close it.")
        
    def on_tracking_reset(self):
        """Called when tracking is reset."""
        self.status_label.config(text="Status: Tracking Reset")
        self.tracking_label.config(text="Tracking: Searching for Key E")
        self.record_btn.config(state=tk.DISABLED)
        
    def on_tracking_failed(self):
        """Called when key E tracking fails."""
        self.status_label.config(text="Status: Key E Not Found")
        self.tracking_label.config(text="Tracking: Failed or Stopped")
        self.start_tracking_btn.config(state=tk.NORMAL)
        self.close_tracking_btn.config(state=tk.DISABLED)
        messagebox.showwarning("Warning", "Key E not found or tracking stopped. Please try again.")
        
    def toggle_recording(self):
        """Toggle video recording."""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        """Start recording video in the key E bounding box area."""
        if self.key_e_bbox is None:
            messagebox.showerror("Error", "Key E not found. Please track first.")
            return
            
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"key_e_recording_{timestamp}.mp4"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Get bounding box dimensions
        x1, y1, x2, y2 = self.key_e_bbox.astype(int)
        width = x2 - x1
        height = y2 - y1
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        self.recording = True
        self.frame_count = 0
        
        # Update GUI
        self.record_btn.config(text="Stop Recording", bg="#f44336")
        self.status_label.config(text=f"Status: Recording to {output_filename}")
        
        # Start recording thread
        record_thread = threading.Thread(target=self.record_video)
        record_thread.daemon = True
        record_thread.start()
        
    def record_video(self):
        """Record video in the key E bounding box area."""
        x1, y1, x2, y2 = self.key_e_bbox.astype(int)
        
        while self.recording and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Crop frame to key E bounding box
            cropped_frame = frame[y1:y2, x1:x2]
            
            # Write frame to video
            if cropped_frame.size > 0:
                self.output_video.write(cropped_frame)
                self.frame_count += 1
                
                # Update status
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Status: Recording... Frame {self.frame_count}"))
            
            # Show the cropped frame
            cv2.imshow('Recording Key E Area', cropped_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.recording = False
                break
                
        # Clean up
        if self.output_video:
            self.output_video.release()
        cv2.destroyAllWindows()
        
        # Update GUI
        self.root.after(0, self.on_recording_complete)
        
    def on_recording_complete(self):
        """Called when recording is complete."""
        self.record_btn.config(text="Start Recording", bg="#2196F3")
        self.status_label.config(text=f"Status: Recording Complete! {self.frame_count} frames saved")
        messagebox.showinfo("Success", f"Recording complete!\nFrames recorded: {self.frame_count}\nVideo saved to: {self.output_dir}")
        
    def stop_recording(self):
        """Stop video recording."""
        self.recording = False
        
    def stop_all(self):
        """Stop all operations and close application."""
        self.recording = False
        self.tracking_complete = True
        self.tracking_active = False
        
        if self.output_video:
            self.output_video.release()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        self.root.quit()
        
    def run(self):
        """Run the application."""
        self.root.protocol("WM_DELETE_WINDOW", self.stop_all)
        self.root.mainloop()
        
    def expand_bbox(self, bbox, frame_shape):
        """Expand bounding box with padding and scaling."""
        x1, y1, x2, y2 = bbox
        
        # Calculate current dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Calculate center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Apply scale factor
        new_width = width * self.bbox_scale
        new_height = height * self.bbox_scale
        
        # Calculate new coordinates
        new_x1 = center_x - new_width / 2
        new_y1 = center_y - new_height / 2
        new_x2 = center_x + new_width / 2
        new_y2 = center_y + new_height / 2
        
        # Add padding
        new_x1 -= self.bbox_padding
        new_y1 -= self.bbox_padding
        new_x2 += self.bbox_padding
        new_y2 += self.bbox_padding
        
        # Ensure coordinates are within frame bounds
        frame_height, frame_width = frame_shape[:2]
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(frame_width, new_x2)
        new_y2 = min(frame_height, new_y2)
        
        return np.array([new_x1, new_y1, new_x2, new_y2])        
    
if __name__ == "__main__":
    app = KeyERecorder()
    app.run()