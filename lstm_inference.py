import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

class KeyPressLSTM(nn.Module):
    """LSTM model for key press detection."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(KeyPressLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last time step output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc(last_output)
        return output

class RealTimeKeyPressDetector:
    """Real-time key press detection using YOLO + LSTM."""
    
    def __init__(self, yolo_model_path='best_v2.pt', lstm_model_path='best_lstm_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLO model for key detection
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize LSTM model
        self.input_size = 23  # Should match the training data
        self.lstm_model = KeyPressLSTM(self.input_size).to(self.device)
        
        # Load trained LSTM weights
        try:
            self.lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=self.device))
            self.lstm_model.eval()
            print("LSTM model loaded successfully!")
        except FileNotFoundError:
            print(f"LSTM model not found at {lstm_model_path}")
            print("Please train the model first using lstm_trainer.py")
            return
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            return
        
        # Initialize variables
        self.sequence_length = 30
        self.frame_buffer = deque(maxlen=self.sequence_length)
        self.predictions = deque(maxlen=100)  # Keep last 100 predictions
        self.key_e_bbox = None
        self.tracking_key_e = True
        
        # Setup camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
    def extract_frame_features(self, frame):
        """Extract features from a single frame."""
        if frame is None or frame.size == 0:
            return np.zeros((15,))
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Basic statistics
        features = [
            np.mean(gray),
            np.std(gray),
            np.max(gray),
            np.min(gray),
            np.median(gray)
        ]
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(grad_mag),
            np.std(grad_mag),
            np.max(grad_mag),
            np.min(grad_mag),
            np.median(grad_mag)
        ])
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-7)  # Normalize
        features.extend(hist.tolist())
        
        return np.array(features)
    
    def find_key_e(self, frame):
        """Find key E using YOLO model."""
        results = self.yolo_model(frame)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]
                    
                    if class_name.upper() == 'E':
                        confidence = float(box.conf[0])
                        if confidence > 0.5:  # Confidence threshold
                            self.key_e_bbox = box.xyxy[0].cpu().numpy()
                            self.tracking_key_e = False
                            return True
        return False
    
    def predict_key_press(self):
        """Predict key press using LSTM model."""
        if len(self.frame_buffer) < self.sequence_length:
            return 0.0
        
        # Convert buffer to numpy array
        sequence = np.array(list(self.frame_buffer))
        sequence = sequence.reshape(1, self.sequence_length, -1)  # Batch size 1
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.lstm_model(sequence_tensor)
            probability = float(prediction.cpu().numpy()[0][0])
        
        return probability
    
    def draw_visualization(self, frame, prediction_prob):
        """Draw visualization on frame."""
        if self.key_e_bbox is not None:
            x1, y1, x2, y2 = self.key_e_bbox.astype(int)
            
            # Draw bounding box
            color = (0, 255, 0) if prediction_prob > 0.5 else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw key press status
            status = "KEY PRESSED!" if prediction_prob > 0.5 else "Key Not Pressed"
            status_color = (0, 0, 255) if prediction_prob > 0.5 else (255, 255, 255)
            
            cv2.putText(frame, status, (x1, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Draw probability
            prob_text = f"Probability: {prediction_prob:.3f}"
            cv2.putText(frame, prob_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw FPS
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
        else:
            fps = 0
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw tracking status
        if self.tracking_key_e:
            cv2.putText(frame, "Tracking Key E...", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Key E Found - Detecting Press", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def run_prediction_plot(self):
        """Run live prediction plot in separate thread."""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        def update_plot(frame):
            ax.clear()
            
            if len(self.predictions) > 1:
                x_data = list(range(len(self.predictions)))
                y_data = list(self.predictions)
                
                ax.plot(x_data, y_data, 'g-', linewidth=2, alpha=0.8)
                ax.fill_between(x_data, y_data, alpha=0.3, color='green')
                
                # Draw threshold line
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.8, label='Threshold')
                
                # Highlight key presses
                for i, prob in enumerate(y_data):
                    if prob > 0.5:
                        ax.scatter(i, prob, color='red', s=50, alpha=0.8)
                
                ax.set_ylim(0, 1)
                ax.set_xlim(max(0, len(self.predictions) - 50), len(self.predictions))
                ax.set_ylabel('Key Press Probability')
                ax.set_xlabel('Frame')
                ax.set_title('Real-time Key E Press Detection')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            return ax
        
        ani = FuncAnimation(fig, update_plot, interval=50, blit=False)
        plt.show()
    
    def run(self):
        """Run the real-time detection."""
        print("Starting real-time key press detection...")
        print("Press 'q' to quit")
        print("Press 'p' to toggle prediction plot")
        
        # Start prediction plot in separate thread
        plot_thread = threading.Thread(target=self.run_prediction_plot)
        plot_thread.daemon = True
        plot_thread.start()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Track key E if not found yet
            if self.tracking_key_e:
                self.find_key_e(frame)
            
            # Process frame if key E is found
            if self.key_e_bbox is not None:
                # Crop frame to key E area
                x1, y1, x2, y2 = self.key_e_bbox.astype(int)
                cropped_frame = frame[y1:y2, x1:x2]
                
                if cropped_frame.size > 0:
                    # Extract features
                    features = self.extract_frame_features(cropped_frame)
                    self.frame_buffer.append(features)
                    
                    # Predict key press
                    prediction_prob = self.predict_key_press()
                    self.predictions.append(prediction_prob)
                    
                    # Draw visualization
                    frame = self.draw_visualization(frame, prediction_prob)
                    
                    # Show cropped frame
                    cv2.imshow('Key E Area', cropped_frame)
                else:
                    # If cropped frame is empty, show original frame
                    frame = self.draw_visualization(frame, 0.0)
            else:
                # Show original frame while tracking
                frame = self.draw_visualization(frame, 0.0)
            
            # Show main frame
            cv2.imshow('Real-time Key Press Detection', frame)
            
            # Update FPS counter
            self.fps_counter += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset tracking
                self.tracking_key_e = True
                self.key_e_bbox = None
                self.frame_buffer.clear()
                self.predictions.clear()
                print("Tracking reset")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function."""
    detector = RealTimeKeyPressDetector()
    detector.run()

if __name__ == "__main__":
    main()
