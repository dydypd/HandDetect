import cv2
import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

class KeyPressDataset(Dataset):
    """Dataset for key press detection from video sequences."""
    
    def __init__(self, sequences, labels, sequence_length=30):
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.FloatTensor([self.labels[idx]])
        return sequence, label

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

class VideoFeatureExtractor:
    """Extract features from video frames for LSTM training."""
    
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        
    def extract_optical_flow_features(self, frames):
        """Extract optical flow features from consecutive frames."""
        if len(frames) < 2:
            return np.zeros((10,))  # Return zeros if not enough frames
            
        # Convert to grayscale
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        
        # Calculate optical flow
        flow_features = []
        for i in range(1, len(gray_frames)):
            flow = cv2.calcOpticalFlowPyrLK(
                gray_frames[i-1], gray_frames[i], 
                None, None, 
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            if flow[0] is not None:
                # Calculate flow statistics
                flow_mag = np.sqrt(flow[0][:, :, 0]**2 + flow[0][:, :, 1]**2)
                flow_features.extend([
                    np.mean(flow_mag),
                    np.std(flow_mag),
                    np.max(flow_mag),
                    np.min(flow_mag),
                    np.median(flow_mag)
                ])
            else:
                flow_features.extend([0, 0, 0, 0, 0])
                
        return np.array(flow_features)
    
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
        hist = hist.flatten() / hist.sum()  # Normalize
        features.extend(hist.tolist())
        
        return np.array(features)
    
    def process_video(self, video_path, labels_data=None):
        """Process video file and extract features."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None, None
            
        frames = []
        frame_count = 0
        
        # Read all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1
            
        cap.release()
        
        if len(frames) < self.sequence_length:
            print(f"Warning: Video too short ({len(frames)} frames)")
            return None, None
            
        # Extract features for sequences
        sequences = []
        labels = []
        
        for i in range(len(frames) - self.sequence_length + 1):
            sequence_frames = frames[i:i + self.sequence_length]
            
            # Extract features for each frame in sequence
            sequence_features = []
            for frame in sequence_frames:
                frame_features = self.extract_frame_features(frame)
                sequence_features.append(frame_features)
            
            sequences.append(sequence_features)
            
            # Determine label (key pressed or not)
            if labels_data and isinstance(labels_data, dict):
                # Use actual labels from video labeler
                sequence_label = 0
                for frame_num in range(i, i + self.sequence_length):
                    if str(frame_num) in labels_data and labels_data[str(frame_num)] == 1:
                        sequence_label = 1
                        break
                labels.append(sequence_label)
            else:
                # For demonstration, create random labels
                # In real usage, you would have actual key press labels
                labels.append(np.random.randint(0, 2))
                
        return np.array(sequences), np.array(labels)
        
    def load_labeled_data(self, labels_file):
        """Load labeled data from video labeler."""
        try:
            with open(labels_file, 'r') as f:
                data = json.load(f)
            
            if 'frame_labels' in data:
                # Frame-by-frame labels
                return data['frame_labels'], data['video_path']
            elif 'sequences' in data:
                # Sequence labels
                sequence_labels = {}
                for seq in data['sequences']:
                    for frame_num in seq['frames']:
                        sequence_labels[str(frame_num)] = seq['label']
                return sequence_labels, data['video_path']
            else:
                # Old format
                return data['labels'], data['video_path']
                
        except Exception as e:
            print(f"Error loading labeled data: {e}")
            return None, None

class LSTMTrainer:
    """Train LSTM model for key press detection."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = KeyPressLSTM(input_size, hidden_size, num_layers, dropout).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (sequences, labels) in enumerate(dataloader):
            sequences, labels = sequences.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        return total_loss / len(dataloader), 100. * correct / total
    
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return total_loss / len(dataloader), 100. * correct / total
    
    def train(self, train_loader, val_loader, num_epochs=100):
        """Train the model."""
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_lstm_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def plot_training_curves(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """Plot training and validation curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(train_accuracies, label='Training Accuracy')
        ax2.plot(val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()

def main():
    """Main function to process videos and train LSTM model."""
    
    # Configuration
    video_folder = "recorded_videos"
    labels_folder = "training_data"
    sequence_length = 30
    
    # Initialize feature extractor
    extractor = VideoFeatureExtractor(sequence_length)
    
    # Process all videos in the folder
    all_sequences = []
    all_labels = []
    
    if not os.path.exists(video_folder):
        print(f"Error: Video folder '{video_folder}' not found.")
        print("Please record some videos first using video_record.py")
        return
    
    # Check for labeled data
    use_labeled_data = False
    if os.path.exists(labels_folder):
        labels_files = [f for f in os.listdir(labels_folder) if f.endswith('_frame_labels_*.json')]
        if labels_files:
            use_labeled_data = True
            print(f"Found {len(labels_files)} labeled data files")
        else:
            print("No labeled data found. Will use random labels for demonstration.")
    
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    
    if not video_files:
        print("No video files found in the recorded_videos folder.")
        return
    
    print(f"Processing {len(video_files)} video files...")
    
    if use_labeled_data:
        # Process videos with labeled data
        for labels_file in labels_files:
            labels_path = os.path.join(labels_folder, labels_file)
            print(f"Processing labeled data: {labels_file}")
            
            # Load labels
            labels_data, video_path = extractor.load_labeled_data(labels_path)
            
            if labels_data is None or video_path is None:
                print(f"Skipping {labels_file} due to loading error")
                continue
                
            # Check if video file exists
            if not os.path.exists(video_path):
                # Try to find video in recorded_videos folder
                video_name = os.path.basename(video_path)
                video_path = os.path.join(video_folder, video_name)
                
                if not os.path.exists(video_path):
                    print(f"Warning: Video file not found for {labels_file}")
                    continue
            
            print(f"Processing video: {video_path}")
            sequences, labels = extractor.process_video(video_path, labels_data)
            
            if sequences is not None and labels is not None:
                all_sequences.extend(sequences)
                all_labels.extend(labels)
                
                # Calculate statistics
                key_press_count = sum(labels)
                no_press_count = len(labels) - key_press_count
                print(f"Extracted {len(sequences)} sequences | Key press: {key_press_count} | No press: {no_press_count}")
            else:
                print(f"Failed to process {video_path}")
    else:
        # Process videos without labeled data (random labels)
        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)
            print(f"Processing {video_file}...")
            
            sequences, labels = extractor.process_video(video_path)
            
            if sequences is not None and labels is not None:
                all_sequences.extend(sequences)
                all_labels.extend(labels)
                print(f"Extracted {len(sequences)} sequences from {video_file}")
    
    if not all_sequences:
        print("No sequences extracted from videos.")
        return
    
    # Convert to numpy arrays
    all_sequences = np.array(all_sequences)
    all_labels = np.array(all_labels)
    
    print(f"\nTraining Data Summary:")
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Sequence shape: {all_sequences.shape}")
    print(f"Key press sequences: {sum(all_labels)}")
    print(f"No press sequences: {len(all_labels) - sum(all_labels)}")
    print(f"Data balance: {sum(all_labels)/len(all_labels)*100:.1f}% key press")
    
    # Check for data imbalance
    if sum(all_labels) == 0:
        print("Warning: No key press sequences found!")
        return
    elif sum(all_labels) == len(all_labels):
        print("Warning: All sequences are key press!")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        all_sequences, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f"\nTrain/Test Split:")
    print(f"Training sequences: {len(X_train)} | Key press: {sum(y_train)} | No press: {len(y_train) - sum(y_train)}")
    print(f"Test sequences: {len(X_test)} | Key press: {sum(y_test)} | No press: {len(y_test) - sum(y_test)}")
    
    # Create datasets
    train_dataset = KeyPressDataset(X_train, y_train, sequence_length)
    test_dataset = KeyPressDataset(X_test, y_test, sequence_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize trainer
    input_size = all_sequences.shape[2]  # Number of features per frame
    trainer = LSTMTrainer(input_size)
    
    print(f"\nModel Configuration:")
    print(f"Input size: {input_size}")
    print(f"Device: {trainer.device}")
    print(f"Using labeled data: {use_labeled_data}")
    
    # Train the model
    trainer.train(train_loader, test_loader, num_epochs=100)
    
    print("\nTraining completed!")
    print("Best model saved as 'best_lstm_model.pth'")
    print("Training curves saved as 'training_curves.png'")
    
    if use_labeled_data:
        print("\nNote: Training used labeled data from video_labeler.py")
    else:
        print("\nNote: Training used random labels. Use video_labeler.py to create proper labels.")

if __name__ == "__main__":
    main()
