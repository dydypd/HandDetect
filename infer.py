import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from torchvision import transforms

import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

class ImprovedCNN_LSTM(nn.Module):
    def __init__(self, cnn_feature_size, lstm_hidden_size, num_layers, num_classes, dropout=0.3):
        super(ImprovedCNN_LSTM, self).__init__()

        # CNN backbone with residual connections
        self.cnn_backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Feature projection with regularization
        self.cnn_output_size = 512 * 4 * 4
        self.feature_projection = nn.Sequential(
            nn.Linear(self.cnn_output_size, cnn_feature_size),
            nn.BatchNorm1d(cnn_feature_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM with layer normalization
        self.lstm = nn.LSTM(
            input_size=cnn_feature_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Classification head
        lstm_output_size = lstm_hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.BatchNorm1d(lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        # CNN feature extraction
        x = x.view(batch_size * seq_len, c, h, w)
        cnn_features = self.cnn_backbone(x)
        cnn_features = cnn_features.view(batch_size * seq_len, -1)

        # Project features
        projected_features = self.feature_projection(cnn_features)
        lstm_input = projected_features.view(batch_size, seq_len, -1)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(lstm_input)

        # Self-attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling over sequence
        final_features = attended_out.mean(dim=1)

        # Classification
        logits = self.classifier(final_features)

        return logits


def train_with_debugging(model, train_loader, val_loader, num_epochs, device, learning_rate=0.001):
    # Weighted loss to handle class imbalance
    class_counts = Counter()
    for _, labels in train_loader:
        for label in labels:
            class_counts[label.item()] += 1

    weights = []
    total_samples = sum(class_counts.values())
    for i in range(len(class_counts)):
        weights.append(total_samples / (len(class_counts) * class_counts[i]))

    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for sequences, labels in train_pbar:
            sequences, labels = sequences.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_train_acc = 100 * train_correct / train_total
        epoch_val_acc = 100 * val_correct / val_total

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')

        # Learning rate scheduling
        scheduler.step(epoch_val_acc)

        # Early stopping and model saving
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'  New best model saved! Val Acc: {best_val_acc:.2f}%')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

        print()

    return train_losses, val_losses, train_accuracies, val_accuracies, all_predictions, all_labels


# 1. Load mô hình đã huấn luyện
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model_on_video(model_path, video_path, num_classes, sequence_length=6, device='cuda'):
    # Load the trained model
    model = ImprovedCNN_LSTM(
        cnn_feature_size=512,
        lstm_hidden_size=256,
        num_layers=2,
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a transformation for preprocessing
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create a buffer to store sequence of frames
    frame_buffer = deque(maxlen=sequence_length)
    predictions = []

    # Process the video frame by frame
    frame_idx = 0
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = preprocess(frame_rgb).unsqueeze(0)

            # Add to buffer
            if len(frame_buffer) < sequence_length:
                # Fill buffer initially
                for _ in range(sequence_length - len(frame_buffer)):
                    frame_buffer.append(processed_frame)
            frame_buffer.append(processed_frame)

            # Make prediction every N frames or at the end
            if frame_idx % 8 == 0 or frame_idx == frame_count - 1:
                # Stack frames into a sequence
                sequence = torch.cat(list(frame_buffer), dim=0).unsqueeze(0).to(device)  # [1, seq_len, C, H, W]

                # Forward pass
                outputs = model(sequence)
                _, predicted = torch.max(outputs.data, 1)

                # Get prediction
                pred_class = predicted.item()
                predictions.append(pred_class)

                # Display prediction on frame
                class_name = f"Class {pred_class}"  # Replace with actual class names if available
                cv2.putText(frame, f"Prediction: {class_name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the frame
                cv2.imshow('Video Prediction', frame)

                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Calculate prediction statistics
    if predictions:
        pred_counts = Counter(predictions)
        print("Prediction statistics:")
        for class_id, count in pred_counts.items():
            print(f"Class {class_id}: {count} frames ({count / len(predictions) * 100:.2f}%)")

        # Most frequent prediction
        most_common = pred_counts.most_common(1)[0]
        print(f"\nMost frequent prediction: Class {most_common[0]} ({most_common[1] / len(predictions) * 100:.2f}%)")

    return predictions


# Example usage
if __name__ == "__main__":
    model_path = "final_improved_model.pth"  # Path to your trained model
    video_path = "demi.mp4"  # Path to your test video
    num_classes = 25  # Replace with your actual number of classes

    predictions = test_model_on_video(model_path, video_path, num_classes)