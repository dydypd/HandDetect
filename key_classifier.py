import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18
import numpy as np
import cv2
import json
import os

class CNNModel(nn.Module):
    """Pure CNN model for single frame classification."""
    def __init__(self, num_classes=2):
        super(CNNModel, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CNNLSTMModel(nn.Module):
    """CNN+LSTM model combining spatial and temporal features."""
    def __init__(self, num_classes=2):
        super(CNNLSTMModel, self).__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(128, 64, num_layers=2, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 128),  # *2 for bidirectional
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Đối với single frame prediction, thêm dim giả cho sequence
        if len(x.shape) == 4:  # (batch, channel, height, width)
            x = x.unsqueeze(1)  # (batch, seq=1, channel, height, width)
            
        batch_size, seq_len, c, h, w = x.shape
        
        # Extract CNN features for each frame
        cnn_features = []
        for i in range(seq_len):
            frame_features = self.cnn(x[:, i, :, :, :])
            frame_features = frame_features.view(batch_size, -1)
            cnn_features.append(frame_features)
        
        # Stack features for LSTM
        cnn_features = torch.stack(cnn_features, dim=1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(cnn_features)
        
        # Use last output
        x = lstm_out[:, -1, :]
        
        # Classifier
        x = self.classifier(x)
        return x

class ResNetModel(nn.Module):
    """ResNet model for robust feature extraction."""
    def __init__(self, num_classes=2):
        super(ResNetModel, self).__init__()
        
        # Load ResNet without pretrained weights
        self.backbone = resnet18(weights=None)
        
        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class KeyClassifier:
    def __init__(self, model_type='resnet', model_path=None):
        """
        Khởi tạo classifier với loại model được chọn
        model_type: 'resnet', 'cnn', 'cnn_lstm'
        """
        self.model_type = model_type
        if model_path is None:
            model_path = f'models/{model_type}.pth'
            
        # Load model info
        self.model_info = self._load_model_info()
        
        # Khởi tạo model dựa trên loại
        if model_type == 'resnet':
            self.model = ResNetModel()
        elif model_type == 'cnn_lstm':
            self.model = CNNLSTMModel()
        elif model_type == 'cnn':
            self.model = CNNModel()
        else:
            raise ValueError(f"Model type {model_type} not supported yet")
            
        # Load weights
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Chuyển model sang CPU mode
        self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
        
        # Class names
        self.class_names = ["No Press", "Key Press"]
        
    def _load_model_info(self):
        """Load thông tin về các model từ file json."""
        info_path = 'models/model_info.json'
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
                return info
        return None
        
    def get_model_accuracy(self):
        """Trả về độ chính xác của model hiện tại."""
        if self.model_info and 'models' in self.model_info:
            if self.model_type in self.model_info['models']:
                return self.model_info['models'][self.model_type]['test_accuracy']
        return None
        
    def preprocess_image(self, image):
        """Tiền xử lý ảnh trước khi đưa vào model."""
        # Đảm bảo ảnh có kích thước 64x64
        if image.shape[:2] != (64, 64):
            image = cv2.resize(image, (64, 64))
            
        # Chuẩn hóa ảnh về dạng tensor
        image = image.astype(np.float32) / 255.0
        # Chuyển từ HWC sang CHW format
        image = np.transpose(image, (2, 0, 1))
        # Thêm batch dimension
        image = np.expand_dims(image, 0)
        # Chuyển sang tensor
        image = torch.FloatTensor(image)
        return image
        
    def predict(self, image):
        """Dự đoán trạng thái của phím."""
        # Tiền xử lý ảnh
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Thực hiện dự đoán
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
        return {
            'class_name': self.class_names[predicted_class],
            'class_id': predicted_class,
            'confidence': confidence * 100,  # Chuyển sang phần trăm
            'model_type': self.model_type,
            'model_accuracy': self.get_model_accuracy()
        }

    @staticmethod
    def get_available_models():
        """Trả về danh sách các model có sẵn."""
        models_dir = 'models'
        available_models = []
        
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.pth') and not file.endswith('_complete.pth'):
                    model_name = file.replace('.pth', '')
                    available_models.append(model_name)
                    
        return available_models

# Ví dụ sử dụng
if __name__ == "__main__":
    # Lấy danh sách model có sẵn
    available_models = KeyClassifier.get_available_models()
    print(f"Available models: {available_models}")
    
    # Khởi tạo classifier với model mặc định
    classifier = KeyClassifier()
    
    # Load một ảnh test
    image = cv2.imread('test_key_image.jpg')
    if image is not None:
        # Dự đoán
        result = classifier.predict(image)
        print(f"Model: {result['model_type']}")
        print(f"Model accuracy: {result['model_accuracy']:.2f}%")
        print(f"Predicted class: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.2f}%") 