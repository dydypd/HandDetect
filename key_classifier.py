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
            
        # Xác định device phù hợp
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        
        # Load weights với device phù hợp
        try:
            # Thử load model weights với map_location tới device
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Error loading model with GPU: {str(e)}")
            print("Falling back to CPU...")
            # Nếu lỗi, thử load lại với CPU
            self.device = torch.device('cpu')
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
        # Chuyển model sang device phù hợp và chế độ eval
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Class names
        self.class_names = ["No Press", "Key Press"]
        
    def _get_device(self):
        """Xác định device tốt nhất để sử dụng."""
        if torch.cuda.is_available():
            # Kiểm tra xem có GPU nào khả dụng không
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                # Lấy thông tin về GPU
                for i in range(gpu_count):
                    gpu_properties = torch.cuda.get_device_properties(i)
                    print(f"Found GPU {i}: {gpu_properties.name} "
                          f"({gpu_properties.total_memory / 1024**3:.1f}GB)")
                
                # Ưu tiên GPU có nhiều memory nhất
                max_memory = 0
                best_gpu = 0
                for i in range(gpu_count):
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory
                    if gpu_memory > max_memory:
                        max_memory = gpu_memory
                        best_gpu = i
                
                print(f"Selected GPU {best_gpu} as primary device")
                return torch.device(f'cuda:{best_gpu}')
        
        print("No GPU available, using CPU")
        return torch.device('cpu')
        
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
        
        # Chuẩn hóa theo ImageNet mean và std (như trong data transforms)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Chuyển từ HWC sang CHW format
        image = np.transpose(image, (2, 0, 1))
        
        # Xử lý dựa trên loại model
        if self.model_type == 'cnn_lstm':
            # Thêm sequence dimension
            image = np.expand_dims(image, 0)  # (1, C, H, W)
            image = np.expand_dims(image, 0)  # (1, 1, C, H, W) - batch, sequence, channels, height, width
        else:
            # Thêm batch dimension cho CNN và ResNet
            image = np.expand_dims(image, 0)  # (1, C, H, W)
        
        # Chuyển sang tensor và đưa lên device phù hợp
        image = torch.FloatTensor(image).to(self.device)
        return image
        
    def predict(self, image):
        """Dự đoán trạng thái của phím."""
        try:
            # Tiền xử lý ảnh
            image_tensor = self.preprocess_image(image)
            
            # Thực hiện dự đoán
            with torch.no_grad():
                outputs = self.model(image_tensor)
                
                # Xử lý kết quả dựa trên loại model
                if self.model_type == 'cnn_lstm':
                    # CNN_LSTM trả về kết quả cho sequence, lấy kết quả cuối cùng
                    probabilities = torch.nn.functional.softmax(outputs[:, -1] if len(outputs.shape) > 2 else outputs, dim=1)
                else:
                    # CNN và ResNet trả về kết quả trực tiếp
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
            return {
                'class_name': self.class_names[predicted_class],
                'class_id': predicted_class,
                'confidence': confidence * 100,  # Chuyển sang phần trăm
                'model_type': self.model_type,
                'model_accuracy': self.get_model_accuracy(),
                'device': str(self.device)
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU out of memory error: {str(e)}")
                print("Falling back to CPU...")
                # Chuyển sang CPU và thử lại
                self.device = torch.device('cpu')
                self.model = self.model.to(self.device)
                return self.predict(image)
            else:
                raise e

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