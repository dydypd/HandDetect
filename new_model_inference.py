# Python
from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image

# Load YOLO detection model
yolo_model = YOLO('best_v2.pt')

# Load CNN classification model
cnn_model = torch.load('models/cnn.pth')
cnn_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def detect_and_classify(image_path):
    image = Image.open(image_path).convert('RGB')
    results = yolo_model(image_path)
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            # Replace 0 with the actual class index for "phím T"
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                cropped = image.crop((x1, y1, x2, y2))
                input_tensor = transform(cropped).unsqueeze(0)
                with torch.no_grad():
                    output = cnn_model(input_tensor)
                    predicted_label = output.argmax(1).item()
                return predicted_label
    return None

# Example usage
result = detect_and_classify('input_image.jpg')
print('Predicted label:', result)