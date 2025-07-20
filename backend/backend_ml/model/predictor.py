import torch
import torchvision.transforms as transforms
from PIL import Image

from efficientnet_pytorch import EfficientNet


from backend_ml.model.resnet_model import build_resnet18
from backend_ml.utils.save_load import load_model
from backend_ml.utils.class_labels import FOOD_CLASSES
from backend_ml.config import MODEL_PATH


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Predict with the ResNet-18 model
def predict_ResNet_18(image_path):
    model = build_resnet18(pretrained=False)
    model = load_model(model, MODEL_PATH)
    model.eval()

    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()
        return {
            "food": FOOD_CLASSES[class_idx],
            "class_id": class_idx,
        }


# def predict_food_sample(image_path: str):
#     img = Image.open(image_path)

#     # dummy response for now
#     return {
#         "food": "pizza",
#         "calories": 266,
#         "fat": 10,
#         "carbs": 33,
#         "protein": 11
#     }