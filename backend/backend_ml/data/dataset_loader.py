import os
from torchvision.datasets import Food101
from torchvision import transforms
from torch.utils.data import DataLoader
from config import DATA_DIR

'''
Download and Preprocess the Food-101 Dataset

Food-101 dataset: contains 101,000 images of food across 101 categories (e.g., pizza, sushi, steak, etc.).
'''

def get_data_loaders(batch_size=32):
    # Resize images to 224x224 pixels and convert them to tensors
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Neural networks work better with normalized input, and PyTorch models require torch.Tensor format
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  

    food101_path = os.path.join(DATA_DIR, "food-101")
    download = not os.path.exists(food101_path)

    train_data = Food101(root=DATA_DIR, split="train", download=download, transform=transform)
    test_data = Food101(root=DATA_DIR, split="test", download=download, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader
