# evaluate.py
import torch
from sklearn.metrics import accuracy_score
from config import *
from data.dataset_loader import get_data_loaders
from model.resnet import build_resnet18
from utils.save_load import load_model
import logging

logging.basicConfig(level=logging.INFO)

_, test_loader = get_data_loaders(BATCH_SIZE)
model = build_resnet18()
load_model(model, MODEL_PATH)
model = model.to(DEVICE)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():                               # disable gradient calculation to save memory and computations
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
logging.info(f"Test Accuracy: {acc:.4f}")