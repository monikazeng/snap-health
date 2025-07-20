import torch
import os

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 101
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "resnet_food101.pth")
DATA_DIR = "backend_ml/data"

EN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "efficientnet_b0_food101.pth")