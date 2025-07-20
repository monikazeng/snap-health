# train.py
import torch
from config import *
from data.dataset_loader import get_data_loaders
from model.resnet_model import build_resnet18
from utils.save_load import save_model
import os, logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def train():
    if os.path.exists(MODEL_PATH):
        logging.info(f"Model already exists at {MODEL_PATH}. Skipping training.")
        return

    logging.info("Starting training")
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Batch size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")

    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    logging.info("Loaded training and test data")

    model = build_resnet18().to(DEVICE)
    logging.info("Model built and moved to device")

    criterion = torch.nn.CrossEntropyLoss()     # measures prediction error for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()                   # clears old gradients from the previous batch
            outputs = model(images)                 # forward pass: compute predicted outputs by passing inputs to the model    
            loss = criterion(outputs, labels)       # calculates how far off the predictions are from the actual labels
            loss.backward()                         # backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()                        # applies those gradients to update the model weights
            running_loss += loss.item()             # loss.item() converts a PyTorch scalar tensor into a Python float so it can be added up

            # Log every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                logging.info(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = running_loss / len(train_loader)
        logging.info(f"Finished training -- epoch [{epoch+1}/{NUM_EPOCHS}] finished -- Final avg loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    save_model(model, MODEL_PATH)
    logging.info(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
    logging.info("Training completed successfully")