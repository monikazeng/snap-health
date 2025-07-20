import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES
'''
This module builds a ResNet-18 model for image classification.
ResNets original output layer was for 1,000 ImageNet classes.

It modifies the new Linear (fully connected) layer to output logits for 101 food classes.
This is necessary because:
    •	ResNet-18 is pretrained on ImageNet, which has 1,000 classes.
    •	We need to adapt it to our specific task of classifying food images into 101 categories.
    •	This is a common practice in transfer learning, where we fine-tune a pretrained model for a new task.
'''
# Loads a ResNet-18 model from torchvision.models, pretrained on ImageNet
def build_resnet18(pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model