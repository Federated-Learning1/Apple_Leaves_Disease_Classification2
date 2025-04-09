# model.py
import timm
import torch.nn as nn

def get_model(num_classes=3):
    model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
