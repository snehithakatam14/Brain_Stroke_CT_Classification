import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os

device = torch.device("cpu")

# ----- Load Model -----
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load("best_stroke_model.pth", map_location=device))
model.eval()
model.to(device)

# Remove final classification layer
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()

# ----- Transform -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----- Load Your TRAIN Dataset Folder -----
# IMPORTANT: Change this to your training dataset path
dataset_path = r"C:\Users\SNEHITHA\OneDrive\Documents\Brain_Stroke_CT_Dataset\Train"

dataset = datasets.ImageFolder(dataset_path, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

all_features = []

with torch.no_grad():
    for images, _ in loader:
        images = images.to(device)
        features = feature_extractor(images)
        features = features.view(features.size(0), -1)
        all_features.append(features.cpu().numpy())

all_features = np.concatenate(all_features, axis=0)

# ----- Compute Mean & Covariance -----
mean = np.mean(all_features, axis=0)
cov = np.cov(all_features, rowvar=False)

# Regularization for numerical stability
cov += np.eye(cov.shape[0]) * 1e-6

cov_inv = np.linalg.inv(cov)

np.save("feature_mean.npy", mean)
np.save("feature_cov_inv.npy", cov_inv)

print("Feature statistics saved successfully.")