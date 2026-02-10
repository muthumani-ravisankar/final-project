import torch
import torch.nn as nn
import timm
from PIL import Image
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Model (CHECKPOINT-ACCURATE)
# -------------------------------
class AdvancedImageClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0
        )

        in_features = self.backbone.num_features  # 1536

        self.classifier = nn.Sequential(
            nn.Identity(),                        # classifier.0 (NO WEIGHTS)

            nn.Linear(in_features, 1024),         # classifier.1
            nn.BatchNorm1d(1024),                 # classifier.2
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),                 # classifier.5
            nn.BatchNorm1d(512),                  # classifier.6
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),                  # classifier.9
            nn.BatchNorm1d(256),                  # classifier.10
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)           # classifier.13
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

# -------------------------------
# Load Model
# -------------------------------
model = AdvancedImageClassifier().to(device)

state_dict = torch.load("models/best_image_model.pth", map_location=device)
model.load_state_dict(state_dict)   # ✅ SHOULD LOAD CLEANLY
model.eval()

# -------------------------------
# Preprocessing
# -------------------------------
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

# -------------------------------
# Inference
# -------------------------------
img = Image.open("training_history.png").convert("RGB")
img = np.array(img)

x = transform(image=img)["image"].unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()

print("Probabilities:", probs.cpu().numpy())

print("Prediction:", "FAKE" if pred == 1 else "REAL")
