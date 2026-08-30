import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.data_loader import get_dataloaders
from src.model import create_model, get_device


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = "checkpoints/best_model.pth"

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]


# ============================================================
# Setup
# ============================================================

device = get_device()

print(f"Device: {device}")

_, val_loader = get_dataloaders()

model = create_model()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# ============================================================
# Predictions
# ============================================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# Plot
# ============================================================

os.makedirs(
    "outputs/plots",
    exist_ok=True
)

plt.figure(
    figsize=(9, 7)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    cmap="Blues"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Diabetic Retinopathy Confusion Matrix")

plt.tight_layout()

output_path = "outputs/plots/confusion_matrix.png"

plt.savefig(
    output_path,
    dpi=300
)

plt.close()

print(
    f"\nConfusion matrix saved to: {output_path}"
)