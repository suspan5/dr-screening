import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.data_loader import get_dataloaders
from src.model import create_model


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load validation data
_, val_loader = get_dataloaders()

# Create model
model = create_model()
checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)
model = model.to(device)
model.eval()

# Class names
class_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

all_predictions = []
all_labels = []

# Evaluation
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

# Classification report
print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4
    )
)

# Confusion matrix
print("\nCONFUSION MATRIX")
print("=" * 60)
print(confusion_matrix(all_labels, all_predictions))