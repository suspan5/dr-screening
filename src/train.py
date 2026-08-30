import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score, classification_report

# Allow imports from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import get_dataloaders
from src.model import create_model, get_device


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

NUM_CLASSES = 5

CHECKPOINT_PATH = "checkpoints/best_model.pth"


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(train_loader):
    """
    Calculate inverse-frequency class weights.

    This helps the model pay more attention to minority classes
    such as Severe and Proliferative DR.
    """

    labels = []

    for _, batch_labels in train_loader:
        labels.extend(batch_labels.tolist())

    counts = torch.bincount(
        torch.tensor(labels),
        minlength=NUM_CLASSES
    ).float()

    weights = 1.0 / counts

    # Normalize weights so their average is approximately 1
    weights = weights / weights.mean()

    return weights


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION
# ============================================================

def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1,
        all_labels,
        all_predictions
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("DIABETIC RETINOPATHY MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print(f"\nDevice: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        gpu_memory = torch.cuda.get_device_properties(0).total_memory

        print(
            f"GPU Memory: {gpu_memory / (1024 ** 3):.2f} GB"
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_loader, val_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating MobileNetV2...")

    model = create_model()

    model = model.to(device)

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    print("\nCalculating class weights...")

    class_weights = calculate_class_weights(
        train_loader
    )

    class_weights = class_weights.to(device)

    print(
        "Class weights:",
        class_weights.detach().cpu().numpy()
    )

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_macro_f1 = 0.0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": []
    }

    print("\nStarting training...")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{NUM_EPOCHS}"
        )

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        (
            val_loss,
            val_accuracy,
            val_macro_f1,
            val_labels,
            val_predictions
        ) = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        print(
            f"Val Macro F1: {val_macro_f1:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_macro_f1"].append(val_macro_f1)

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = val_macro_f1

            os.makedirs(
                "checkpoints",
                exist_ok=True
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_accuracy,
                    "val_macro_f1": val_macro_f1,
                    "epoch": epoch + 1,
                },
                CHECKPOINT_PATH
            )

            print(
                f"✓ Best model saved: {CHECKPOINT_PATH}"
            )

    os.makedirs("outputs", exist_ok=True)

    import json

    with open("outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print("Training history saved to: outputs/training_history.json")
    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FINAL VALIDATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            val_labels,
            val_predictions,
            labels=[0, 1, 2, 3, 4],
            target_names=[
                "No DR",
                "Mild",
                "Moderate",
                "Severe",
                "Proliferative DR"
            ],
            zero_division=0
        )
    )

    print("=" * 60)
    print("Training complete!")
    print(f"Best Macro F1: {best_macro_f1:.4f}")
    print(f"Model saved to: {CHECKPOINT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()