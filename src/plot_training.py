import json
import os
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Load training history
# ---------------------------------------------------------

with open("outputs/training_history.json", "r") as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"]) + 1)


# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------

os.makedirs("outputs/plots", exist_ok=True)


# ---------------------------------------------------------
# Loss curve
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "outputs/plots/loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Accuracy curve
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    "outputs/plots/accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Macro F1 curve
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["val_macro_f1"],
    marker="o",
    label="Validation Macro F1"
)

plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.title("Validation Macro F1")
plt.legend()
plt.grid(True)

plt.savefig(
    "outputs/plots/macro_f1_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Training plots generated successfully!")

print("\nSaved:")
print("outputs/plots/loss_curve.png")
print("outputs/plots/accuracy_curve.png")
print("outputs/plots/macro_f1_curve.png")