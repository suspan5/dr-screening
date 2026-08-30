import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.model import create_model, get_device


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = "checkpoints/best_model.pth"

OUTPUT_DIR = "outputs/gradcam"

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

IMAGE_SIZE = 224


# ============================================================
# Grad-CAM implementation
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self.save_activation
        )

        self.backward_hook = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):

        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_index):

        self.model.zero_grad()

        output = self.model(image_tensor)

        score = output[:, class_index]

        score.backward()

        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted combination
        cam = (weights * activations).sum(
            dim=1,
            keepdim=True
        )

        cam = torch.relu(cam)

        # Remove batch/channel dimensions
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam

    def remove_hooks(self):

        self.forward_hook.remove()
        self.backward_hook.remove()


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Load model
# ============================================================

device = get_device()

print("Device:", device)

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

print("Model loaded successfully.")


# ============================================================
# Target layer
# ============================================================

# Last convolutional feature layer of MobileNetV2
target_layer = model.features[-1]

gradcam = GradCAM(
    model,
    target_layer
)


# ============================================================
# Select an image
# ============================================================

# We'll automatically select the first APTOS image.
# Later the frontend will provide the image dynamically.

image_dir = "data/raw/aptos/train_images"

image_files = [
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

if len(image_files) == 0:

    raise FileNotFoundError(
        "No images found in data/raw/aptos/train_images"
    )

image_path = os.path.join(
    image_dir,
    image_files[0]
)

print("Using image:", image_path)


# ============================================================
# Load image
# ============================================================

original_image = Image.open(
    image_path
).convert("RGB")

input_tensor = transform(
    original_image
).unsqueeze(0).to(device)


# ============================================================
# Prediction
# ============================================================

with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

confidence = probabilities[
    0,
    predicted_class
].item()

print(
    f"Prediction: {CLASS_NAMES[predicted_class]}"
)

print(
    f"Confidence: {confidence * 100:.2f}%"
)


# ============================================================
# Generate Grad-CAM
# ============================================================

cam = gradcam.generate(
    input_tensor,
    predicted_class
)


# ============================================================
# Prepare images
# ============================================================

original = np.array(
    original_image
)

original = cv2.cvtColor(
    original,
    cv2.COLOR_RGB2BGR
)

height, width = original.shape[:2]

cam_resized = cv2.resize(
    cam,
    (width, height)
)

heatmap = np.uint8(
    255 * cam_resized
)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

overlay = cv2.addWeighted(
    original,
    0.6,
    heatmap,
    0.4,
    0
)


# ============================================================
# Save outputs
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

cv2.imwrite(
    f"{OUTPUT_DIR}/original.png",
    original
)

cv2.imwrite(
    f"{OUTPUT_DIR}/heatmap.png",
    heatmap
)

cv2.imwrite(
    f"{OUTPUT_DIR}/overlay.png",
    overlay
)


print("\nGrad-CAM generated successfully!")

print(
    f"Original: {OUTPUT_DIR}/original.png"
)

print(
    f"Heatmap: {OUTPUT_DIR}/heatmap.png"
)

print(
    f"Overlay: {OUTPUT_DIR}/overlay.png"
)

gradcam.remove_hooks()