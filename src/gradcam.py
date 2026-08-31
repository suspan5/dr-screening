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

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (weights * activations).sum(
            dim=1,
            keepdim=True
        )

        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam

    def remove_hooks(self):

        self.forward_hook.remove()
        self.backward_hook.remove()


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
# Generate Grad-CAM for an image
# ============================================================

def generate_gradcam(image_path):

    """
    Generate prediction and Grad-CAM visualization
    for a dynamically supplied image.

    Parameters
    ----------
    image_path : str
        Path to the input retinal image.

    Returns
    -------
    prediction : str
        Predicted diabetic retinopathy class.

    confidence : float
        Prediction confidence.

    original : numpy.ndarray
        Original image in RGB format.

    heatmap : numpy.ndarray
        Grad-CAM heatmap in RGB format.

    overlay : numpy.ndarray
        Heatmap overlaid on the original image.
    """

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    input_tensor = transform(
        original_image
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    # Grad-CAM requires gradients, so do NOT use torch.no_grad()
    # around this prediction.

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

    prediction = CLASS_NAMES[
        predicted_class
    ]

    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    cam = gradcam.generate(
        input_tensor,
        predicted_class
    )

    # --------------------------------------------------------
    # Prepare original image
    # --------------------------------------------------------

    original = np.array(
        original_image
    )

    # OpenCV uses BGR
    original_bgr = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    height, width = original_bgr.shape[:2]

    # --------------------------------------------------------
    # Resize CAM
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = cv2.addWeighted(
        original_bgr,
        0.6,
        heatmap,
        0.4,
        0
    )

    # Convert OpenCV BGR → RGB
    heatmap_rgb = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay_rgb = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return (
        prediction,
        confidence,
        original,
        heatmap_rgb,
        overlay_rgb
    )