import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.model import create_model, get_device
from src.gradcam import GradCAM


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
# Load model
# ============================================================

device = get_device()

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
# Grad-CAM
# ============================================================

target_layer = model.features[-1]

gradcam = GradCAM(
    model,
    target_layer
)

# ============================================================
# Clinical screening guidance
# ============================================================

SCREENING_GUIDANCE = {
    "No DR": {
        "severity": "No diabetic retinopathy detected",
        "urgency": "Routine",
        "referral": "Continue routine diabetic eye screening.",
    },

    "Mild": {
        "severity": "Mild diabetic retinopathy",
        "urgency": "Routine follow-up",
        "referral": "Recommend routine ophthalmic follow-up.",
    },

    "Moderate": {
        "severity": "Moderate diabetic retinopathy",
        "urgency": "Ophthalmology review",
        "referral": "Recommend ophthalmologist evaluation.",
    },

    "Severe": {
        "severity": "Severe diabetic retinopathy",
        "urgency": "Prompt referral",
        "referral": "Prompt ophthalmologist referral is recommended.",
    },

    "Proliferative DR": {
        "severity": "Proliferative diabetic retinopathy",
        "urgency": "Urgent referral",
        "referral": "Urgent ophthalmologist evaluation is recommended.",
    }
}

# ============================================================
# Main inference function
# ============================================================

def predict_image(image):

    """
    Run DR classification + Grad-CAM on an image.

    Parameters
    ----------
    image : PIL.Image.Image
        Retinal fundus image.

    Returns
    -------
    dict
        Prediction results and Grad-CAM overlay.
    """

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    # Prepare tensor
    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

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

    predicted_label = CLASS_NAMES[
        predicted_class
    ]

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    cam = gradcam.generate(
        input_tensor,
        predicted_class
    )

    # Convert original image
    original = np.array(image)

    original = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    height, width = original.shape[:2]

    # Resize CAM
    cam_resized = cv2.resize(
        cam,
        (width, height)
    )

    # Heatmap
    heatmap = np.uint8(
        255 * cam_resized
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Overlay
    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    # Convert BGR → RGB
    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Probability dictionary
    # --------------------------------------------------------

    probabilities_dict = {}

    for i, class_name in enumerate(CLASS_NAMES):

        probabilities_dict[class_name] = float(
            probabilities[0, i].item()
        )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "prediction": predicted_label,
        "class_id": predicted_class,
        "confidence": float(confidence),
        "probabilities": probabilities_dict,
        "severity": SCREENING_GUIDANCE[predicted_label]["severity"],
        "urgency": SCREENING_GUIDANCE[predicted_label]["urgency"],
        "referral": SCREENING_GUIDANCE[predicted_label]["referral"],
        "gradcam": overlay
    }


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("Testing inference pipeline...")
    print("Device:", device)

    test_image_path = (
        "data/raw/aptos/train_images/000c1434d8d7.png"
    )

    image = Image.open(
        test_image_path
    )

    result = predict_image(
        image
    )

    print("\nPrediction:")
    print(result["prediction"])

    print(
        f"\nConfidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    print("\nProbabilities:")

    for class_name, probability in result[
        "probabilities"
    ].items():

        print(
            f"{class_name}: "
            f"{probability * 100:.2f}%"
        )

    print("\nScreening result:")
    print("Severity:", result["severity"])
    print("Urgency:", result["urgency"])
    print("Referral:", result["referral"])

    print(
        "\nDisclaimer: This system is intended for "
        "screening and decision support and does not "
        "replace professional medical diagnosis."
    )

    print("\nInference test successful!")