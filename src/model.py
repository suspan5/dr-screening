import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


NUM_CLASSES = 5


def create_model():
    """
    Create MobileNetV2 with ImageNet pretrained weights
    and a 5-class classifier for diabetic retinopathy.
    """

    # Load pretrained MobileNetV2
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)

    # Replace the original 1000-class classifier
    # with our 5-class diabetic retinopathy classifier
    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        NUM_CLASSES
    )

    return model


def get_device():
    """Use NVIDIA GPU if available, otherwise CPU."""
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


if __name__ == "__main__":

    print("Creating MobileNetV2 model...")

    device = get_device()
    print("Device:", device)

    model = create_model()
    model = model.to(device)

    print("Model created successfully!")
    print("Number of classes:", NUM_CLASSES)

    # Test with a dummy image
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    print("\nModel test successful!")