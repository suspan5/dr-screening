import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# APTOS DATASET
# ============================================================

class APTOSDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_id = row["id_code"]
        label = int(row["diagnosis"])

        # APTOS images are .png
        image_path = os.path.join(self.image_dir, image_id + ".png")

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATALOADERS
# ============================================================

def get_dataloaders(
    train_csv="data/processed/train.csv",
    val_csv="data/processed/val.csv",
    image_dir="data/raw/aptos/train_images",
    batch_size=32,
    num_workers=0
):
    train_dataset = APTOSDataset(
        csv_file=train_csv,
        image_dir=image_dir,
        transform=train_transform
    )

    val_dataset = APTOSDataset(
        csv_file=val_csv,
        image_dir=image_dir,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    print("Testing APTOS DataLoader...")

    train_loader, val_loader = get_dataloaders()

    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))

    images, labels = next(iter(train_loader))

    print("Batch image shape:", images.shape)
    print("Batch labels shape:", labels.shape)
    print("Labels:", labels[:10].tolist())

    print("\nDataLoader test successful!")