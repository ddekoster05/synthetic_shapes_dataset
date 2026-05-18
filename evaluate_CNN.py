import os
import random

random.seed(42)

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms

batch_size = 1
num_classes = 6

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# Locate the base directory
base_directory = os.path.dirname(os.path.abspath(__file__))


class PairDataset(Dataset):
    """
    This Dataset class provides pairs according to three possible configurations.
    0: two informative views
    1: one informative and one uninformative view
    2: two uninformative views

    For the unambiguous shapes, two informative views are used.
    """
    def __init__(self, root, transform=transform):
        self.root = root
        self.transform = transform

        self.classes = ["cone", "cube", "cylinder", "pyramid", "ring", "sphere"]

        self.pairs = []
        self.build_pairs()

    def preprocess(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply a Gaussian Blur
        blur = cv2.GaussianBlur(gray, (15, 15), 0)

        # smooth textures
        smooth = cv2.bilateralFilter(blur, 9, 75, 75)

        # posterize
        levels = 6
        poster = np.floor(smooth / (256 / levels)) * (256 / levels)

        # slight blur
        final = cv2.GaussianBlur(poster.astype(np.uint8), (3, 3), 0)

        final = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        #final = cv2.Canny(image=blur, threshold1=100, threshold2=200)
        #final = cv2.Laplacian(src=blur, ddepth=cv2.CV_64F, ksize=5)

        return final

    def class_to_idx(self, class_name):
        return self.classes.index(class_name)

    def build_pairs(self):
        # This function builds pairs according to the passed configuration
        for class_name in self.classes:
            class_dir = os.path.join(self.root, class_name)

            files = [
                f for f in os.listdir(class_dir)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]

            # Unambiguous objects are always pairs of two informative views
            for _ in range(len(files)):
                a = random.choice(files)
                b = random.choice(files)
                self.pairs.append((class_name, a, b))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Retrieve class name and two used files.
        class_name, file1, file2 = self.pairs[idx]

        # Retrieve images for unambiguous objects.
        image1 = Image.open(os.path.join(self.root,class_name, file1)).convert("RGB")
        image2 = Image.open(os.path.join(self.root,class_name, file2)).convert("RGB")

        image1 = np.asarray(image1)
        image2 = np.asarray(image2)

        image1 = self.preprocess(image1)
        image2 = self.preprocess(image2)

        image1 = Image.fromarray(image1).convert("RGB")
        image2 = Image.fromarray(image2).convert("RGB")

        # Transform objects according to passed transform.
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Stack both views, thus making the exemplar multidimensional.
        x = torch.stack([image1, image2], dim=1)
        label = self.class_to_idx(class_name)

        return x, label

def test(model, test_loader, device):
    """
    Test the model accuracy per class.

    """
    model.eval()
    classes = ["cone", "cube", "cylinder", "pyramid", "ring", "sphere"]

    # Initialize variables
    per_class_correct = {cls: 0 for cls in classes}
    per_class_total = {cls: 0 for cls in classes}

    num_samples = 0
    num_correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Calculate predictions, and compare them to ground truth.
            logits = model(inputs)
            _, preds = torch.max(logits, dim=1)
            num_correct += (preds == labels).sum().item()
            num_samples += labels.size(0)

            # Update per-class counters
            for cls_idx, cls_name in enumerate(classes):
                cls_mask = labels == cls_idx
                per_class_correct[cls_name] += (preds[cls_mask] == labels[cls_mask]).sum().item()
                per_class_total[cls_name] += cls_mask.sum().item()

    # Calculate overall and per class accuracy
    overall_accuracy = num_correct / num_samples
    per_class_accuracy = {cls: per_class_correct[cls] / per_class_total[cls]
                          for cls in classes}

    # Report accuracy scores
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print("Per-class accuracy:")
    for cls, acc in per_class_accuracy.items():
        print(f"{cls}: {acc:.4f}")

# Create datasets
test_dataset = PairDataset(os.path.join(base_directory, "scraped_images"))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained model
checkpoint = torch.load(os.path.join(base_directory, "model_multiview_0_sobel.ckpt"))
model = models.video.r3d_18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

test(model, test_dataloader, device)