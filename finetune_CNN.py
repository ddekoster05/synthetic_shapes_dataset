import os
import random

import numpy as np

random.seed(42)

from tqdm import tqdm
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms

batch_size = 5
num_classes = 6

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Locate the base directory
base_directory = os.path.dirname(os.path.abspath(__file__))

all_losses = []
all_accuracies = []


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

        #image1 = self.preprocess(image1)
        #image2 = self.preprocess(image2)

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


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the CNN classifier on the validation set.

    Args:
        model (CNN): CNN classifier to evaluate.
        test_loader (torch.utils.data.DataLoader): Data loader for the test set.
        criterion (callable): Loss function to use for evaluation.
        device (torch.device): Device to use for evaluation.

    Returns:
        float: Average loss on the test set.
        float: Accuracy on the test set.
    """
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        total_loss = 0.0
        num_correct = 0
        num_samples = 0

        for inputs, labels in test_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Compute the logits and loss
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Compute the accuracy
            _, predictions = torch.max(logits, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(inputs)

    # Evaluate the model on the validation set
    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples

    return avg_loss, accuracy


def train(model, train_loader, optimizer, criterion, device,
          num_epochs):
    """
    Train the CNN classifer on the training set and evaluate it on the validation set every epoch.

    Args:
    model (CNN): CNN classifier to train.
    train_loader (torch.utils.data.DataLoader): Data loader for the training set.
    val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
    optimizer (torch.optim.Optimizer): Optimizer to use for training.
    criterion (callable): Loss function to use for training.
    device (torch.device): Device to use for training.
    num_epochs (int): Number of epochs to train the model.
    """

    # Place the model on device
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch + 1}/{num_epochs}',
                  position=0,
                  leave=True) as pbar:
            for inputs, labels in train_loader:
                # Move inputs and labels to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Compute the logits and loss
                logits = model(inputs)
                # loss = torch.tensor(0)
                loss = criterion(logits, labels)

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

            # Compute average loss and accuracy on the validation dataset after one epoch
            avg_loss, accuracy = evaluate(model, train_loader, criterion, device)
            print(
                f'Training set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}'
            )
            all_losses.append(avg_loss)
            all_accuracies.append(accuracy)


# Create datasets
train_dataset = PairDataset(os.path.join(base_directory, "scraped_images"))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained model
checkpoint = torch.load(os.path.join(base_directory, "model_multiview_0.ckpt"))
model = models.video.r3d_18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Setup optimizer, loss function, training loop
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
train(model, train_dataloader, optimizer, criterion,
      device, num_epochs=10)

# Save trained model once finished
torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 'model_multiview_0_finetuned.ckpt')

# Plot the training loss
plt.clf()
epochs = np.arange(1,11)
plt.plot(epochs, all_losses)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('finetuning_training_loss.png')

# Plot the training accuracy
plt.clf()
epochs = np.arange(1,11)
plt.plot(epochs, all_accuracies)
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('finetuning_training_accuracy.png')