import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from torchvision import models
from torchvision import transforms

batch_size = 256
num_classes = 6

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Locate the base directory
base_directory = os.path.dirname(os.path.abspath(__file__))

# Find the dataset, split it, and load it in.
dataset = ImageFolder(root=os.path.join(base_directory, "samples"), transform=transform)
train_dataset, validation_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


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


def train(model, train_loader, val_loader, optimizer, criterion, device,
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
            avg_loss, accuracy = evaluate(model, val_loader, criterion, device)
            print(
                f'Validation set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}'
            )

def get_view_type_from_path(path):
    if "uninformative" in path:
        return "uninformative"
    else:
        return "informative"


def test(model, dataset, test_dataset, device):
    """
    Test the model accuracy for informative and uninformative views separately

    Args:
    :param model:
    :param test_loader:
    :param device:
    """

    with torch.no_grad():

        # Initialize variables
        informative_correct = 0
        informative_samples = 0

        uninformative_correct = 0
        uninformative_samples = 0

        for i in test_dataset.indices:
            # Retrieve path and label from the dataset, only for the test set
            path, label = dataset.samples[i]

            # Load and transform the image and label, and move to device
            img = dataset.loader(path)
            img = dataset.transform(img).unsqueeze(0).to(device)
            label = torch.tensor([label]).to(device)

            # Compute the logits and loss
            logits = model(img)
            prediction = torch.argmax(logits, dim=1)

            # Count the prediction as correct if it is identical to the label
            if prediction == label:
                correct = 1
            else:
                correct = 0

            # Determine if an uninformative or informative view was used
            view_type = get_view_type_from_path(path)

            # Keep track of performance
            if view_type == "informative":
                informative_samples += 1
                informative_correct += correct
            else:
                uninformative_samples += 1
                uninformative_correct += correct

    # Calculate accuracy
    informative_accuracy = informative_correct / informative_samples
    uninformative_accuracy = uninformative_correct / uninformative_samples

    # Report accuracy
    print(f"Informative accuracy   : {informative_accuracy:.4f}, based on {informative_samples} samples")
    print(f"Uninformative accuracy : {uninformative_accuracy:.4f}, based on {uninformative_samples} samples")

# Choose GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the chosen model
model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)

# Freeze early parameters
for param in model.features.parameters():
    param.requires_grad = False

# Replace the final layer with a linear layer that maps to 6 nodes.
model.classifier[6] = nn.Linear(4096, num_classes)

# Setup optimizer, loss function, training loop
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
train(model, train_dataloader, validation_dataloader, optimizer, criterion,
      device, num_epochs=3)

# Save trained model once finished
torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 'model_baseline.ckpt')

test(model, dataset, test_dataset, device)