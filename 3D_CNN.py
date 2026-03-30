import os
import random
random.seed(42)

from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms

batch_size = 256
num_classes = 6
configuration = 0

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
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
    def __init__(self, root, informative_set, uninformative_set, configuration, transform=transform):
        self.root = root
        self.informative_set = informative_set
        self.uninformative_set = uninformative_set
        self.configuration = configuration
        self.transform = transform

        self.pairs = []
        self.build_pairs()

    def class_to_idx(self, class_name):
        classes = ["cone", "cube", "cylinder", "pyramid", "ring", "sphere"]
        return classes.index(class_name)

    def build_pairs(self):
        # This function builds pairs according to the passed configuration
        for class_name in self.informative_set.keys():
            # Unambiguous objects are always pairs of two informative views.
            if class_name not in self.uninformative_set:
                files = self.informative_set[class_name]

                # Create pairs and store them with their label
                for _ in range(len(files)):
                    a, b = random.sample(files, 2)
                    self.pairs.append((class_name, a, b))
            else:
                informative_files = self.informative_set[class_name]
                uninformative_files = self.uninformative_set[class_name]

                # Create pairs according to the passed configuration, and store them with their label
                for _ in range(len(informative_files)):
                    if self.configuration == 0:
                        a = random.choice(informative_files)
                        b = random.choice(informative_files)
                        self.pairs.append((class_name, a, b))
                    elif self.configuration == 1:
                        a = random.choice(informative_files)
                        b = random.choice(uninformative_files)
                        self.pairs.append((class_name, a, b))
                    else:
                        a = random.choice(uninformative_files)
                        b = random.choice(uninformative_files)
                        self.pairs.append((class_name, a, b))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Retrieve class name and two used files.
        class_name, file1, file2 = self.pairs[idx]

        # Retrieve images for unambiguous objects.
        if class_name not in self.uninformative_set:
            image1 = Image.open(os.path.join(self.root,class_name, file1)).convert("RGB")
            image2 = Image.open(os.path.join(self.root,class_name, file2)).convert("RGB")
        # Retrieve images for ambiguous objects.
        else:
            if self.configuration == 0:
                image1 = Image.open(os.path.join(self.root, class_name, "informative", file1)).convert("RGB")
                image2 = Image.open(os.path.join(self.root, class_name, "informative", file2)).convert("RGB")
            elif self.configuration == 1:
                image1 = Image.open(os.path.join(self.root, class_name, "informative", file1)).convert("RGB")
                image2 = Image.open(os.path.join(self.root, class_name, "uninformative", file2)).convert("RGB")
            else:
                image1 = Image.open(os.path.join(self.root, class_name, "uninformative", file1)).convert("RGB")
                image2 = Image.open(os.path.join(self.root, class_name, "uninformative", file2)).convert("RGB")

        # Transform objects according to passed transform.
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Stack both views, thus making the exemplar multidimensional.
        x = torch.stack([image1, image2], dim=1)
        label = self.class_to_idx(class_name)

        return x, label


def split_data():
    """
    This function splits the dataset into train and test sets for informative and uninformative views.
    :return: two dictionaries, one containing the different sets for all informative views,
    one containing the different sets for all uninformative views.
    """
    train_ratio = 0.8
    validation_ratio = 0.1

    # Retrieve all classes
    classes = os.listdir(os.path.join(base_directory, "samples"))
    informative_exemplars = {"train": {}, "validation": {}, "test": {}}
    uninformative_exemplars = {"train": {}, "validation": {}, "test": {}}

    for class_name in classes:
        class_directory = os.path.join(base_directory, "samples", class_name)

        # For unambigous shapes, only informative views are possible
        if class_name == "ring" or class_name == "sphere":
            informative_files = os.listdir(class_directory)
            random.shuffle(informative_files)

        else:
            # Retrieve and shuffle informative and uninformative views.
            uninformative_directory = os.path.join(class_directory, "uninformative")
            informative_directory = os.path.join(class_directory, "informative")

            uninformative_files = os.listdir(uninformative_directory)
            random.shuffle(uninformative_files)

            informative_files = os.listdir(informative_directory)
            random.shuffle(informative_files)

            # Split uninformative views in a training, validation and test set.
            train_uninformative_exemplars = uninformative_files[
                :int(len(uninformative_files) * train_ratio)
            ]
            validation_uninformative_exemplars = uninformative_files[
                int(len(uninformative_files) * train_ratio):int(len(uninformative_files) * (train_ratio + validation_ratio))
            ]
            test_uninformative_exemplars = uninformative_files[
                int(len(uninformative_files) * (train_ratio + validation_ratio)):
            ]

            # Store uninformative views in a dictionary according to the set and class they are part of.
            train_dictionary_uninformative = uninformative_exemplars["train"]
            train_dictionary_uninformative[class_name] = train_uninformative_exemplars
            uninformative_exemplars["train"] = train_dictionary_uninformative

            validation_dictionary_uninformative = uninformative_exemplars["validation"]
            validation_dictionary_uninformative[class_name] = validation_uninformative_exemplars
            uninformative_exemplars["validation"] = validation_dictionary_uninformative

            test_dictionary_uninformative = uninformative_exemplars["test"]
            test_dictionary_uninformative[class_name] = test_uninformative_exemplars
            uninformative_exemplars["test"] = test_dictionary_uninformative

        # Split informative views in a training, validation and test set.
        train_informative_exemplars = informative_files[
            :int(len(informative_files) * train_ratio)
        ]
        validation_informative_exemplars = informative_files[
            int(len(informative_files) * train_ratio):int(len(informative_files) * (train_ratio + validation_ratio))
        ]
        test_informative_exemplars = informative_files[
            int(len(informative_files) * (train_ratio + validation_ratio)):
        ]

        # Store informative views in a dictionary according to the set and class they are part of.
        train_dictionary_informative = informative_exemplars["train"]
        train_dictionary_informative[class_name] = train_informative_exemplars
        informative_exemplars["train"] = train_dictionary_informative

        validation_dictionary_informative = informative_exemplars["validation"]
        validation_dictionary_informative[class_name] = validation_informative_exemplars
        informative_exemplars["validation"] = validation_dictionary_informative

        test_dictionary_informative = informative_exemplars["test"]
        test_dictionary_informative[class_name] = test_informative_exemplars
        informative_exemplars["test"] = test_dictionary_informative

    # Return the dictionaries containing the informative and uninformative exemplars.
    return informative_exemplars, uninformative_exemplars

informative_exemplars, uninformative_exemplars = split_data()

# Create datasets
train_dataset = PairDataset(os.path.join(base_directory, "samples"),
                            informative_exemplars["train"],
                            uninformative_exemplars["train"],
                            configuration
                            )
validation_dataset = PairDataset(os.path.join(base_directory, "samples"),
                            informative_exemplars["validation"],
                            uninformative_exemplars["validation"],
                            configuration
                            )
test_dataset = PairDataset(os.path.join(base_directory, "samples"),
                            informative_exemplars["test"],
                            uninformative_exemplars["test"],
                            configuration
                            )

# Load the datasets
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


# Choose GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the chosen model
model = models.video.r3d_18(pretrained=True)

# Freeze early parameters
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

# Replace the final layer with a linear layer that maps to 6 nodes.
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Setup optimizer, loss function, training loop
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
train(model, train_dataloader, validation_dataloader, optimizer, criterion,
      device, num_epochs=1)

# Save trained model once finished
torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 'model_multiview.ckpt')

test(model, test_dataloader, device)