import os
import random
random.seed(42)

from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.io import read_image
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
            image1 = read_image(os.path.join(self.root,class_name, file1))
            image2 = read_image(os.path.join(self.root,class_name, file2))
        # Retrieve images for ambiguous objects.
        else:
            if self.configuration == 0:
                image1 = read_image(os.path.join(self.root, class_name, "informative", file1))
                image2 = read_image(os.path.join(self.root, class_name, "informative", file2))
            elif self.configuration == 1:
                image1 = read_image(os.path.join(self.root, class_name, "informative", file1))
                image2 = read_image(os.path.join(self.root, class_name, "uninformative", file2))
            else:
                image1 = read_image(os.path.join(self.root, class_name, "uninformative", file1))
                image2 = read_image(os.path.join(self.root, class_name, "uninformative", file2))

        # Transform objects according to passed transform.
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Stack both views, thus making the exemplar multidimensional.
        x = torch.stack([image1, image2], dim=0)
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
                            0
                            )
validation_dataset = PairDataset(os.path.join(base_directory, "samples"),
                            informative_exemplars["validation"],
                            uninformative_exemplars["validation"],
                            0
                            )
test_dataset = PairDataset(os.path.join(base_directory, "samples"),
                            informative_exemplars["test"],
                            uninformative_exemplars["test"],
                            0
                            )

# Load the datasets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)