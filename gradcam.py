import os

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms

num_classes = 6
model_type  = "3D"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

class_names = ["cone", "cube", "cylinder", "pyramid", "ring", "sphere"]

# Locate the base directory
base_directory = os.path.dirname(os.path.abspath(__file__))

def prepare_2D(image):
    """
    Prepares the image for PyTorch.

    : param image: an image
    :return: a processed and transformed image
    """
    image = transform(image)
    image = image.unsqueeze(0)

    return image

def prepare_3D(image1, image2):
    """
    Prepares two images for Pytorch.

    :param image1: an image
    :param image2: an image
    :return:
    """
    # Prepare image 1
    image1 = transform(image1)

    # Prepare image 2
    image2 = transform(image2)

    # Stack the two images
    prepared_images = torch.stack([image1, image2], dim=1)
    # Add the batch dimension that is lost for single exemplars
    prepared_images = prepared_images.unsqueeze(0)

    return prepared_images

def load_model(type):
    """
    Loads the passed model

    :param type: type of the model to be loaded
    :return: a loaded model
    """
    if type == "2D":
        checkpoint = torch.load(os.path.join(base_directory, "model_baseline.ckpt"))
        model = models.alexnet()
        model.classifier[6] = nn.Linear(4096, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = torch.load(os.path.join(base_directory, "model_multiview.ckpt"))
        model = models.video.r3d_18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    return model

def print_prediction(pred):
    pred_idx = pred.item()
    class_name = class_names[pred_idx]
    print(f"Prediction: {class_name} (class {pred_idx})")

def forwards_hook(module, input, output):
    """
    Adapted from https://towardsdatascience.com/grad-cam-from-scratch-with-pytorch-hooks/

    Parameters:
            module (nn.Module): The module where the hook is applied.
            input (tuple of Tensors): Input to the module.
            output (Tensor): Output of the module."""
    ...

def backwards_hook(module, grad_in, grad_out):

    """
    Adapted from https://towardsdatascience.com/grad-cam-from-scratch-with-pytorch-hooks/

    Parameters:
            module (nn.Module): The module where the hook is applied.
            grad_in (tuple of Tensors): Gradients w.r.t. the input of the module.
            grad_out (tuple of Tensors): Gradients w.r.t. the output of the module."""
    ...

# Replace all in-place ReLU activations with out-of-place ones
def replace_relu(model):

    for name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, name, torch.nn.ReLU(inplace=False))
            print(f"Replacing ReLU activation in layer: {name}")
        else:
            replace_relu(child)  # Recursively apply to submodules

# List to store activations
activations = []

# Function to save activations
def save_activations(module, input, output):
    # Adapted from https://towardsdatascience.com/grad-cam-from-scratch-with-pytorch-hooks/
    activations.append(output.detach().cpu().numpy().squeeze())

# List to store gradients
gradients = []

# Function to save gradients
def save_gradient(module, grad_in, grad_out):
    # Adapted from https://towardsdatascience.com/grad-cam-from-scratch-with-pytorch-hooks/
    gradients.append(grad_out[0].cpu().numpy().squeeze())

def compute_heatmap(model_type, model, image, layer):
    """
    Adapted from https://towardsdatascience.com/grad-cam-from-scratch-with-pytorch-hooks/
    A function to compute the GRADCAM heatmap

    :param model: The model to be used
    :param image: The image to be evaluated on
    :param layer: The layer to be evaluated on
    :return: A heatmap of the given image
    """
    if model_type == "2D":
        hook = model.features[layer].register_forward_hook(save_activations)
    else:
        # TO-DO: MAKE THE LAYER OF THE 3D MODEL SELECTABLE!
        layer = model.layer4[1].conv2
        hook = layer.register_forward_hook(save_activations)
    prediction = model(image)
    hook.remove()

    act_shape = np.shape(activations[0])
    print(f"Shape of activations: {act_shape}")  # (512, 14, 14)

    # Register the backward hook on a convolutional layer
    if model_type == "2D":
        hook = model.features[layer].register_full_backward_hook(save_gradient)
    else:
        #layer = model.layer4[1].conv2
        hook = layer.register_backward_hook(save_gradient)
    # Forward pass
    output = model(image)
    # Pick the class with highest score
    score = output[0].max()
    # Backward pass from the score
    score.backward()
    # Remove the hook after use
    hook.remove()

    print(gradients)
    # Obtain shape of the gradients
    grad_shape = np.shape(gradients[0])
    print(f"Shape of gradients: {grad_shape}")  # (512, 14, 14)
    model.zero_grad()

    # Aggregrate all gradients
    gradients_aggregated = np.mean(gradients[0], axis=(1, 2))
    # Weight the activations by the aggregated gradients and sum them up
    weighted_activations = np.sum(activations[0] *
                                gradients_aggregated[:, np.newaxis, np.newaxis],
                                axis=0)
    # Calculate ReLU summed activations
    relu_weighted_activations = np.maximum(weighted_activations, 0)

    return relu_weighted_activations

def upsampleHeatmap(model_type, relu_weighted_activations, image):
    """
    Adapted from https://towardsdatascience.com/grad-cam-from-scratch-with-pytorch-hooks/
    A function to upsample the GRADCAM heatmap

    :param relu_weighted_activations: the GRADCAM heatmap
    :param image: The image that was used to create the GRADCAM heatmap
    :return: The upsampled heatmap
    """

    if model_type == "2D":
        # Upsample the heatmap to the original image size
        upsampled_heatmap = cv2.resize(relu_weighted_activations,
                                    (image.size(3), image.size(2)),
                                    interpolation=cv2.INTER_LINEAR)

        print(np.shape(upsampled_heatmap))  # Should be (224, 224)
    else:
        print(relu_weighted_activations.shape)
        T = relu_weighted_activations.shape[0]
        H, W = image.shape[3], image.shape[4]

        upsampled_heatmap = []

        for t in range(T):
            frame = cv2.resize(
                relu_weighted_activations[t],
                (W, H),
                interpolation=cv2.INTER_LINEAR
            )
            upsampled_heatmap.append(frame)

        upsampled_heatmap = np.array(upsampled_heatmap)

        print(np.shape(upsampled_heatmap))

    return upsampled_heatmap

def display_images(upsampled_heatmap, original_image):
    """
    Show the upsampled heatmap and original image

    :param upsampled_heatmap: the upsampled heatmap
    :param original_image: the original image
    :return: None
    """
    # Visualise the heatmap
    fig, ax = plt.subplots(1, 2, figsize=(8, 8))

    # Input image
    resized_img = original_image.resize((224, 224))
    ax[0].imshow(resized_img)
    ax[0].axis("off")

    # Edge map for the input image
    edge_img = cv2.Canny(np.array(resized_img), 100, 200)
    ax[1].imshow(255 - edge_img, alpha=0.5, cmap='gray')

    # Overlay the heatmap
    ax[1].imshow(upsampled_heatmap, alpha=0.5, cmap='coolwarm')
    ax[1].axis("off")

    plt.show()

# Load the model of choice
loaded_model = load_model(model_type)
# Replace ReLU activations such that they don't switch
replace_relu(loaded_model)

# Prepare a test image
image_directory = os.path.join(base_directory, "samples", "ring", "ring_informative_23.png")
original_image = Image.open(image_directory).convert("RGB")
#test_image = prepare_2D(original_image)
image1_directory = os.path.join(base_directory, "samples", "cube", "uninformative", "cube_uninformative_23.png")
image2_directory = os.path.join(base_directory, "samples", "cube", "informative", "cube_informative_23.png")
original_image1 = Image.open(image1_directory).convert("RGB")
original_image2 = Image.open(image2_directory).convert("RGB")
test_image = prepare_3D(original_image1, original_image2)

# Obtain the relu weighted activations, upsample the heatmap and display it.
relu_weighted_activations = compute_heatmap(model_type, loaded_model, test_image, 10)
upsampled_heatmap = upsampleHeatmap(model_type, relu_weighted_activations, test_image)
display_images(upsampled_heatmap, original_image)