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

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Locate the base directory
base_directory = os.path.dirname(os.path.abspath(__file__))

def load_model(type):
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

def compute_heatmap(model,img):
    """
    Adapted from https://medium.com/@bmuskan007/grad-cam-a-beginners-guide-adf68e80f4bb

    :param model:
    :param img:
    :return:
    """
    # compute logits from the model
    logits = model(img)
    # model's prediction
    pred = logits.max(-1)[-1]
    # activations from the model
    activations = image_to_heatmaps(img)
    print(activations)
    # compute gradients with respect to the model's most confident prediction
    logits[0, pred].backward(retain_graph=True)
    # average gradients of the featuremap
    # Change this line to inspect per layer?
    pool_grads = model.features[-3].weight.grad.data.mean((0,2,3))
    print(pool_grads)
    # multiply each activation map with corresponding gradient average
    # This should probably be changed for the 3D CNN
    for i in range(activations.shape[1]):
        activations[:,i,:,:] *= pool_grads[i]
    # calculate mean of weighted activations
    heatmap = torch.mean(activations, dim=1)[0].cpu().detach()
    return heatmap, pred

def upsampleHeatmap(map, image):
    """
    Adapted from https://medium.com/@bmuskan007/grad-cam-a-beginners-guide-adf68e80f4bb
    :param map:
    :param image:
    :return:
    """
    # permute image
    image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # maximum and minimum value from heatmap
    m, M = map.min(), map.max()
    # normalize the heatmap
    map = 255 * ((map-m)/ (m-M))
    map = np.uint8(map)
    # resize the heatmap to the same as the input
    map = cv2.resize(map, (224, 224))
    map = cv2.applyColorMap(255-map, cv2.COLORMAP_JET)
    map = np.uint8(map)
    # change this to balance between heatmap and image
    map = np.uint8(map*0.7 + image*0.3)
    return map

def display_images(upsampled_map, image):
    """
    Adapted from https://medium.com/@bmuskan007/grad-cam-a-beginners-guide-adf68e80f4bb

    :param upsampled_map:
    :param image:
    :return:
    """
    image = image.squeeze(0).permute(1, 2, 0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(upsampled_map)
    axes[0].set_title("Heatmap")
    axes[0].axis('off')
    axes[1].imshow(image)
    axes[1].set_title("Original Image")
    axes[1].axis('off')
    plt.show()

loaded_model = load_model("2D")

# selecting layers from the model to generate activations
image_to_heatmaps = nn.Sequential(*list(loaded_model.features[:-4]))

test_image = os.path.join(base_directory, "samples", "ring", "ring_informative_1.png")
test_image = Image.open(test_image).convert("RGB")
test_image = transform(test_image)

test_image = test_image.unsqueeze(0)
heatmap,pred = compute_heatmap(loaded_model,test_image)
upsampled_map = upsampleHeatmap(heatmap, test_image)
print(f"Prediction: {pred}")

display_images(upsampled_map, test_image)