import os
from PIL import Image

base_directory = os.path.dirname(os.path.abspath(__file__))
alpha = 0.3

def blend_image(alpha, image_1_directory, image_2_directory):
    # Open the images
    image_1 = Image.open(image1_directory).convert("RGB")
    image_2 = Image.open(image2_directory)

    # Make sure dimensions of images match
    image_1 = image_1.resize(image_2.size)

    # Blend image and save it
    blended = Image.blend(image_1, image_2, alpha=alpha)
    blended.save("blended.png")

# Retrieve the directories of the images that have to be blended
image1_directory = os.path.join(base_directory, "paired_images", "custom_shape.png")
image2_directory = os.path.join(base_directory, "paired_images", "1.png")

blend_image(alpha, image1_directory, image2_directory)