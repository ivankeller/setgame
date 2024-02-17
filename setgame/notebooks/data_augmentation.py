# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python (myenv)
#     language: python
#     name: myenv
# ---

# %% [markdown]
# # Data augmentation
# Generate new images from diverse random transformations on images in a given directory.  
# Repeat the generation several times.  
# Save the results on sub-directories `aug{clycle number}` of parent directory `augmented_images_parent`.
#

# %%
# %pip install imgaug

# %%
import os
import imageio
from imgaug import augmenters as iaa
from imgaug import parameters as iap

# %%
raw_images_path = '/Users/ivankeller/Projects/setgame_project/data/1_segmented_cards/'
augmented_images_parent = '/Users/ivankeller/Projects/setgame_project/data/3_augmented_test/'

# %%
# Define your augmentation sequence
# You can customize this sequence with the transformations you need
crop_amount = 0.01
translation_amount = 0.01
rotation_amount = 5
shear_amount = 2
seq = iaa.Sequential([
    iaa.Crop(percent=(crop_amount, crop_amount, crop_amount, crop_amount)),  # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    iaa.Multiply((0.8, 1.2)),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-translation_amount, translation_amount), "y": (-translation_amount, translation_amount)},
        rotate=iap.Uniform(-rotation_amount, rotation_amount),  
        shear=(-shear_amount, shear_amount)
    ),
    iaa.PerspectiveTransform(scale=(0.05, 0.05))
], random_order=True)  # apply augmenters in random order

# %%
# %%time
# process each file in image directory and repeat a num_cycles times 
starting_cycle = 10  # id of the first cycle for the name of the sub directory
num_cycles = 10
for cycle in range(starting_cycle, starting_cycle + num_cycles):
    print(f'processing images, cycle {cycle}')
    counter = 0
    augmented_images_dir = os.path.join(augmented_images_parent, f'aug{cycle}')
    os.makedirs(augmented_images_dir, exist_ok=True)
    
    for filename in os.listdir(raw_images_path):
        if filename.endswith(".jpg"):
            # Construct the full file paths
            image_path = os.path.join(raw_images_path, filename)
            augmented_images_path = os.path.join(augmented_images_dir, filename)
    
            # Load the image
            image = imageio.v2.imread(image_path)
            images = [image]  # imgaug works on a batch of images, even if you have one
    
            # Perform the augmentation
            images_aug = seq(images=images)
    
            # Save the augmented image
            imageio.v2.imwrite(augmented_images_path, images_aug[0])
            counter += 1
        #if counter > 3:
        #    break

# %%
