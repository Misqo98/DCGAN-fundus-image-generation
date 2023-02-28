# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:35:27 2023

@author: user
"""

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image

# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True

def Load_Image(Path):
    img = cv.imread(Path)[:,:,::-1] # opencv read the images in BGR format 
                                    # so we use [:,:,::-1] to convert from BGR to RGB
    return img

def Show_Image(Image, Picture_Name):
    plt.imshow(Image)
    plt.title(Picture_Name)
    plt.show()
    
    
    
def Show_Image(Image, Picture_Name):
    plt.imshow(Image)
    plt.title(Picture_Name)
    plt.show()
    
image_example1 = Load_Image('007-5971-300.jpg')
img = Image.open('007-5971-300.jpg')
imgTensor = transforms.ToTensor()(img)
earsing_transform = transforms.RandomErasing(p=1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False)  
earsing_transform_random = transforms.RandomErasing(p=1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random', inplace=False)
    
rotate_Transformation = transforms.Compose([
   transforms.ToPILImage(), # the transform usually work with PIL images
   transforms.RandomRotation(degrees=(33,66)),
])

translate_x_Transformation = transforms.Compose([
   transforms.ToPILImage(), # the transform usually work with PIL images
   transforms.RandomAffine(translate=(0.4,0), degrees=0),
])

translate_y_Transformation = transforms.Compose([
   transforms.ToPILImage(), # the transform usually work with PIL images
   transforms.RandomAffine(translate=(0,0.4), degrees=0),
])

h_flip_Transformation = transforms.Compose([
   transforms.ToPILImage(), # the transform usually work with PIL images
   transforms.RandomHorizontalFlip(p=1),
])

v_flip_Transformation = transforms.Compose([
   transforms.ToPILImage(), # the transform usually work with PIL images
   transforms.RandomVerticalFlip(p=1),
])

crop_Transformation = transforms.Compose([
   transforms.ToPILImage(), # the transform usually work with PIL images
   transforms.FiveCrop(size=(500,500)),
])



brightness_Transformation = transforms.Compose([
   transforms.ToPILImage(), # the transform usually work with PIL images
   transforms.ColorJitter(brightness=(2, 2.5)),
])

contrast_Transformation = transforms.Compose([
   transforms.ToPILImage(), # the transform usually work with PIL images
   transforms.ColorJitter(contrast=(2, 4)),
])

hue_Transformation = transforms.Compose([
   transforms.ToPILImage(), # the transform usually work with PIL images
   transforms.ColorJitter(hue=(0.01, 0.1)),
])

saturation_Transformation = transforms.Compose([
   transforms.ToPILImage(), # the transform usually work with PIL images
   transforms.ColorJitter(saturation=(1.5, 2)),
])

blur_Transformation = transforms.Compose([
   transforms.ToPILImage(), # the transform usually work with PIL images
  transforms.GaussianBlur(kernel_size=(13, 27), sigma=(48,50)),
])

# Testing The Transformation...
earsed_Transformation = earsing_transform(imgTensor)
earsed_Transformation_random = earsing_transform_random(imgTensor)

rotated_example1 = rotate_Transformation(image_example1)
translated_x_example1 = translate_x_Transformation(image_example1)
translated_y_example1 = translate_y_Transformation(image_example1)
h_flip_example1 = h_flip_Transformation(image_example1)
v_flip_example1 = v_flip_Transformation(image_example1)

example1_brightness = brightness_Transformation(image_example1)
example1_contrast = contrast_Transformation(image_example1)
example1_hue = hue_Transformation(image_example1)
example1_saturation = saturation_Transformation(image_example1)
example1_blur =  blur_Transformation(image_example1)



plt.figure(0)
plt.subplot(2, 3, 1)
plt.imshow(image_example1)
plt.axis('off')
plt.title('Originál')


plt.subplot(2, 3, 2)
plt.imshow(rotated_example1)
plt.axis('off')
plt.title('Rotácia')

plt.subplot(2, 3, 3)
plt.imshow(translated_x_example1)
plt.axis('off')
plt.title('Translácia (x)')

plt.subplot(2, 3, 4)
plt.imshow(translated_y_example1)
plt.axis('off')
plt.title('Translácia (y)')

plt.subplot(2, 3, 5)
plt.imshow(h_flip_example1)
plt.axis('off')
plt.title('Prevrátanie (x)')


plt.subplot(2, 3, 6)
plt.imshow(v_flip_example1)
plt.axis('off')
plt.title('Prevrátenie (y)')


plt.show()

img_earsed = transforms.ToPILImage()(earsed_Transformation)
img_earsed_random = transforms.ToPILImage()(earsed_Transformation_random)

plt.figure(1)
plt.subplot(1, 3, 1)
plt.imshow(image_example1)
plt.axis('off')
plt.title('Originál')

plt.subplot(1, 3, 2)
plt.imshow(img_earsed)
plt.axis('off')
plt.title('Vymazanie 1')

plt.subplot(1, 3, 3)
plt.imshow(img_earsed_random)
plt.axis('off')
plt.title('Vymazanie 2')


plt.figure(2)
plt.subplot(2, 3, 1)
plt.imshow(image_example1)
plt.axis('off')
plt.title('Originál')

plt.subplot(2, 3, 2)
plt.imshow(example1_brightness)
plt.axis('off')
plt.title('Jas')

plt.subplot(2, 3, 3)
plt.imshow(example1_contrast)
plt.axis('off')
plt.title('Kontrast')

plt.subplot(2, 3, 4)
plt.imshow(example1_hue)
plt.axis('off')
plt.title('Odtieň')

plt.subplot(2, 3, 5)
plt.imshow(example1_saturation)
plt.axis('off')
plt.title('Saturácia')

plt.subplot(2, 3, 6)
plt.imshow(example1_blur)
plt.axis('off')
plt.title('Rozmazanie')


