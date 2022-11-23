# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:04:33 2022

@author: user
"""
import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

from dcgan import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='model/model_360cnn_final.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=1, help='Number of generated outputs')
args = parser.parse_args()

path = 'model/model_360cnn_final.pth'
state_dict = torch.load(path,map_location='cpu')


# Set the device to run on: GPU or CPU.
# device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
device = torch.device("cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator(params).to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['generator'])

for i in range(5):
    noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)

	# Turn off gradient calculation to speed up the process.
    with torch.no_grad():
			# Get generated image from the noise vector using
			# the trained generator.
            generated_img = netG(noise).detach().cpu()
    # Display the generated image.
    plt.figure()
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(generated_img[0], (1,2,0)))
plt.show()
