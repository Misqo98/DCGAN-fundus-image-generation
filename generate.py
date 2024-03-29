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

# Load the checkpoint file.
for epoch in range(150,200,5):
    path = f'./model/model_360cnn_epoch_{epoch}.pth'
    state_dict = torch.load(path,map_location='cpu')
    print("Generating images for model : {}".format(epoch))

	# Set the device to run on: GPU or CPU.
	# device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    device = torch.device("cpu")
	# Get the 'params' dictionary from the loaded state_dict.
    params = state_dict['params']

	# Create the generator network.
    netG = Generator(params).to(device)
	# Load the trained generator weights.
    netG.load_state_dict(state_dict['generator'])
	# print(netG)

	# print(args.num_output)
	# Get latent vector Z from unit normal distribution.
    for i in range(2):
        noise = torch.randn(int(args.num_output), params['nz'], 1, 1, device=device)

		# Turn off gradient calculation to speed up the process.
        with torch.no_grad():
			# Get generated image from the noise vector using
			# the trained generator.
            generated_img = netG(noise).detach().cpu()

		# Display the generated image.
        plt.figure()
        plt.axis("off")
        plt.title(f"{i} Images generated for {epoch} Epoch")
        plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))
        #plt.savefig(f'./generated_images/pic_generated_idrid_lrelu_label_smth_{epoch}_{i}.jpg',format='jpg')
        #plt.imsave(f'./generated_images/pic_generated_{epoch}_{i}.jpg', np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))
        print(f"{i} Images generated for {epoch} Epoch")

    plt.show()
