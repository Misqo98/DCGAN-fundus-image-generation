
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

from utils import get_celeba
from dcgan import weights_init, Generator, Discriminator
from torchmetrics.image.fid import FrechetInceptionDistance
from fid import InceptionV3, calculate_fretchet

# smoothing class=1 to [0.7, 1.2]
def smooth_positive_labels(y):
	return y - 0.3 + (np.random.random(y.shape) * 0.5)

# smoothing class=0 to [0.0, 0.3]
def smooth_negative_labels(y):
	return y + np.random.random(y.shape) * 0.3


# Set random seed for reproducibility.
seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Parameters to define the model.
params = {
    "bsize" : 16,# Batch size during training.
    'imsize' : 360,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 3,# Number of channles in the training images. For coloured images this is 3.
    'nz' : 100,# Size of the Z latent vector (the input to the generator).
    'ngf' : 32,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs' : 200,# Number of training epochs.
    'lr_D' : 0.0002,# Learning rate for Discriminator optimizer
    'lr_G' : 0.0002,# Learning rate for Generator optimizer
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'save_epoch' : 5,# Save step.
    'fid_features':64,# Number of features for FID
    'save_run_name': 'DDR_02_fid_200epch'}


block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx])
model=model.cuda()

fid = FrechetInceptionDistance(feature=params['fid_features'])
# Use GPU is available else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

# Get the data.
dataloader = get_celeba(params)

#Plot the training images.
sample_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(
    sample_batch[0].to(device)[ : 64], padding=2, normalize=True).cpu(), (1, 2, 0)))

plt.show()

# Create the generator.
netG = Generator(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netG.apply(weights_init)
# Print the model.
print(netG)

# Create the discriminator.
netD = Discriminator(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netD.apply(weights_init)
# Print the model.
print(netD)

# Binary Cross Entropy loss function.
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

real_label = 1
fake_label = 0

# Optimizer for the discriminator.
optimizerD = optim.Adam(netD.parameters(), lr=params['lr_D'], betas=(params['beta1'], 0.999))
# Optimizer for the generator.
optimizerG = optim.Adam(netG.parameters(), lr=params['lr_G'], betas=(params['beta1'], 0.999))

# Stores generated images as training progresses.
img_list = []
# Stores generator losses during training.
G_losses = []
# Stores discriminator losses during training.
D_losses = []

D_x_log = []
D_G_z1_log = []
D_G_z2_log = []
fit_log = []

fid_min = 1000

iters = 0

print("Starting Training Loop...")
print("-"*25)

for epoch in range(params['nepochs']):
    batches_done = 0
    for i, data in enumerate(dataloader, 0):
        batches_done +=1
        # Transfer data tensor to GPU/CPU (device)
        real_data = data[0].to(device)
        # Get batch size. Can be different from params['nbsize'] for last batch in epoch.
        b_size = real_data.size(0)
        
        # Make accumalated gradients of the discriminator zero.
        netD.zero_grad()
        # Create labels for the real data. (label=1)
        label_np = np.full((b_size), real_label)
        # Smoothing the labels
        label_np = smooth_positive_labels(label_np)
        # Convert labels to tensor
        label = torch.from_numpy(label_np).float().to(device)
        #label = torch.full((b_size, ), real_label, device=device, dtype=torch.float)
        
        
        output = netD(real_data).view(-1)
        errD_real = criterion(output, label)
        # Calculate gradients for backpropagation.
        errD_real.backward()
        D_x = output.mean().item()
        
        # Sample random data from a unit normal distribution.
        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        # Generate fake data (images).
        fake_data = netG(noise)
        # Create labels for fake data. (label=0)
        label_np = np.full((b_size), fake_label)
        # Smoothing the labels
        label_np = smooth_negative_labels(label_np)
        # Convert labels to tensor
        label = torch.from_numpy(label_np).float().to(device)
        #label.fill_(fake_label  )
        # Calculate the output of the discriminator of the fake data.
        # As no gradients w.r.t. the generator parameters are to be
        # calculated, detach() is used. Hence, only gradients w.r.t. the
        # discriminator parameters will be calculated.
        # This is done because the loss functions for the discriminator
        # and the generator are slightly different.
        output = netD(fake_data.detach()).view(-1)
        errD_fake = criterion(output, label)
        # Calculate gradients for backpropagation.
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Net discriminator loss.
        errD = errD_real + errD_fake
        # Update discriminator parameters.
        optimizerD.step()
        
        # Make accumalted gradients of the generator zero.
        netG.zero_grad()
        # We want the fake data to be classified as real. Hence
        # real_label are used. (label=1)
        label_np = np.full((b_size), real_label)
        # Smoothing the labels
        label_np = smooth_positive_labels(label_np)
        # Convert labels to tensor
        label = torch.from_numpy(label_np).float().to(device)
        #label.fill_(real_label)
        # No detach() is used here as we want to calculate the gradients w.r.t.
        # the generator this time.
        output = netD(fake_data).view(-1)
        errG = criterion(output, label)
        # Gradients for backpropagation are calculated.
        # Gradients w.r.t. both the generator and the discriminator
        # parameters are calculated, however, the generator's optimizer
        # will only update the parameters of the generator. The discriminator
        # gradients will be set to zero in the next iteration by netD.zero_grad()
        errG.backward()

        D_G_z2 = output.mean().item()
        # Update generator parameters.
        optimizerG.step()
        
        
        
        # Check progress of training.
        if i%50 == 0:
            fretchet_dist=calculate_fretchet(real_data,fake_data,model)
            fit_log.append(fretchet_dist)
            print(torch.cuda.is_available())
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\t FID: %.4f'
                  % (epoch, params['nepochs'], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, fretchet_dist))

        # Save the losses for plotting.
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        D_x_log.append(D_x)
        D_G_z1_log.append(D_G_z1)
        D_G_z2_log.append(D_G_z2)
        

        # Check how the generator is doing by saving G's output on a fixed noise.
        if (iters % 100 == 0) or ((epoch == params['nepochs']-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake_data = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))

        iters += 1
        print("Epoch : {}, Batch : {}".format(epoch, batches_done))
        
    
    # Save the model.
    if epoch % params['save_epoch'] == 0:
        torch.save({
            'generator' : netG.state_dict(),
            'discriminator' : netD.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'optimizerD' : optimizerD.state_dict(),
            'params' : params
            }, 'model/model_360cnn_epoch_{}.pth'.format(epoch))
    if fretchet_dist < fid_min:
        print("Save best model at epoch " + str(epoch))
        fid_min = fretchet_dist
        torch.save({
            'generator' : netG.state_dict(),
            'discriminator' : netD.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'optimizerD' : optimizerD.state_dict(),
            'params' : params,
            'saved_epoch': epoch
            }, 'model/model_360cnn_best.pth')

# Save the final trained model.
torch.save({
            'generator' : netG.state_dict(),
            'discriminator' : netD.state_dict(),
            'optimizerG' : optimizerG.state_dict(),
            'optimizerD' : optimizerD.state_dict(),
            'params' : params
            }, 'model/model_360cnn_final.pth')

#Plot the training losses.
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig('loss_'+ params['save_run_name'] +'.jpg',format='jpg')

plt.figure(figsize=(10,5))
plt.title("Average discriminator output")
plt.plot(D_x_log,label="D(x)")
plt.plot(D_G_z2_log,label="D(G(z))")
plt.xlabel("Iteration")
plt.ylabel("Average output")
plt.legend()
plt.show()
plt.savefig('D_average_out_'+ params['save_run_name'] +'.jpg',format='jpg')

plt.figure(figsize=(10,5))
plt.title("Frechet inception distance")
plt.plot(fit_log,label="FID")
plt.xlabel("Iterations")
plt.ylabel("FID")
plt.legend()
plt.show()
plt.savefig('FID_'+ params['save_run_name'] +'.jpg',format='jpg')

# Animation showing the improvements of the generator.
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
plt.show()
anim.save('fundus_'+ params['save_run_name'] +'.gif', dpi=80, writer='imagemagick')
