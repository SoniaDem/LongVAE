# ----------------------------- Load Packages --------------------------------------- 0.

import os
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from VAE.dataloader import BrainDataset3D
from VAE.models import VAE_GAN, get_latest_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('0. Imported packages')
print(f'\tDevice: {device}')

# ----------------------------- Retrieve the image paths ---------------------------- 1.

cv_path = 'D:\\ADNI_VAE\\CrossValidationFiles\\adni_5fold_all.csv'
cv_split = pd.read_csv(cv_path)

fold = 0
train = cv_split[cv_split[f'fold_{fold}'] == 'train']['extracted_paths'].tolist()
val = cv_split[cv_split[f'fold_{fold}'] == 'val']['extracted_paths'].tolist()

print(f'1. Number of brains in train:  {len(train)}')
print(f'   Number of brains in val:    {len(val)}')

# ----------------------------- Load data into Data Loader ---------------------------- 2.

# Define some transformations. It must at least convert the image to a tensor
rescale_factor = 4
filetype = 'pt'
train_data = BrainDataset3D(train,
                            filetype=filetype,
                            scale=rescale_factor)

val_data = BrainDataset3D(val,
                          filetype=filetype,
                          scale=rescale_factor)

batch = 32
trainLoader = DataLoader(train_data,
                         batch_size=batch,
                         shuffle=True)

valLoader = DataLoader(val_data,
                       batch_size=batch,
                       shuffle=True)

print('2. Data loaders have been created.')

# Note that you pass the data into the BrainDataset3d class, it changes the order of the axes.
# (batch, 0, 1, 2) to (batch, 2, 0, 1). Is this a problem

# ------------------------------------ File to store loss values -------------------------------------- 3.

cwd = os.getcwd()
loss_filename = 'loss_file.txt'
loss_filename = os.path.join(cwd, loss_filename)
rewrite_loss = True

if os.path.isfile(loss_filename) and not rewrite_loss:
    print(f'\tThis file exists. Any training will append to this file\n\t\t{loss_filename}')

else:
    print(f'\tCreating file {loss_filename}')
    # 'w+' means write and add. If this file does not exist then this will create it.
    loss_file = open(loss_filename, 'w+')
    loss_file.close()

print('3. Create or loaded log file for loss data.')
# ----------------------------- Initialise Some Training Parameters ---------------------------- 4.

# Intialize model
vae_gan = VAE_GAN()

model_path = 'D:\\ADNI_VAE\\Models\\DevModels\\'
vae_name = f'VAEGAN_vae_'
vae_path = os.path.join(model_path, vae_name)

d_name = f'VAEGAN_d_'
d_path = os.path.join(model_path, d_name)


vae_m, pre_epochs = get_latest_model(model_path, vae_name)
if vae_m is not None:
    vae_m = os.path.join(model_path, vae_m)
    vae_gan.VAE.load_state_dict(torch.load(vae_m))
    print(f'\tLoaded VAE model: {vae_m}')
else:
    print('\tInitialised new VAE model')
    pre_epochs = 0


vae_d, pre_epochs = get_latest_model(model_path, d_name)
if vae_d is not None:
    vae_d = os.path.join(model_path, vae_d)
    vae_gan.discriminator.load_state_dict(torch.load(vae_d))
    print(f'\tLoaded discriminator model: {vae_d}')
else:
    print('\tInitialised new discriminator model')
    pre_epochs = 0


vae_lr = 1e-5
d_lr = 1e-6
epochs = 5000
vae_optimizer = Adam(vae_gan.VAE.parameters(), lr=vae_lr)
d_optimizer = Adam(vae_gan.discriminator.parameters(), lr=d_lr)
print('4. Training parameters initialised')

# ----------------------------- Choo Choo Tren ---------------------------- 5.

vae_gan.beta = 5
vae_gan.gamma = 0.05
vae_gan.VAE.z_dim = 128
losses, d_losses, v_losses = vae_gan.train(trainLoader,
                                           valLoader,
                                           vae_optimizer,
                                           d_optimizer,
                                           loss_txt=loss_filename,
                                           num_epochs=epochs,
                                           pre_epochs=pre_epochs,
                                           eval_epochs=1,
                                           save=model_path)


vae_save_name = vae_path + f'{epochs + pre_epochs}.h5'
torch.save(vae_gan.VAE.state_dict(), vae_save_name)

d_save_name = d_path + f'{epochs + pre_epochs}.h5'
torch.save(vae_gan.discriminator.state_dict(), d_save_name)


## This is for plotting the discriminator loss.
# from VAE.plotting import plot_loss
# plot_loss(v_losses)
