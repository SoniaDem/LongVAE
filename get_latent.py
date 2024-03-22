"""
In this code we pass all the data through the model, reparameterise and return the latent vector z.
For each subject this becomes a row in a csv.
If specified then this latent space can be reduced to 2d and plotted to see the distribution.

Joe/Sonia 06/03/2023
"""
# ----------------------------- Load Packages --------------------------------------- 0.

import os
import pandas as pd
import torch
import sys
import numpy as np
from torch.utils.data import DataLoader

from VAE.dataloader import BrainDataset3D, dataloader_to_z
from VAE.models import VAE3d
from VAE.plotting import plot_z

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('0. Imported packages')
print(f'\tDevice: {device}')

# ----------------------------- Retrieve the image paths ---------------------------- 1.

out_path = 'D:\\ADNI_VAE\\CrossValidationFiles\\adni_5fold.csv'
cv_split = pd.read_csv(out_path)

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
# shuffle is now set to false so that we can match the batches to the subjects
trainLoader = DataLoader(train_data,
                         batch_size=batch,
                         shuffle=False)

valLoader = DataLoader(val_data,
                       batch_size=batch,
                       shuffle=False)

print('2. Data loaders have been created.')

# ----------------------------- Load Model ---------------------------- 3.

model = VAE3d(128)
model = model.to(device)

model_path = 'D:\\ADNI_VAE\\Models\\DevModels\\'
# model_name = f'VAE_half_data_10000.h5'
model_name = f'VAE_half_data_{sys.argv[1]}.h5'
load_model = os.path.join(model_path, model_name)
model.load_state_dict(torch.load(load_model))
print(f'3. \tLoaded model {load_model}')

# ------------------------ Get Latent ----------------------------- 4.

print('4. Getting z')
loader = 'train'
save_dir = 'D:\\ADNI_VAE\\LatentSpaceFiles\\'
file_name = f'train_z_{sys.argv[1]}.csv'
# file_name = 'train_z_10000.csv'
file_path = os.path.join(save_dir, file_name)

if loader == 'train':
    # Convert the data from input to z.
    z_dims = np.array(dataloader_to_z(model, trainLoader, device))
    # Convert this to a numpy array in shape (inputs, z)
    z_dims_df = pd.DataFrame(z_dims)
    # Set the indices of each row to the specific file path.
    z_dims_df.index = train
else:
    z_dims = np.array(dataloader_to_z(model, valLoader, device))
    z_dims_df = pd.DataFrame(z_dims)
    z_dims_df.index = val

# Save the data to csv
z_dims_df.to_csv(file_path, index=True)
print(f'\tSaved {file_path}')

# ------------------------ Plot Latent ----------------------------- 5.

plot_latent = True
if plot_latent:
    plot_z(z_dims)








