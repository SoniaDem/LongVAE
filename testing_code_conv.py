"""
This code is just a tester for the VAE to load some of the data and make sure the VAE fits to the data.

Joe/Sonia 06/03/2023
"""
# ----------------------------- Load Packages --------------------------------------- 0.

import os
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from VAE.dataloader import BrainDataset3D
from VAE.models import VAE3d
from VAE.train import loss_fn, train_loop, eval_loop

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('0. Imported packages')
print(f'\tDevice: {device}')

# ----------------------------- Retrieve the image paths ---------------------------- 1.

out_path = 'D:\\ADNI_VAE\\CrossValidationFiles\\adni_5fold_all.csv'
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
trainLoader = DataLoader(train_data,
                         batch_size=batch,
                         shuffle=True)

valLoader = DataLoader(val_data,
                       batch_size=batch,
                       shuffle=True)

print(f'2. Data loaders have been created. Train: {len(trainLoader.dataset)} Validation: {len(valLoader)}')

# Note that you pass the data into the BrainDataset3d class, it changes the order of the axes.
# (batch, 0, 1, 2) to (batch, 2, 0, 1). Is this a problem

# ------------------------------------ File to store loss values -------------------------------------- 3.

cwd = os.getcwd()
loss_filename = 'loss_file_normal.txt'
loss_filename = os.path.join(cwd, loss_filename)

if os.path.isfile(loss_filename):
    print(f'\tThis file exists. Any training will append to this file\n\t\t{loss_filename}')

else:
    print(f'\tCreating file {loss_filename}')
    # 'w+' means write and add. If this file does not exist then this will create it.
    loss_file = open(loss_filename, 'w+')
    loss_file.close()

print('3. Create or loaded log file for loss data.')
# ----------------------------- Initialise Some Training Parameters ---------------------------- 4.

# Intialize model
model = VAE3d()
model = model.to(device)

model_path = 'D:\\ADNI_VAE\\Models\\DevModels\\'
model_name = f'VAE_full_data'
model_name = os.path.join(model_path, model_name)

model_list = [m for m in os.listdir(model_path) if model_name in m]
if len(model_list)>0:
    model_epochs = [int(m.split('_')[-1][:-3]) for m in model_list]
    pre_epochs = max(model_epochs)
    load_model = os.path.join('D:\\ADNI_VAE\\Models\\DevModels\\', model_list[model_epochs.index(pre_epochs)])
    model.load_state_dict(torch.load(load_model))
    print(f'\tLoaded model {load_model}')

else:
    print('\tInitialised new model')
    pre_epochs = 0

lr = 1e-5
optimizer = Adam(model.parameters(), lr=lr)
k_div = True
beta = 0.5
epochs = 5000
save_epochs = 1000
print('4. Training parameters initialised')


# ------------------------------------ Start training -------------------------------------- 5.

train_losses = []
val_losses = []
for epoch in range(1, epochs+1):

    train_loss = train_loop(model,
                            trainLoader,
                            optimizer,
                            k_div=k_div,
                            beta=beta,
                            print_batches=False,
                            device=device)

    val_loss = eval_loop(model,
                         valLoader,
                         k_div=k_div,
                         beta=beta,
                         device=device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f'[{epoch} / {epochs}]\ttrain: {train_loss :.10f}\tval: {val_loss :.10f}')

    if train_loss != train_loss or val_loss != val_loss:
        raise Exception('Loss is nan.')

    if epoch % save_epochs == 0:
        save_name = model_name + f'_{pre_epochs + epoch}.h5'
        torch.save(model.state_dict(), save_name)

    # 'a' means append
    loss_file = open(loss_filename, 'a')
    loss_file.write(f'\ntrain: {train_loss}')
    loss_file.write(f'\nval: {val_loss}')
    loss_file.close()


import nibabel as nib
import nibabel.processing as proc
import numpy as np

image = nib.load(train[0])
image = proc.resample_to_output(image, 4)
image = torch.Tensor(image.get_fdata(dtype='float64'))

