# ----------------------------- Load Packages --------------------------------------- 0.

import os
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from VAE.dataloader import LongBrainDataset3D
from VAE.models import VAE3d

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

# ----------------------------- Retrieve the image paths ---------------------------- 2.
# Get patient data
pat_df = pd.read_csv('C:\\Users\\Sonia\\Documents\\R\\ADNI_FPCA\\ADNI_df_paths.csv')

# ----------------------------- Load data into Data Loader ---------------------------- 3.

# Define some transformations. It must at least convert the image to a tensor
rescale_factor = 4
filetype = 'pt'
train_data = LongBrainDataset3D(train,
                                pat_df,
                                filetype=filetype,
                                scale=rescale_factor)

val_data = LongBrainDataset3D(val,
                              pat_df,
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
