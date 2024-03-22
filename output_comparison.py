"""

This code is to load in data, pass it through a trained model and then view it next to the ground truth.

Joe/Sonia 07/03/2023
"""
# ----------------------------- Load Packages --------------------------------------- 0.

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from VAE.dataloader import BrainDataset3D
from VAE.plotting import plot_outputs, plot_outputs_with_diff
from VAE.models import VAE3d, get_output



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
                         shuffle=False)

valLoader = DataLoader(val_data,
                       batch_size=batch,
                       shuffle=True)

print('2. Data loaders have been created.')

# ----------------------------- Load Model ---------------------------- 3.

model = VAE3d()
model = model.to(device)

model_path = 'D:\\ADNI_VAE\\Models\\DevModels\\'
# model_name = f'VAE_half_data_1000.h5'
model_name = f'{sys.argv[1]}.h5'
load_model = os.path.join(model_path, model_name)
model.load_state_dict(torch.load(load_model))

# ----------------------------- Select Data ---------------------------- 4.

img_in, img_out, _, _ = get_output(model, trainLoader)

img_in = torch.squeeze(img_in, axis=0).cpu().numpy()
img_out = torch.squeeze(img_out, axis=0).cpu().numpy()

with_diff = input('\tPlot difference? [y/n]\t')
if with_diff == 'y':
    diff = np.absolute(img_in - img_out)
    plot_outputs_with_diff(img_in, img_out, diff)
else:
    plot_outputs(img_in, img_out)


