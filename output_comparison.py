"""

This code is to load in data, pass it through a trained model and then view the reconstruction
next to the ground truth.

Joe/Sonia 07/03/2023
"""
# ----------------------------- Load Packages --------------------------------------- 0.

import os
import sys
from glob import glob

import pandas as pd
import torch
from torch.utils.data import DataLoader
import albumentations as a

from VAE.dataloader import LongDataset
from VAE.models import VAE_IGLS, LVAE_LIN
from VAE.utils import get_args
from VAE.plotting import plot_slice_prediction

# ----------------------------------------- Load parameters ----------------------------------------------------

param_path = 'D:\\Projects\\SoniaVAE\\ParamFiles\\IGLS_test_params.txt'
# param_path = sys.argv[1]

params = get_args(param_path)
name = params["NAME"]
project_dir = os.path.join(params["PROJECT_DIR"], name)

h_flip = 0.
v_flip = 0.
min_subj_t = None if "MIN_DATA" not in params else int(params["MIN_DATA"])
epochs = int(params["EPOCHS"])
version = int(params["VERSION"])
axis = 0 if "AXIS" not in params else int(params["AXIS"])
image_no = None if 'IMAGE_NO' not in params else int(params['IMAGE_NO'])

print('Loaded parameters')
# ----------------------------------------- Load data ----------------------------------------------------

# Retrieve list of image paths
root_path = params["IMAGE_DIR"]
paths = glob(os.path.join(root_path, '*'))
subject_key = pd.read_csv(os.path.join(os.getcwd(), 'subject_id_key.csv'))

# Get cuda device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device = {device}')

transforms = a.Compose([
    a.HorizontalFlip(p=h_flip),
    a.VerticalFlip(p=v_flip)
])

loaded_data = LongDataset(image_list=paths,
                          subject_key=subject_key,
                          transformations=transforms,
                          min_data=min_subj_t)

dataloader = DataLoader(dataset=loaded_data,
                        num_workers=0,
                        batch_size=1,
                        shuffle=False)

print(f'Loaded data: \n\tTotal data points {len(dataloader.dataset)}, '
      f'\n\tBatches {len(dataloader)}, '
      f'\n\tBatch_size {dataloader.batch_size}.')

# ----------------------------- Load Model ---------------------------- 1.

if version == 1:
    model = VAE_IGLS(int(params["Z_DIM"]))

if version == 2:
    model = LVAE_LIN(int(params["Z_DIM"]))

model_dir = os.path.join(project_dir, 'Models')
model_list = os.listdir(model_dir)
model_names = ['_'.join(m.split('_')[:-1]) for m in model_list]
model_epochs = [int(m.split('_')[-1].replace('.h5', '')) for m in model_list]
model_name = model_list[model_epochs.index(epochs)]
print(f'Model name: {model_name}')

try:
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
    print('Matched model keys successfully.')
except NameError:
    print(f'Model matching unsuccessful: \n\t"{model_name}"')

model = model.to(device)

model.mixed_model = False

# ------------------------- Select Data and Pass Through Model ------------------------ 2.

print('Getting subject and passing through model.')
image_no = image_no if image_no is not None else torch.randint(0, len(paths), (1,)).item()

data_iter = iter(dataloader)
for i in range(len(paths)):
    img, subj_id, time = next(data_iter)
    if i == image_no:
        break

img = img.to(device)
subj_id = subj_id.to(device)
time = time.to(device)

model.eval()
pred, _, _, _, _, _, _ = model(img, subj_id, time)

img = img.cpu() # shape (1, 1, 56, 48, 48)
pred = pred.cpu().detach().numpy() # shape (1, 1, 56, 48, 48)

img = img.squeeze(0).squeeze(0)
pred = pred.squeeze(0).squeeze(0)

print('Plotting')

plot_slice_prediction(img, pred, axis=2, title=f'Subject ID {subj_id.item()}')














