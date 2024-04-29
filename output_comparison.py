"""

This code is to load in data, pass it through a trained model and then view the reconstruction
next to the ground truth.

Jonia 18/04/2024
"""
# ----------------------------- Load Packages --------------------------------------- 0.

import os
import sys
from glob import glob

import pandas as pd
import torch
from torch.utils.data import DataLoader
import albumentations as a

from VAE.dataloader import LongDataset, SubjectPerBatchSampler
from VAE.models import LMMVAEGAN
# from VAE.utils import get_args
from get_params import get_params
from VAE.plotting import plot_slice_prediction

# ----------------------------------------- Load parameters ----------------------------------------------------

param_path = 'D:\\Projects\\SoniaVAE\\ParamFiles\\IGLS_test_params.txt'
# param_path = sys.argv[1]

params = get_params(param_path)
name = params["NAME"]
project_dir = os.path.join(params["PROJECT_DIR"], name)

print('Loaded parameters')
# ----------------------------------------- Load data ----------------------------------------------------

# Retrieve list of image paths
root_path = params["IMAGE_DIR"]
paths = glob(os.path.join(root_path, '*'))
subject_key = pd.read_csv(os.path.join(params["SUBJ_DIR"], params["SUBJ_PATH"]))

# Get cuda device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device = {device}')

transforms = a.Compose([
    a.HorizontalFlip(p=params["H_FLIP"]),
    a.VerticalFlip(p=params["V_FLIP"])
])

loaded_data = LongDataset(image_list=paths,
                          subject_key=subject_key,
                          transformations=transforms,
                          min_data=params["MIN_DATA"])

custom_sampler = SubjectPerBatchSampler(subject_dict=loaded_data.subj_dict,
                                        min_data=int(params["SAMPLER_PARAMS"][0]))

dataloader = DataLoader(dataset=loaded_data,
                        num_workers=0,
                        batch_sampler=custom_sampler)


print(f"Loaded data: \n\tTotal data points {len(dataloader.dataset)}")

# ----------------------------------------- Initiate Model ----------------------------------------------------

model = LMMVAEGAN(params["Z_DIM"], params["GAN"], params["VERSION"])

model_list = os.listdir(params["MODEL_DIR"])
model_names = ['_'.join(m.split('_')[:-1]) for m in model_list]
model_epochs = [int(m.split('_')[-1].replace('.h5', '')) for m in model_list]
model_name = model_list[model_epochs.index(params["PLOT_EPOCH"])]
print(f'Model name: {model_name}')

try:
    model.load_state_dict(torch.load(os.path.join(params["MODEL_DIR"], model_name)))
    print('Matched model keys successfully.')
except NameError:
    print(f'Model matching unsuccessful: \n\t"{model_name}"')

model = model.to(device)

model.mixed_model = params["MIXED_MODEL"]
model.igls_iterations = params["IGLS_ITERATIONS"]
model.save_latent = os.path.join(params["LATENT_DIR"], 'z_ijk.pt') if params["SAVE_LATENT"] else None

if params["VERSION"] == 1:
    model.slope = params["SLOPE"]
    model.a01 = params["INCLUDE_A01"]

print('\tmodel.slope', model.slope)
print('\tmodel.a01', model.a01)

# ------------------------- Select Data and Pass Through Model ------------------------ 2.

print('Getting subject and passing through model.')
data_iter = iter(dataloader)


for i in range(len(loaded_data.subj_dict.keys())):
    img, subj_id, time = next(data_iter)
    if params["IMAGE_NO"] in subj_id:
        break

print(f'Subject {params["IMAGE_NO"]}: {time.tolist()}')

time_idx = time.tolist().index(params["TIMEPOINT"])

img = img.to(device)
subj_id = subj_id.to(device)
time = time.to(device)

model.eval()
pred, _, _, _, _, _, _ = model(img, subj_id, time)

img = img.cpu()  # shape (batch, 1, 56, 48, 48)
pred = pred.cpu().detach().numpy()  # shape (batch, 1, 56, 48, 48)

img = img[time_idx].squeeze(0)
pred = pred[time_idx].squeeze(0)

print('Plotting')

title = f'Subject ID {params["IMAGE_NO"]}, Time {time[params["TIMEPOINT"]].item()}'
plot_slice_prediction(img, pred, axis=params["AXIS"], title=title)
