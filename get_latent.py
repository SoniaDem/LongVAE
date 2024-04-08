"""
    Hello this is a test using the IGLS estimation method within the VAE.
    JONIA 2024-04-04.

    This file is used to extract the latent variables from the models.
"""

# ----------------------------------------- Load Packages ----------------------------------------------------
from glob import glob
import albumentations as a
import numpy as np
from torch.utils.data import DataLoader
import torch
import pandas as pd
import os
# import sys

from VAE.models import LVAE_LIN
from VAE.dataloader import LongDataset, SubjectBatchSampler
from VAE.train import lvae_lin_loss, loss_fn
from VAE.utils import get_args, list_to_str

# ----------------------------------------- Load Parameters ----------------------------------------------------

# First get the parameters from the text file.
param_path = 'D:\\ADNI_VAE\\ParamFiles\\IGLS_test_params.txt'
# param_path = sys.argv[1]  # Use this if running the code externally.
params = get_args(param_path)  # This will return a dictionary of parameters that are stored.

name = params["NAME"]
project_dir = os.path.join(params["PROJECT_DIR"], name)

# reset = True if 'RESET' in params.keys() and params['RESET'].lower() == 'true' else False

if not os.path.isdir(project_dir):
    os.mkdir(project_dir)
    print(f'Made project {project_dir}')

project_files = os.listdir(project_dir)
h_flip = 0.
v_flip = 0.
batch_size = int(params["BATCH_SIZE"])
shuffle_batches = True if params['SHUFFLE_BATCHES'].lower() == 'true' else False
epochs = int(params["EPOCHS"])
save_epochs = int(params["SAVE_EPOCHS"])
recon_loss = True if params["RECON_LOSS"].lower() == 'true' else False
kl_loss = True if params["KL_LOSS"].lower() == 'true' else False
align_loss = True if params["ALIGN_LOSS"].lower() == 'true' else False
beta = 1 if "BETA" not in params["BETA"] else float(params["BETA"])
gamma = 1 if "GAMMA" not in params["GAMMA"] else float(params["GAMMA"])
lr = float(params["LR"]) if "LR" in params.keys() else 1e-4
momentum = float(params["MOMENTUM"]) if "MOMENTUM " in params.keys() else 0.9
delta = None if "DELTA" not in params.keys() else float(params["DELTA"])
sampler_params = [3, 6] if 'SAMPLER_PARAMS' not in params.keys() else params['SAMPLER_PARAMS']
use_sampler = True if 'USE_SAMPLER' not in params.keys() or params["USE_SAMPLER"].lower() == 'true' else False
mixed_model = True if 'MIXED_MODEL' in params.keys() and params['MIXED_MODEL'].lower() == 'true' else False
igls_iterations = int(params['IGLS_ITERATIONS']) if 'IGLS_ITERATIONS' in params.keys() else None

print('Loaded parameters')
# ----------------------------------------- Load Data ----------------------------------------------------

# Retrieve list of image paths
root_path = params["IMAGE_DIR"]
paths = glob(os.path.join(root_path, '*'))
subject_key = pd.read_csv(os.path.join(os.getcwd(), 'subject_id_key.csv'))

# Get cuda device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device = {device}')


transforms = a.Compose([
    a.HorizontalFlip(p=h_flip),
    a.VerticalFlip(p=v_flip),
])

loaded_data = LongDataset(image_list=paths,
                          subject_key=subject_key,
                          transformations=transforms)

if use_sampler:
    custom_sampler = SubjectBatchSampler(subject_dict=loaded_data.subj_dict,
                                         batch_size=batch_size,
                                         min_data=int(sampler_params[0]),
                                         max_data=int(sampler_params[1]))

    dataloader = DataLoader(dataset=loaded_data,
                            num_workers=0,
                            batch_sampler=custom_sampler)

else:
    dataloader = DataLoader(dataset=loaded_data,
                            num_workers=0,
                            batch_size=batch_size,
                            shuffle=shuffle_batches)


print(f'Loaded data: \n\tTotal data points {len(dataloader.dataset)}, '
      f'\n\tBatches {len(dataloader)}, '
      f'\n\tBatch_size {dataloader.batch_size}.')

# ----------------------------------------- Get Model ----------------------------------------------------

# Initialise the model
model = LVAE_LIN(int(params["Z_DIM"]))

# Here, get the latest model along with the number of epochs or create a new model if one doesn't exist.
model_dir = os.path.join(project_dir, 'Models')

model_list = os.listdir(model_dir)
model_names = ['_'.join(m.split('_')[:-1]) for m in model_list]
model_epochs = [int(m.split('_')[-1].replace('.h5', '')) for m in model_list]
pre_epochs = max(model_epochs)
model_name = model_list[model_epochs.index(pre_epochs)]
print(f'Latest model: {model_name}')
print(f'\nOther model epochs\n\t{sorted(model_epochs, reverse=True)}')

use_model_epoch = 3660

model_name = model_list[model_epochs.index(use_model_epoch)]

try:
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
    print('Matched model keys successfully.')
except NameError:
    print(f'Model matching unsuccessful: \n\t"{model_name}"')

model = model.to(device)
model.eval()

# ----------------------------------------- Get Latent Variables ----------------------------------------------------

n_samples = 0
for batch_no, batch in enumerate(dataloader):
    n_samples += batch[0].shape[0]

all_preds = np.zeros((n_samples, 1, 56, 48, 48))
all_lin_z_hats = np.zeros((n_samples, int(params["Z_DIM"])))
all_lin_mus = np.zeros((n_samples, int(params["Z_DIM"])))
all_lin_log_vars = np.zeros((n_samples, int(params["Z_DIM"])))
all_mm_z_hats = np.zeros((n_samples, int(params["Z_DIM"])))
all_mm_mus = np.zeros((n_samples, int(params["Z_DIM"])))
all_mm_vars = np.zeros((n_samples, int(params["Z_DIM"])))

for batch_no, batch in enumerate(dataloader):
    print(f'Batch [{batch_no + 1} / {len(dataloader)}]')

    imgs = batch[0].to(device)
    subj_ids = batch[1].to(device)
    times = batch[2].to(device)

    pred, lin_z_hat, lin_mu, lin_logvar, mm_z_hat, mm_mu, mm_var = model(imgs, subj_ids, times)

    batch_start = batch_no*batch_size
    batch_end = (batch_no+1)*batch_size
    all_preds[batch_start:batch_end, :, :, :, :] = pred.detach().cpu().numpy()
    all_lin_z_hats[batch_start:batch_end, :] = lin_z_hat.detach().cpu().numpy()
    all_lin_mus[batch_start:batch_end, :] = lin_mu.detach().cpu().numpy()
    all_lin_log_vars[batch_start:batch_end, :] = lin_logvar.detach().cpu().numpy()
    all_mm_mus[batch_start:batch_end, :] = mm_z_hat.detach().cpu().numpy()
    all_mm_vars[batch_start:batch_end, :] = mm_var.detach().cpu().numpy()


# ----------------------------------------- Save Latent Variables ----------------------------------------------------

param_dir = os.path.join(project_dir, 'Latent Params')
if not os.path.isdir(param_dir):
    os.mkdir(param_dir)
    print('Made directory', param_dir)

all_lin_z_hats = pd.DataFrame(all_lin_z_hats)
all_lin_z_hats.to_csv(os.path.join(param_dir, f'lin_z_hats_e{use_model_epoch}.csv'), index=False)
print('Saved', os.path.join(param_dir, f'lin_z_hats_e{use_model_epoch}.csv'))

all_lin_mus = pd.DataFrame(all_lin_mus)
all_lin_mus.to_csv(os.path.join(param_dir, f'all_lin_mus_e{use_model_epoch}.csv'), index=False)
print('Saved', os.path.join(param_dir, f'all_lin_mus_e{use_model_epoch}.csv'))

all_lin_log_vars = pd.DataFrame(all_lin_log_vars)
all_lin_log_vars.to_csv(os.path.join(param_dir, f'all_lin_log_vars_e{use_model_epoch}.csv'), index=False)
print('Saved', os.path.join(param_dir, f'all_lin_log_vars_e{use_model_epoch}.csv'))

all_mm_mus = pd.DataFrame(all_mm_mus)
all_mm_mus.to_csv(os.path.join(param_dir, f'all_mm_mus_e{use_model_epoch}.csv'), index=False)
print('Saved', os.path.join(param_dir, f'all_mm_mus_e{use_model_epoch}.csv'))

all_mm_vars = pd.DataFrame(all_mm_vars)
all_mm_vars.to_csv(os.path.join(param_dir, f'all_mm_vars_e{use_model_epoch}.csv'), index=False)
print('Saved', os.path.join(param_dir, f'all_mm_vars_e{use_model_epoch}.csv'))
