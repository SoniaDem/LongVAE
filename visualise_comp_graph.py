"""
    This should take the parameters from a parameter file like the training and evaluation files but uses these to plot
    the computational graph.
"""

# ----------------------------------------- Load Packages ----------------------------------------------------
import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import logging
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import albumentations as a

from torchviz import make_dot


from get_params import get_params
from VAE.models import LMMVAEGAN, VAE_IGLS
from VAE.dataloader import LongDataset, SubjectBatchSampler
from VAE.train import lvaegan_loss, lvaegan2_loss, d_loss
from VAE.utils import list_to_str

# ----------------------------------- Set up project and load parameters -----------------------------------------------


# path = sys.argv[1]
path = 'D:\\Projects\\SoniaVAE\\ParamFiles\\IGLS_test_params.txt'
params = get_params(path)
name = params["NAME"]

# ----------------------------------------- Load data ----------------------------------------------------

root_path = params["IMAGE_DIR"]
paths = glob(os.path.join(root_path, '*'))
subject_key = pd.read_csv(os.path.join(params["SUBJ_DIR"], params["SUBJ_PATH"]))

# Get cuda device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device = {device}")

transforms = a.Compose([
    a.HorizontalFlip(p=params["H_FLIP"]),
    a.VerticalFlip(p=params["V_FLIP"])
])

loaded_data = LongDataset(image_list=paths,
                          subject_key=subject_key,
                          transformations=transforms,
                          min_data=params["MIN_DATA"])

if params["USE_SAMPLER"]:
    custom_sampler = SubjectBatchSampler(subject_dict=loaded_data.subj_dict,
                                         batch_size=params["BATCH_SIZE"],
                                         min_data=int(params["SAMPLER_PARAMS"][0]),
                                         max_data=int(params["SAMPLER_PARAMS"][1]))

    dataloader = DataLoader(dataset=loaded_data,
                            num_workers=0,
                            batch_sampler=custom_sampler)

else:
    dataloader = DataLoader(dataset=loaded_data,
                            num_workers=0,
                            batch_size=params["BATCH_SIZE"],
                            shuffle=params["SHUFFLE_BATCHES"])

print(f"Loaded data: \n\tTotal data points {len(dataloader.dataset)},")

# ----------------------------------------- Initiate Model ----------------------------------------------------

model = LMMVAEGAN(params["Z_DIM"], params["GAN"], params["VERSION"])

model = model.to(device)

model.mixed_model = params["MIXED_MODEL"]
model.igls_iterations = params["IGLS_ITERATIONS"]
model.save_latent = os.path.join(params["LATENT_DIR"], 'z_ijk.pt') if params["SAVE_LATENT"] else None

if params["VERSION"] == 1:
    model.slope = params["SLOPE"]
    model.a01 = params["INCLUDE_A01"]

# ----------------------------------------- Pass Data ----------------------------------------------------

batch = next(iter(dataloader))
imgs = batch[0].to(device)
subj_ids = batch[1].to(device)
times = batch[2].to(device)

if params["VERSION"] == 1:
    pred, z_prior, z_post, cov_mat, mu, betahat, igls_vars = model(imgs, subj_ids, times)

if params["VERSION"] == 2:
    pred, lin_z_hat, lin_mu, lin_logvar, mm_z_hat, mm_mu, mm_var = model(imgs, subj_ids, times)

g = 'GAN' if params["GAN"] else "NOGAN"
save_name = f'torchviz_v{params["VERSION"]}_{g}_{params["IGLS_ITERATIONS"]}'
save_dir = os.path.join('D:\\Projects\\SoniaVAE\\ComputationalGraph', save_name)
make_dot(pred, params=dict(list(model.named_parameters()))).render(save_dir, format="png")
