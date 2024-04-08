"""
    Hello this is a test using the IGLS estimation method within the VAE.
    JONIA 2024-03-22
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

from VAE.models import VAE_IGLS
from VAE.dataloader import LongDataset, SubjectBatchSampler
from VAE.train import lvae_loss, loss_fn
from VAE.utils import get_args, list_to_str

# ----------------------------------------- Load parameters ----------------------------------------------------

# First get the parameters from the text file.
param_path = 'D:\\ADNI_VAE\\ParamFiles\\IGLS_noa01_params.txt' # this is the directory on Sonia's PC.
# param_path = 'D:\\Projects\\SoniaVAE\\ParamFiles\\IGLS_noa01_params.txt'  # this is the directory on Joe's PC.
# param_path = sys.argv[1]  # Use this if running the code externally.
params = get_args(param_path)  # This will return a dictionary of parameters that are stored.

name = params["NAME"]
project_dir = os.path.join(params["PROJECT_DIR"], name)

# reset = True if 'RESET' in params.keys() and params['RESET'].lower() == 'true' else False

if not os.path.isdir(project_dir):
    os.mkdir(project_dir)
    print(f'Made project {project_dir}')
    os.mkdir(os.path.join(project_dir, 'Latent Params'))


project_files = os.listdir(project_dir)
h_flip = 0. if "H_FLIP" not in params.keys() else float(params["H_FLIP"])
v_flip = 0. if "V_FLIP" not in params.keys() else float(params["V_FLIP"])
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
train_with_igls = True if 'ESTIMATE_IGLS' in params.keys() and params["USE_SAMPLER"].lower() == 'true' else False
mixed_model = True if 'MIXED_MODEL' in params.keys() and params['MIXED_MODEL'].lower() == 'true' else False
igls_iterations = int(params['IGLS_ITERATIONS']) if 'IGLS_ITERATIONS' in params.keys() else None
save_latent = True if "SAVE_LATENT" in params.keys() and params["SAVE_LATENT"].lower() == 'true' else False
latent_dir = os.path.join(project_dir, 'Latent Params')
slope = True if "SLOPE" in params.keys() and params["SLOPE"].lower() == 'true' else False
min_subj_t = None if "MIN_DATA" not in params.keys() else int(params["MIN_DATA"])
include_a01 = True if "INLCUDE_A01" in params.keys() and params["INLCUDE_A01"].lower == 'true' else False

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
                            shuffle=True)

print(f'Loaded data: \n\tTotal data points {len(dataloader.dataset)}, '
      f'\n\tBatches {len(dataloader)}, '
      f'\n\tBatch_size {dataloader.batch_size}.')

# for i, test in enumerate(dataloader):
#     print(f'batch {i}, size {test[0].shape[0]}')

# ----------------------------------------- Initiate Loss File ----------------------------------------------------

# If a loss file does not exist then create one because why not lol.
# Also, if the loss file exists, but you don't care about it there is an "OVERWRITE_LOSS" argument that can be used
overwrite = False if 'OVERWRITE_LOSS' not in params.keys() or params["OVERWRITE_LOSS"].lower() == 'false' else True
loss_filename = name + '_loss.txt'

if loss_filename not in project_files or overwrite:
    loss_file = open(os.path.join(project_dir, loss_filename), 'w+')
    loss_file.close()

# ----------------------------------------- Initiate Model ----------------------------------------------------

# Initialise the model
model = VAE_IGLS(int(params["Z_DIM"]))

# Here, get the latest model along with the number of epochs or create a new model if one doesn't exist.
model_dir = os.path.join(project_dir, 'Models')
if os.path.isdir(model_dir) and len(os.listdir(model_dir)) > 0:
    model_list = os.listdir(model_dir)
    model_names = ['_'.join(m.split('_')[:-1]) for m in model_list]
    model_epochs = [int(m.split('_')[-1].replace('.h5', '')) for m in model_list]
    pre_epochs = max(model_epochs)
    model_name = model_list[model_epochs.index(pre_epochs)]
    print(f'Model name: {model_name}')
    try:
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
        print('Matched model keys successfully.')
    except NameError:
        print(f'Model matching unsuccessful: \n\t"{model_name}"')

elif os.path.isdir(model_dir) and len(os.listdir(model_dir)) == 0:
    pre_epochs = 0
else:
    os.mkdir(model_dir)
    pre_epochs = 0
    print('Made new model.')

model = model.to(device)

# ----------------------------------------- Initiate Optimiser ----------------------------------------------------

optimizer = torch.optim.SGD(list(model.parameters()),
                            lr=lr,
                            momentum=momentum)
if delta is not None:
    model.delta = delta

model.mixed_model = mixed_model

if igls_iterations is not None:
    model.igls_iterations = igls_iterations

if save_latent:
    model.save_latent = latent_dir

model.slope = slope
model.a01 = include_a01

print('\tmodel.save_latent', model.save_latent)
print('\tmodel.slope', model.slope)
print('\tmodel.a01', model.a01)

# ----------------------------------------- Train Model ----------------------------------------------------

losses = []
for epoch in range(pre_epochs, pre_epochs + epochs):
    # print(f'Epoch [{epoch + 1} / {pre_epochs + epochs}]')

    epoch_losses = []
    for batch_no, batch in enumerate(dataloader):
        print(f'\tEpoch [{epoch + 1} / {pre_epochs + epochs}]  -  Batch [{batch_no + 1} / {len(dataloader)}]')

        imgs = batch[0].to(device)
        subj_ids = batch[1].to(device)
        times = batch[2].to(device)
        # print(f'No. unique patients {subj_ids.unique().shape}')
        # print('Attached to devices')

        optimizer.zero_grad()
        # print('Zerod optimizer')
        pred, z_prior, z_post, cov_mat, mu, betahat, igls_vars = model(imgs, subj_ids, times)
        # print('Passed through model')

        loss, each_loss = lvae_loss(target=imgs,
                                    output=pred,
                                    prior_z=z_prior,
                                    post_z=z_post,
                                    mu=mu,
                                    cov_mat=cov_mat,
                                    igls_vars=igls_vars,
                                    bse=recon_loss,
                                    kl=kl_loss,
                                    align=align_loss,
                                    beta=beta,
                                    gamma=gamma
                                    )

        # print('got loss value', each_loss)
        loss.backward(retain_graph=True)
        # print('done loss.backward()')
        optimizer.step()
        # print('stepped optimizer')
        epoch_losses.append(each_loss)
    # print(epoch_losses)
    epoch_losses = np.asarray(epoch_losses).mean(axis=0).tolist()
    # print(epoch_losses)
    losses.append(epoch_losses)

    # Save the model and the losses to the file if the correct epoch
    if (epoch + 1) % save_epochs == 0:
        torch.save(model.state_dict(), os.path.join(model_dir, f'{name}_{epoch + 1}.h5'))
        print(f'Saved {name}_{epoch}.h5')

        loss_file = open(os.path.join(project_dir, loss_filename), 'a+')
        for loss_line in losses[-save_epochs:]:
            loss_line = list_to_str(loss_line) + '\n'
            loss_file.write(loss_line)
        loss_file.close()
        print('Saved losses')

    print(f'\n\tLoss: {losses[-1][0]:.6f}')
    print(f'\tRecon {losses[-1][1]:.6f}')
    print(f'\tKL {losses[-1][2]:.6f}')
    print(f'\tAlign {losses[-1][3]:.6f}\n')

    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

print('Done')




#
# from VAE.plotting import plot_loss
#
# losses_txt = [l.strip('\n') for l in open(os.path.join(project_dir, loss_filename), 'r')]
# losses_txt = [float(l.split(' ')[0]) for l in losses_txt]
# plot_loss(losses_txt[10:])
#
# plot_loss(np.asarray(losses)[:, 3])
#
# test_imgs, test_ids, test_times = next(iter(dataloader))
#
# test_imgs = test_imgs.to(device)
# test_ids = test_ids.to(device)
# test_times = test_times.to(device)
#
# model = VAE_IGLS(64).to(device)
#
# # out, sig, beta, mu = model(test_imgs, test_ids, test_times)
#
# x = model(test_imgs, test_ids, test_times)

#
# #
# from torch import eye, zeros, flatten, cat
#
# z1 = eye(batch_size).to(device)
# z2 = zeros((batch_size, batch_size)).to(device)
# z3 = zeros((batch_size, batch_size)).to(device)
# z4 = zeros((batch_size, batch_size)).to(device)
#
# for i in range(batch_size):
#     for j in range(batch_size):
#
#         subj_i = subj_ids[i]
#         subj_j = subj_ids[j]
#
#         visit_i = times[i]
#         visit_j = times[j]
#
#         if subj_i == subj_j:
#             z2[i, j] = 1
#             z3[i, j] = visit_i + visit_j
#             z4[i, j] = visit_i * visit_j
#
# vz1 = flatten(z1.transpose(1, 0)).expand(1, -1).T  # size (batch_size^2, 1)
# vz2 = flatten(z2.transpose(1, 0)).expand(1, -1).T
# vz3 = flatten(z3.transpose(1, 0)).expand(1, -1).T
# vz4 = flatten(z4.transpose(1, 0)).expand(1, -1).T
# zz = cat((vz1, vz2, vz3, vz4), axis=1)
#
# zzt_zz = zz.T @ zz

# import pandas as pd
# z_prior = z_prior.detach().cpu().numpy()
# z_df = pd.DataFrame(z_prior)
# z_df.to_csv('D:\\ADNI_VAE\\NonsenseFiles\\z_vae_ijk.csv', index=False)
#
# times = times.detach().cpu().numpy()
# time_df = pd.DataFrame(times)
# time_df.to_csv('D:\\ADNI_VAE\\NonsenseFiles\\z_vae_time.csv', index=False)
#
# subj_ids = subj_ids.detach().cpu().numpy()
# subj_df = pd.DataFrame(subj_ids)
# subj_df.to_csv('D:\\ADNI_VAE\\NonsenseFiles\\z_vae_subj.csv', index=False)
