"""
    This file takes a set of parameters and uses them for training.
    JOE 22/04/2024
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

from get_params import get_params
from VAE.models import LMMVAEGAN
from VAE.dataloader import LongDataset, SubjectBatchSampler
from VAE.train import lvaegan_loss, lvaegan2_loss, d_loss
from VAE.utils import list_to_str

# ----------------------------------- Set up project and load parameters -----------------------------------------------

# path = sys.argv[1]
path = '..\\ParamFiles\\IGLS_32_new.txt'
params = get_params(path)
name = params["NAME"]
loss_filename = os.path.join(params["PROJECT_DIR"], name + '_loss.txt')

if not os.path.isdir(params["PROJECT_DIR"]):
    os.mkdir(params["PROJECT_DIR"])
    print(f'Made project {params["PROJECT_DIR"]}')
    os.mkdir(params["MODEL_DIR"])
    os.mkdir(params["LATENT_DIR"])
    os.mkdir(params["LOG_DIR"])
    loss_file = open(loss_filename, "w+")
    loss_file.close()

log_no = len(os.listdir(params["LOG_DIR"]))
logger = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(params["LOG_DIR"], f'{name}_{log_no}.log'),
                    level=logging.DEBUG)
logger.info("Loaded packages and parameter file.")

# ----------------------------------------- Load data ----------------------------------------------------

root_path = params["IMAGE_DIR"]
paths = glob(os.path.join(root_path, '*'))
subject_key = pd.read_csv(os.path.join(params["SUBJ_DIR"], params["SUBJ_PATH"]))

# Get cuda device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device = {device}")

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

    n_batches = 0
    data_size = 0
    for batch in dataloader:
        n_batches += 1
        data_size += batch[0].shape[0]

else:
    dataloader = DataLoader(dataset=loaded_data,
                            num_workers=0,
                            batch_size=params["BATCH_SIZE"],
                            shuffle=params["SHUFFLE_BATCHES"])
    n_batches = len(dataloader)
    data_size = len(dataloader.dataset)

logger.info(f"Loaded data: \n\tTotal data points {data_size},")

# ----------------------------------------- Initiate Model ----------------------------------------------------

model = LMMVAEGAN(params["Z_DIM"], params["GAN"], params["VERSION"])

if os.path.isdir(params["MODEL_DIR"]) and len(os.listdir(params["MODEL_DIR"])) > 0:
    model_list = os.listdir(params["MODEL_DIR"])
    model_names = ['_'.join(m.split('_')[:-1]) for m in model_list]
    model_epochs = [int(m.split('_')[-1].replace('.h5', '')) for m in model_list]
    pre_epochs = max(model_epochs)
    model_name = model_list[model_epochs.index(pre_epochs)]
    logger.info(f'Model name: {model_name}')
    try:
        model.load_state_dict(torch.load(os.path.join(params["MODEL_DIR"], model_name)))
        logger.info('Matched model keys successfully.')
    except NameError:
        logger.critical(f'Model matching unsuccessful: \n\t"{model_name}"')

elif os.path.isdir(params["MODEL_DIR"]) and len(os.listdir(params["MODEL_DIR"])) == 0:
    pre_epochs = 0
else:
    os.mkdir(params["MODEL_DIR"])
    pre_epochs = 0
    logger.info('Made new model.')

model = model.to(device)

# ----------------------------------------- Initiate Optimiser(s) ---------------------------------------------------

vae_parameters = list(model.encoder.parameters()) + \
                 list(model.decoder.parameters()) + \
                 list(model.linear_z_ijk.parameters())

if params["VERSION"] == 2:
    vae_parameters += list(model.linear_var.parameters()) + \
                      list(model.linear_mu.parameters())

optimizer = Adam(vae_parameters, lr=params["LR"])

if params["GAN"]:
    d_optimizer = SGD(model.discriminator.parameters(), lr=params["D_LR"], momentum=0.9)

model.mixed_model = params["MIXED_MODEL"]
model.igls_iterations = params["IGLS_ITERATIONS"]
model.save_latent = os.path.join(params["LATENT_DIR"], 'z_ijk.pt') if params["SAVE_LATENT"] else None

model.slope = params["SLOPE"]
model.a01 = params["INCLUDE_A01"]

logger.info('Parameters and optimizers initialised.')
# ----------------------------------------------- Train Loop --------------------------------------------------------

losses = []
for epoch in range(pre_epochs, pre_epochs + params["EPOCHS"]):
    epoch_losses = []
    for batch_no, batch in enumerate(dataloader):
        progress = f'\tEpoch [{epoch + 1} / {pre_epochs + params["EPOCHS"]}]  -  '\
                   + f'Batch [{batch_no + 1} / {n_batches}]'
        logger.info(progress)
        print(progress)

        imgs = batch[0].to(device)
        subj_ids = batch[1].to(device)
        times = batch[2].to(device)

        optimizer.zero_grad()

        if params["VERSION"] == 1:
            pred, z_prior, z_post, cov_mat, mu, betahat, igls_vars = model(imgs, subj_ids, times)

            if params["D_LOSS"]:
                # Training the discriminator
                label_real = torch.rand((imgs.size(0),), dtype=torch.float) / 10 + 0.05 * torch.ones((imgs.size(0),))
                label_fake = torch.rand((imgs.size(0),), dtype=torch.float) / 10 + 0.85 * torch.ones((imgs.size(0),))
                labels = torch.cat((label_real, label_fake)).to(device)

                model.discriminator.zero_grad()
                d_input = torch.cat((imgs, pred)).detach()
                d_output = model.discriminator(d_input).view(-1)
                loss_d = d_loss(d_output, labels)
                loss_d.backward()
                d_optimizer.step()
                d_output = model.discriminator(pred).view(-1)

                d_labels = label_real.to(device)
            else:
                d_output = None
                d_labels = None

            loss, each_loss = lvaegan_loss(target=imgs,
                                           output=pred,
                                           d_output=d_output,
                                           d_labels=d_labels,
                                           prior_z=z_prior,
                                           post_z=z_post,
                                           mu=mu,
                                           cov_mat=cov_mat,
                                           igls_vars=igls_vars,
                                           bse=params["RECON_LOSS"],
                                           disc_loss=params["D_LOSS"],
                                           align=params["ALIGN_LOSS"],
                                           beta=params["BETA"],
                                           gamma=params["GAMMA"]
                                           )

        if params["VERSION"] == 2:
            pred, lin_z_hat, lin_mu, lin_logvar, mm_z_hat, mm_mu, mm_var = model(imgs, subj_ids, times)

            if params["D_LOSS"]:
                label_real = torch.rand((imgs.size(0),), dtype=torch.float) / 10 + 0.05 * torch.ones((imgs.size(0),))
                label_fake = torch.rand((imgs.size(0),), dtype=torch.float) / 10 + 0.85 * torch.ones((imgs.size(0),))
                labels = torch.cat((label_real, label_fake)).to(device)

                model.discriminator.zero_grad()
                d_input = torch.cat((imgs, pred)).detach()
                d_output = model.discriminator(d_input).view(-1)
                loss_d = d_loss(d_output, labels)
                loss_d.backward()
                d_optimizer.step()
                d_output = model.discriminator(pred).view(-1)
                d_labels = label_real.to(device)
            else:
                d_output = None
                d_labels = None

            loss, each_loss = lvaegan2_loss(target=imgs,
                                            output=pred,
                                            lin_z_hat=lin_z_hat,
                                            mm_z_hat=mm_z_hat,
                                            lin_mu=lin_mu,
                                            lin_logvar=lin_logvar,
                                            mm_mu=mm_mu,
                                            mm_var=mm_var,
                                            d_output=d_output,
                                            d_labels=d_labels,
                                            beta=params["BETA"],
                                            gamma=params["GAMMA"],
                                            nu=params["NU"],
                                            recon=params["RECON_LOSS"],
                                            kl=params["KL_LOSS"],
                                            align=params["ALIGN_LOSS"],
                                            disc_loss=params["D_LOSS"],
                                            )

        loss.backward()
        optimizer.step()
        epoch_losses.append(each_loss)
    epoch_losses = np.asarray(epoch_losses).mean(axis=0).tolist()
    losses.append(epoch_losses)

    # Save the model and the losses to the file if the correct epoch
    if (epoch + 1) % params["SAVE_EPOCHS"] == 0:
        torch.save(model.state_dict(), os.path.join(params["MODEL_DIR"], f'{name}_{epoch + 1}.h5'))
        logger.info(f'Saved {name}_{epoch + 1}.h5')

        loss_file = open(os.path.join(params["PROJECT_DIR"], loss_filename), 'a+')
        for loss_line in losses[-params["SAVE_EPOCHS"]:]:
            loss_line = list_to_str(loss_line) + '\n'
            loss_file.write(loss_line)
        loss_file.close()
        logger.info('Saved losses')

    logger.info(f'\n\tLoss: {losses[-1][0]:.6f}')
    logger.info(f'\tRecon {losses[-1][1]:.6f}')
    logger.info(f'\tAlign {losses[-1][2]:.6f}')
    logger.info(f'\tDiscr {losses[-1][3]:.6f}')
    logger.info(f'\tKL    {losses[-1][4]:.6f}\n')

    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
