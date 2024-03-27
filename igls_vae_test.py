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
from VAE.dataloader import LongDataset
from VAE.train import lvae_loss, loss_fn
from VAE.utils import get_args, list_to_str

# ----------------------------------------- Load parameters ----------------------------------------------------

# First get the parameters from the text file.
param_path = 'D:\\ADNI_VAE\\ParamFiles\\IGLS_test_params.txt'
# param_path = sys.argv[1]  # Use this if running the code externally.
params = get_args(param_path)  # This will return a dictionary of parameters that are stored.

name = params["NAME"]
project_dir = os.path.join(params["PROJECT_DIR"], name)

reset = True if 'RESET' in params.keys() and params['RESET'].lower() == 'True' else False

if not os.path.isdir(project_dir) or reset:
    os.mkdir(project_dir)
    print(f'Made project {project_dir}')

project_files = os.listdir(project_dir)
h_flip = 0. if "H_FLIP" not in params.keys() else float(params["H_FLIP"])
v_flip = 0. if "V_FLIP" not in params.keys() else float(params["V_FLIP"])
rand_rot = 0. if "RAND_ROTATE" not in params.keys() else float(params["RAND_ROTATE"])
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
    a.VerticalFlip(p=v_flip),
    a.RandomRotate90(p=rand_rot),
])

loaded_data = LongDataset(image_list=paths,
                          subject_key=subject_key,
                          transformations=transforms)

dataloader = DataLoader(dataset=loaded_data,
                        batch_size=batch_size,
                        num_workers=0,
                        shuffle=shuffle_batches)

print(f'Loaded data: \n\tTotal data points {len(dataloader.dataset)}, '
      f'\n\tBatches {len(dataloader)}, '
      f'\n\tBatch_size {dataloader.batch_size}.')

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

model.delta = 1e-1

# ----------------------------------------- Train Model ----------------------------------------------------

losses = []
for epoch in range(pre_epochs, pre_epochs + epochs):
    print(f'Epoch [{epoch + 1} / {pre_epochs + epochs}]')

    epoch_losses = []
    for batch_no, batch in enumerate(dataloader):

        print(f'\tBatch [{batch_no + 1} / {len(dataloader)}]')

        imgs = batch[0].to(device)
        subj_ids = batch[1].to(device)
        times = batch[2].to(device)
        # print('Attached to devices')

        optimizer.zero_grad()
        print('Zerod optimizer')
        pred, z_prior, z_post, cov_mat, mu, betahat, igls_vars = model(imgs, subj_ids, times)
        print('Passed through model')


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

        print('got loss value', each_loss)
        loss.backward(retain_graph=True)
        print('done loss.backward()')
        optimizer.step()
        print('stepped optimizer')
        epoch_losses.append(each_loss)

    epoch_losses = np.asarray(epoch_losses).mean(axis=0).tolist()
    losses.append(epoch_losses)

    # Save the model and the losses to the file if the correct epoch
    if (epoch+1) % save_epochs == 0:
        torch.save(model.state_dict(), os.path.join(model_dir, f'{name}_{epoch+1}.h5'))
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


from VAE.plotting import plot_loss

losses_txt = [l.strip('\n') for l in open(os.path.join(project_dir, loss_filename), 'r')]
losses_txt = [float(l.split(' ')[0]) for l in losses_txt]
plot_loss(losses_txt[10:])



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
#
# from torch import tensor, zeros
# from torch.distributions.normal import Normal
#
# s_a0 = tensor([1.0000e-06, 1.0000e-06, 3.0826e-01, 1.2234e-01, 1.0000e-06, 1.0000e-06,
#                5.7495e-02, 1.0000e-06, 1.9340e-01, 5.3452e-02, 1.4071e-01, 1.0000e-06,
#                1.0000e-06, 9.9936e-02, 7.2411e-02, 6.8768e-02, 2.7696e-02, 6.0275e-02,
#                2.0357e-03, 3.9366e-02, 1.0974e-01, 4.0478e-02, 1.0270e-01, 1.0191e-01,
#                2.8810e-03, 1.0000e-06, 1.0000e-06, 1.0000e-06, 1.0000e-06, 1.0000e-06,
#                2.5366e-01, 2.9570e-03, 5.9013e-02, 1.5969e-01, 5.3727e-02, 2.6780e-02,
#                1.3791e-01, 1.0000e-06, 1.0000e-06, 5.6404e-02, 1.2999e-01, 8.0231e-02,
#                6.8059e-02, 1.0000e-06, 5.3010e-02, 1.2680e-02, 1.3311e-01, 1.0000e-06,
#                1.0426e-02, 1.0000e-06, 2.9582e-02, 1.3844e-01, 8.7853e-02, 1.0000e-06,
#                1.0000e-06, 5.6054e-02, 1.0000e-06, 1.0000e-06, 1.0000e-06, 4.3480e-02,
#                1.0000e-06, 1.0000e-06, 1.0805e-01, 1.0000e-06])
#
# # a0 = Normal(zeros(s_a0.shape[0]), s_a0).sample([50]).T
# t = torch.cat([s_a0.expand(1, -1), s_a0.expand(1, -1), s_a0.expand(1,-1)], 0)
