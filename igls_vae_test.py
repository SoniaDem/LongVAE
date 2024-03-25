"""
    Hello this is a test using the IGLS estimation method within the VAE.
    JONIA 2024-03-22
"""

# ----------------------------------------- Load Packages ----------------------------------------------------
from glob import glob
import albumentations as a
from torch.utils.data import DataLoader
import torch
import pandas as pd
import os
# import sys

from VAE.models import VAE_IGLS
from VAE.dataloader import LongDataset
from VAE.train import lvae_loss
from VAE.utils import get_args, list_to_str

# ----------------------------------------- Load parameters ----------------------------------------------------

# First get the parameters from the text file.
param_path = 'D:\\ADNI_VAE\\ParamFiles\\IGLS_test_params.txt'
# param_path = sys.argv[1]  # Use this if running the code externally.
params = get_args(param_path)  # This will return a dictionary of parameters that are stored.

name = params["NAME"]
project_dir = os.path.join(params["PROJECT_DIR"], name)

if not os.path.isdir(project_dir):
    os.mkdir(project_dir)

project_files = os.listdir(project_dir)
h_flip = 0. if "H_FLIP" not in params.keys() else float(params["H_FLIP"])
v_flip = 0. if "V_FLIP" not in params.keys() else float(params["V_FLIP"])
h_flip = 0. if "RAND_ROTATE" not in params.keys() else float(params["RAND_ROTATE"])
batch_size = int(params["BATCH_SIZE"])
shuffle_batches = True if params['SHUFFLE_BATCHES'].lower() == 'true' else False
epochs = int(params["EPOCHS"])
save_epochs = int(params["SAVE_EPOCHS"])
recon_loss = True if params["RECON_LOSS"].lower() == 'true' else False
kl_loss = True if params["KL_LOSS"].lower() == 'true' else False
align_loss = True if params["ALIGN_LOSS"].lower() == 'true' else False
gamma = 1 if "GAMMA" not in params["GAMMA"] else float(params["GAMMA"])
lr = float(params["LR"]) if "LR" in params.keys() else 1e-4
momentum = float(params["MOMENTUM"]) if "MOMENTUM " in params.keys() else 0.9

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
    a.HorizontalFlip(p=0.),
    a.VerticalFlip(p=0.),
    a.RandomRotate90(p=0.),
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
if os.path.isdir(model_dir):
    model_list = os.listdir(model_dir)
    model_names = ['_'.join(m.split('_')[:-1]) for m in model_list]
    model_epochs = [int(m.split('_')[-1].replace('.h5', '')) for m in model_list]
    pre_epochs = max(model_epochs)
    model_name = model_names[model_epochs.index(pre_epochs)]
    try:
        model.load_state_dict(torch.load(model_name))
        print('Matched model keys successfully.')
    except NameError:
        print(f'Model matching unsuccessful: \n\t"{model_name}"')

else:
    os.mkdir(model_dir)
    pre_epochs = 0

model = model.to(device)

# ----------------------------------------- Initiate Optimiser ----------------------------------------------------

optimizer = torch.optim.SGD(list(model.parameters()),
                            lr=lr,
                            momentum=momentum)


# ----------------------------------------- Train Model ----------------------------------------------------

losses = []
for epoch in range(pre_epochs, pre_epochs + epochs):
    print(f'Epoch [{epoch + 1} / {epochs}]')

    epoch_losses = []
    for batch_no, batch in enumerate(dataloader):
        print(f'\tBatch [{batch_no+1} / {len(dataloader)}]')

        imgs = batch[0].to(device)
        subj_ids = batch[1].to(device)
        times = batch[2].to(device)

        pred, z_prior, z_post, cov_mat, beta, mean = model(imgs, subj_ids, times)

        loss, each_loss = lvae_loss(target=imgs,
                                    output=pred,
                                    prior_z=z_prior,
                                    post_z=z_post,
                                    mean=mean,
                                    cov_mat=cov_mat,
                                    bse=recon_loss,
                                    kl=kl_loss,
                                    align=align_loss,
                                    beta=beta,
                                    gamma=gamma
                                    )

        epoch_losses.append(each_loss)

    if epoch%save_epochs == 0:
        torch.save(model.state_dict(), os.path.join(model_dir, f'{name}_{epoch}.h5'))









# .to(device)

# test_imgs, test_ids, test_times = next(iter(dataloader))
#
# test_imgs = test_imgs.to(device)
# test_ids = test_ids.to(device)
# test_times = test_times.to(device)

model = VAE_IGLS(64).to(device)

# out, sig, beta, mu = model(test_imgs, test_ids, test_times)










