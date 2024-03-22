import albumentations as a
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import glob
import os

# Packages from this module
from VAE.dataloader import brain_data_loader
from VAE.models import AE
from VAE.train import train
from VAE import models, plotting
from VAE.plotting import plot_loss, plotting_predictions

# Prepare the data. ---------------------------------------------------------
transforms = a.Compose([
    a.HorizontalFlip(p=0.5),
    a.VerticalFlip(p=0.5),
    a.RandomRotate90(p=0.5),
    a.Transpose(p=0.5),
])

# Define the root directory
root_path = 'D:\\Data\\ds000003_R2.0.2_raw\\separate_inplaneT2_images\\'
# Retrieve all files in one list.
file_paths = [file for file in glob.glob(os.path.join(root_path, '*'))]

loaded_data = brain_data_loader(image_list=file_paths,
                                transforms=transforms,
                                batch_size=100,
                                n_duplicates=10)
test_data = next(iter(loaded_data))
print(f'Number of images in dataset = {len(loaded_data.dataset)}')


# Setup model. ---------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device = {device}')



# enc = models.Encoder()

# enc = enc.to(device)
vae = models.VAE(256)
vae = vae.to(device)
test_data = test_data.to(device)
out, mean, log_var = vae(test_data)

# out = enc(test_data)
# print(out.shape)
# print(test_data.shape)
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)


losses = train(n_epochs=200,
               model=vae,
               dataloader=loaded_data,
               optimizer=optimizer,
               device=device)

plotting_predictions(vae, loaded_data, samples=4, device=device)
