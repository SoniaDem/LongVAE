"""
This code is just a tester for the VAE to load some of the data and make sure the VAE fits to the data.

Joe/Sonia 06/03/2023

------------------------

Now this code is to test out a new encoder and decoder architecture and to make sure the shapes are correct.

Joe 03/05/2024
"""

# ----------------------------------------- Load Packages ----------------------------------------------------
import os
import sys
from glob import glob
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn import Conv3d, BatchNorm3d, Linear, Flatten, Unflatten, Sigmoid, ConvTranspose3d, \
    Module, Sequential, LeakyReLU
import albumentations as a


from get_params import get_params
from VAE.dataloader import LongDataset, SubjectBatchSampler

# ----------------------------------- Set up project and load parameters -----------------------------------------------

# path = sys.argv[1]
path = 'D:\\Projects\\SoniaVAE\\ParamFiles\\IGLS_test_params.txt'
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

print("Loaded packages and parameter file.")

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

    n_batches = 0
    data_size = 0
    # for batch in dataloader:
    #     n_batches += 1
    #     data_size += batch[0].shape[0]

else:
    dataloader = DataLoader(dataset=loaded_data,
                            num_workers=0,
                            batch_size=params["BATCH_SIZE"],
                            shuffle=params["SHUFFLE_BATCHES"])
    n_batches = len(dataloader)
    data_size = len(dataloader.dataset)

print(f"Loaded data: \n\tTotal data points {data_size},")


# ----------------------------------------- Improvised Code ----------------------------------------------------

img, subj_ids, times = next(iter(dataloader))
img = img.to(device)

# ---------------------- model stuff -------------------------

class Encoder3d(Module):
    def __init__(self):
        super(Encoder3d, self).__init__()

        self.ConvBlock1 = Sequential(Conv3d(1, 8, kernel_size=3, stride=2, padding=1),
                                     BatchNorm3d(8),
                                     LeakyReLU(),
                                     Conv3d(8, 8, kernel_size=3, stride=1, padding=1),
                                     BatchNorm3d(8),
                                     LeakyReLU()
                                     )

        self.ConvBlock2 = Sequential(Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
                                     BatchNorm3d(16),
                                     LeakyReLU(),
                                     Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
                                     BatchNorm3d(16),
                                     LeakyReLU()
                                     )

        self.ConvBlock3 = Sequential(Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
                                     BatchNorm3d(32),
                                     LeakyReLU(),
                                     Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
                                     BatchNorm3d(32),
                                     LeakyReLU()
                                     )

        self.flatten = Flatten()

    def forward(self, x):
        x = self.ConvBlock1(x)
        print('ConvBlock1', x.shape)
        x = self.ConvBlock2(x)
        print('ConvBlock2', x.shape)
        x = self.ConvBlock3(x)
        print('ConvBlock3', x.shape)
        x = self.flatten(x)
        print('flat', x.shape)
        return x


class Decoder3d(Module):
    def __init__(self, z_dims):
        super(Decoder3d, self).__init__()

        self.leakyrelu = LeakyReLU()

        self.unflatten = Sequential(Linear(z_dims, 32*7*6*6),
                                    UnflattenManual3d(),
                                    LeakyReLU())
        self.ConvBlock3 = Sequential(ConvTranspose3d(32, 32, kernel_size=3, stride=1, padding=1),
                                     BatchNorm3d(32),
                                     LeakyReLU(),
                                     ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     BatchNorm3d(16),
                                     LeakyReLU())
        self.ConvBlock2 = Sequential(ConvTranspose3d(16, 16, kernel_size=3, stride=1, padding=1),
                                     BatchNorm3d(16),
                                     LeakyReLU(),
                                     ConvTranspose3d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     BatchNorm3d(8),
                                     LeakyReLU())
        self.ConvBlock1 = Sequential(ConvTranspose3d(8, 8, kernel_size=3, stride=1, padding=1),
                                     BatchNorm3d(8),
                                     LeakyReLU(),
                                     ConvTranspose3d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     Sigmoid())

    def forward(self, x):
        x = self.unflatten(x)
        print('unflat', x.shape)
        x = self.ConvBlock3(x)
        print('cb 3', x.shape)
        x = self.ConvBlock2(x)
        print('cb 2', x.shape)
        x = self.ConvBlock1(x)
        print('cb 1', x.shape)

        return x

#

class UnflattenManual3d(Module):
    def forward(self, x):
        return x.view(x.size(0), 32, 7, 6, 6)


lin = Linear(8064, 64)
lin = lin.to(device)

enc = Encoder3d()

print('img', img.shape)
enc = enc.to(device)
out = enc(img)

out2 = lin(out)
print('lin', out2.shape)

dec = Decoder3d(64)
dec = dec.to(device)
out3 = dec(out2)

print('out', out3.shape)