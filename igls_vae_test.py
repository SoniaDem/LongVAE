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

from VAE.models import VAE_IGLS
from VAE.dataloader import LongDataset

# ----------------------------------------- Load data ----------------------------------------------------

# Retrieve list of image paths
root_path = 'D:\\norm_subjects\\nuyl_4x4_down\\'
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

batch_size = 50
dataloader = DataLoader(dataset=loaded_data,
                        batch_size=batch_size,
                        num_workers=0,
                        shuffle=False)

test1, test2, test3 = next(iter(dataloader))

test1 = test1.to(device)
test2 = test2.to(device)
test3 = test3.to(device)


model = VAE_IGLS(64).to(device)





