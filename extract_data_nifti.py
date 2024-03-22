"""
In this file we load the .gz nifty data and we scale, convert and save it to a .gz nifty format

Joe/Sonia 23/06/2023
"""

import pandas as pd
from VAE.dataloader import extract_image_nifti
from VAE.utils import print_prog
from glob import glob

out_path = 'D:\\ADNI_VAE\\adni_5fold.csv'

cv_split = pd.read_csv(out_path)
image_paths = cv_split['image_path'].tolist()

out_dir = 'E:\\ADNI_downsampled\\'

for i, in_path in enumerate(image_paths):
    print_prog(i, image_paths)
    extract_image_nifti(in_path, out_dir, 4)

new_paths = [d for d in glob(out_dir + '*')]
cv_split['Downsampled_Image_Paths'] = new_paths
cv_split.to_csv(out_path, index=False)
