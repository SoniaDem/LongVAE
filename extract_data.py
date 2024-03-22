"""
In this file we load the .gz nifty data and we scale, convert and save it to a .pt format.

Joe/Sonia 06/06/2023
"""

import pandas as pd
import os
from VAE.dataloader import extract_image
from VAE.utils import print_prog

cv_path = 'D:\\ADNI_VAE\\CrossValidationFiles\\adni_5fold_all.csv'
cv_split = pd.read_csv(cv_path)
image_paths = cv_split['image_path'].tolist()

out_dir = 'D:\\ADNI_VAE\\ExtractedFiles\\'

out_paths = []
for i, in_path in enumerate(image_paths):
    print_prog(i, image_paths)
    extract_image(in_path, out_dir, 4)
    out_name = os.path.basename(in_path).split('.')[0] + '.pt'
    out_path = os.path.join(out_dir, out_name)
    out_paths.append(out_path)

cv_split['extracted_paths'] = out_paths
cv_split.to_csv(cv_path, index=False)

