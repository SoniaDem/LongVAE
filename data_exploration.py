from glob import glob
from os import listdir
import pandas as pd
import numpy as np
import nibabel as nib

# Define the root directory.
root = "D:\\Data\\Brain MRI Dataset of Multiple Sclerosis\\"

# List all the items in the root directory.
patient_list = listdir(root)
# Specify the patient whose folder to look at
patient_id = 1
# List the file in that patients directory.
patient_images = [f for f in glob(root + f'Patient-{patient_id}\\*')]
# Image type
img_type = '1-T1.nii'
image_path = [img for img in patient_images if img.split('\\')[-1] == img_type][0]
# Load the image
image = nib.load(image_path).get_fdata()