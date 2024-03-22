"""
Given an input of a directoryt, we extract images and downsample them to put in a different
directory.
22/09/2023
"""

from VAE.dataloader import extract_image_nifti
from VAE.utils import print_prog
from glob import glob
import os

in_dir = "D:\\norm_subjects\\nuyl"
out_dir = "D:\\norm_subjects\\nuyl_2x2_down"

inpaths = glob(in_dir+"\\*.nii.gz")

for i, path in enumerate(inpaths):
    print_prog(i, inpaths)
    extract_image_nifti(path,
                        out_dir,
                        scale=2)

# This is to add '4x4' to the names
# import subprocess
# start_names = glob(out_dir + '\\*')
# end_names = [f.replace('.nii', '_4x4.nii') for f in start_names]
# for start, end in zip(start_names, end_names):
#     print(start)
#     subprocess.call(['mv', start, end])

