"""
Given an input of a directoryt, we extract images and downsample them to put in a different
directory.
22/09/2023
"""

from VAE.utils import print_prog
from glob import glob
import nibabel as nib
import nibabel.processing
import nilearn
import os

in_dir = "D:\\norm_subjects\\mask"
out_dir = "D:\\norm_subjects\\mask_2x2_down"

def extract_image_nifti(path, out_dir, scale):
    '''
    :param path: path to the image
    :param out_dir: dierctory where we save the files
    :param scale: one integer that indicated how much we scale down an image
    :return:
    '''

    voxel_size = [scale, scale, scale]

    input_img = nib.load(path)
    resampled_img = nib.processing.resample_to_output(input_img, voxel_size)
    nib.save(resampled_img, out_dir)

inpaths = glob(in_dir+"\\*.nii.gz")

for i, path in enumerate(inpaths):
    print_prog(i, inpaths)
    out_path = os.path.join(out_dir, os.path.basename(path))
    extract_image_nifti(path,
                        out_path,
                        scale=2)

# This is to add '4x4' to the names
# import subprocess
# start_names = glob(out_dir + '\\*')
# end_names = [f.replace('.nii', '_4x4.nii') for f in start_names]
# for start, end in zip(start_names, end_names):
#     print(start)
#     subprocess.call(['mv', start, end])

