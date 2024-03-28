"""
    There is a problem. The subject IDs are in a string format (e.g. 'ADNI002S0295').
    However, this does not allow these to become tensors because torch.tensor only
    accepts ints, floats, etc.
    Therefore, this code is going to assign a key and save this key to a csv file so
    the subject can be identified with regard to the dataset.

    Joe 2024-03-23
"""

# ------------------------------------------------------------------------------------------------------
from glob import glob
import os
import pandas as pd

# ------------------------------------------------------------------------------------------------------

# Determine image directory
root_path = 'D:\\norm_subjects\\nuyl_4x4_down\\'

# Retrieve list of image paths (including the image names).
paths = glob(root_path + '*')

# Isolate the image names
name_list = [os.path.basename(n) for n in paths]

# The image name is formatted as 'sub-ADNI941S1194_ses...nii.gz'. We just want the subject id 'ADNI...'
subj_ids = [n.replace('_', '-').split('-')[1] for n in name_list]

# Some subjects have more than one image so get a list of unique subjects. ('set()' creates a tuple)
unique_ids = list(set(subj_ids))

# Order these just so if anything fails we can return to just ordering the subjects.
unique_ids = sorted(unique_ids)

# Determine a set of unique numerical characters for each subject.
char_ids = list(range(1, len(unique_ids) + 1))

# Save this key as a csv file
key_df = pd.DataFrame({'ADNI_ID': unique_ids,
                       'NUM_ID': char_ids})
cwd = os.getcwd()
key_df.to_csv(os.path.join(cwd, 'subject_id_key.csv'), index=False)




