# ----------------------------- Load Packages --------------------------------------- 0.

from glob import glob
from os import listdir, path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# ----------------------------- Retrieve the image paths ---------------------------- 1.

# Define root path
root = 'E:\\subjects'

# First get the paths to all subject's folders.
subjects = [d for d in glob(path.join(root, '*'))]

# For each subject get the session folders.
sessions = []
for subject in subjects:
    # Get the sessions for each subject
    sub_sessions = [s for s in glob(path.join(subject, '*'))]
    # Rather than use append which would create a list of lists, use extend.
    sessions.extend(sub_sessions)

# I've checked and every session contains the 't1_linear' folder, so we will add this to the dir.
sessions = [path.join(s, 't1_linear') for s in sessions]

# Now we will get all cropped images within the sessions.
image_paths = []
for session in sessions:
    images = [im for im in glob(path.join(session, '*')) if 'Crop' in im]
    if len(images) > 0:
        image_paths.extend(images)

# There was one image that didn't fully download so removing it to avoid problems.
image_paths = image_paths[:-1]
print(f'1. Number of cropped t1 linear brain MRIs:  {len(image_paths)}')

# ----------------------------- Split the data --------------------------------------- 2.

df = pd.DataFrame({'image_path': image_paths})

k = 5
kf = KFold(n_splits=k, shuffle=True)

# This section will retrieve the indices for each fold,
# label then and add each fold as a column in the file.
for i, (train_id, test_id) in enumerate(kf.split(df)):
    # Define a list of zeros ('<U6' forces the format to be a string).
    train_test = np.empty((len(df)), dtype='<U6')
    # for the train indices, set these to 1
    train_test[train_id] = 'train'
    # Add a validation set.
    val_ids = np.random.choice(test_id, round(len(test_id) / 2), replace=False)
    train_test[val_ids] = 'val'
    # If the value = 1 then it is a train label, else it is a test value.
    train_test = np.where(train_test == '', 'test', train_test)
    # Assign labels to the dataframe.
    df[f"fold_{i}"] = train_test


for i in range(k):
    print(df[f'fold_{i}'].value_counts())

out_path = 'D:\\ADNI_VAE\\adni_5fold_all.csv'

if path.isfile(out_path):
    ow = str(input('This file already exists! Do you want to overwrite it? [y/n]\t'))
    if ow == 'y':
        df.to_csv(out_path, index=False)

else:
    df.to_csv(out_path, index=False)
    print(f'Saved {out_path}')

