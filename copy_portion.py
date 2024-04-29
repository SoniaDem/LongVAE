"""
  This code takes a number of epochs and then copys the data from one directory upto that specified
  number of epochs. This allows the model to continue to be trained whilst keeping the progress from 
  the original.
  
  Takes 2 arguments:
   - Project
   - Epochs
"""

# --------------------------------- Import packages -------------------------------------------
import sys
import os
from os.path import join
import subprocess


project = sys.argv[1]
epochs = int(sys.argv[2])

projects_dir = '/nobackup/projects/bdlds05/scjps/LVAE/Projects'
project_dir = join(projects_dir, project)

model_dir = join(project_dir, 'Models')
models = os.listdir(model_dir)
model_epochs = [int(m.split('_')[-1][:-3]) for m in models]

print('')
if epochs in model_epochs:
    print(f"Model {epochs} exists.")
else:
  raise Exception(f"Model {epochs} does not exist.")
  
new_project = f'{project}_e{epochs}'
new_dir = join(projects_dir, new_project)
print(f'Creating directory: {new_project}')

os.mkdir(new_dir)
print(f'Created new directory: {new_project}')
new_model_dir = join(new_dir, 'Models')
os.mkdir(new_model_dir)

model_start =  join(model_dir, models[model_epochs.index(epochs)])
new_model_name = models[model_epochs.index(epochs)].replace(project, f"{project}_e{epochs}")
model_end = join(new_model_dir, new_model_name)
print(f'Copying model\n\tStart path: {model_start}\n\tEnd path: {model_end}')
subprocess.call(['cp', model_start, model_end])
print('Copied model')

print('Copying loss')
new_loss = f"{project}_e{epochs}_loss.txt"
old_loss = f"{project}_loss.txt"

loss_file = open(join(project_dir, old_loss), 'r')
loss_lines = loss_file.readlines()[:epochs]
loss_file.close()
new_losses = open(join(new_dir, new_loss), 'w+')
for l in loss_lines:
    new_losses.write(l)
new_losses.close()

print(f'Copied losses. len(loss) = {len(open(join(new_dir, new_loss), "r").readlines())}') 

print('\n\tDone\n')

