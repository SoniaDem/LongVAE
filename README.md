# LongVAE

### Sonia's VAE to apply to ADNI.

Below is a description of the files and their intended method of running.

1. `python testing_code_conv.py`

    This code can be run as it is. Within the file you can change the number of epochs. It already has the path
    to save the model to and so will save the future models with the same name in the same directory but with the
    number of epochs. For example, `D:\\models\\vae_1000.h5`. You can change within the file after how many epochs
    the model should be saved.


2. `python output_comparison.py $EPOCHS`
    
    This code loads in the data, passes it through a trained model and then view it next to the ground truth. Assuming
    that the model directory and prefix are correctly specified in the file, the only argument that is required to
    be specified is the epoch number for the model you would like to use. 


3. `python get_latent.py $EPOCHS`

    This code passes all the data through the model to get mu and log_var. Then it reparameterises and returns the 
    latent vector z. For each subject this becomes a row and is saved as a csv. If specified then this latent space 
    can be reduced to 2d and plotted to see the distribution.


4. `python plot_loss.py loss_file.txt`
   
    This code takes the file containing the losses through training and plots them. It assumes the loss file is 
    formatted as train: `train: loss_val\nval: loss_val\n`


5. `python extract_data.py`

    Within the file, you specify the path for the images and the directory that you would like to save the new files.
    The files are reformatted from .gz to .pt, and you can specify whether to scale them down or not before saving. 


6. `python create_train_test.py`

    This takes all data specified, splits these into k cross validation folds as specified within the file. 
    It splits the validation in half to be used as validation and test. It then saves this as a csv.


7. `python data_exploration.py`
    
    A throwaway file that we used to understand how the data looked. 
   
    