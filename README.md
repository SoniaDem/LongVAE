# Longitudinal Mixed Effects VAEGAN 

## Description

Here is a description a brief description about the data, model and purpose.

## Training
`python LVAEGAN.py param_file.txt`

This executable script takes the experiment parameters from `param_file.txt`, creating the
following project folders/files.
   
- Models/
- Latent Parameters/
- Logs/
- project_name_loss.txt



| Parameter Name            | Description                                                              |
| :-----------------------  | :----------------------------------------------------------------------- |    
| NAME                      | The name of the project. |
| PROJECT_DIR               | Specify the directory to save the project in. |
| IMAGE_DIR                 | Specify the directory where the images are located. |
| SUBJ_DIR                  | Specify the directory where the subject information (.csv file) is located. |
| VERSION                   | Options are `1` or `2`. |
| Z_DIM                     | The size of the size of the latent dimension. (Default `64`) |
| MIXED_MODEL               | If `True` then the IGLS estimator will be applied to the latent space. (Default `False`) |
| GAN                       | If `True` then a discriminator will be used to train the model (like a GAN). (Default `False`) |
| IGLS_ITERATIONS           | The number of IGLS iterations used in the estimation. (Default `1`) |
| SLOPE                     | This specifies whether or not to estimate the individual gradient, a1. (Default `False`). |
| INCLUDE_A01               | This specifies whether or not to estimate the covariance, a01. (Default `False`). |
| SAVE_LATENT               | If `True` then the latent variable `z_ijk` will be saved and overwritten every minibatch. (Default `False`) |
| USE_SAMPLER               | If `True` or MIXED_MODEL is `True` then the custom sampler will be used forces each batch to have subjects with a number of time points specified by SAMPLER_PARAMS. (Default `False`) |
| SAMPLER_PARAMS            | Two integer values defining boundaries for the number of time points sampled from each subject. (Default `4, 6`) |
| MIN_DATA                  | This specifies the minimum number of time points that will be sampled from each subject. If a subject has less than this number of time points then they will not be included in training. (Default `4`) |
| BATCH_SIZE                | The size of the batch. (Default `100`) |
| SHUFFLE_BATCHES           | If `True` then the batches will be randomly sampled. (Default `False`) |
| H_FLIP                    | The probability of the image being horizontally flipped. (Default `0.`) |
| V_FLIP                    | The probability of the image being vertically flipped. (Default `0.`) |
| EPOCHS                    | The number of epochs to train for, starting from 0 if the project is new, or the latest save model. (Default `100`) |
| LR                        | This is the learning rate of the main model. (Default `1e-5`) |
| D_LR                      | When the GAN is true, the discriminator has it's own optimizer with it's own learning rate. (Default `1e-5`). |
| RECON_LOSS                | If `True` then the reconstruction loss between the input image and output image will be used in training (Default `True`) |
| ALIGN_LOSS                | If `True` then the loss between z_ijk from the encoder and z_hat from the mixed effect model will be used in training. (Default `False`) |
| KL_LOSS                   | This will **only** work with VERSION 2. If `True` then use KL loss. (Default `False`) |
| D_LOSS                    | If `True` then the GAN discriminator will be trained. (Default `False`). |
| BETA                      | This parameter specifies the factor by which the KL loss will be multiplied to contribute to the total loss. (Not recommended. Default `1`). |
| GAMMA                     | This parameter specifies the factor by which the align loss contributes to the total loss. (Default `1`) |
| NU                        | This parameter specifies the factor by which the discriminator loss contributes to the total loss. (Default `1`) |

## Plotting Reconstructions

`python output_comparisons param_file.txt`

Given a set of parameters, specifying a model and subject, a plot of the input image will be displayed adjacent to the reconstruction. 

The following parameters are specified and can remain in the `param_file.txt` used for training without effecting the training. 
Many of the parameters used for training are also required for the visualisation of the reconstruction. 

| Parameter Name            | Description                                                              |
| :-----------------------  | :----------------------------------------------------------------------- |    
| PLOT_EPOCH                | When using this parameter file for evaluation, this is the number of epochs corresponding to the model being evaluation. (Default `100`) |
| IMAGE_NO                  | The specific subject you want to visualise. (Default `0`) |
| TIMEPOINT                 | The specific time point index from the subject to visualise (Default `0`) |


## Other Stuff to describe later.



    
    


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
   
    