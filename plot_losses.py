from VAE.plotting import plot_losses
from VAE.train import loss_txt_to_array


project_name = 'IGLS_v2_0'
path = f'D:\\ADNI_VAE\\Projects\\{project_name}\\{project_name}_loss.txt'


loss_lines = [l.strip('\n') for l in open(path, 'r')]

losses = loss_txt_to_array(path)

plot_losses(losses)
