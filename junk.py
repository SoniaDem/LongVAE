import torch
from torch import inverse
torch.set_printoptions(threshold=90000)

project_name = 'IGLS_V1_32'
path = f'D:\\Projects\\SoniaVAE\\Projects\\{project_name}\\LatentParams\\z_ijk.pt'
sigma_update = torch.load(path)

inv_sig = inverse(sigma_update)

el18 = sigma_update[18, :, :]

torch.det(sigma_update)

from torch.linalg import pinv

pinv(sigma_update)
