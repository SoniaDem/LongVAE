import glob
import albumentations as a
from VAE import models, train
import torch
from VAE.dataloader import brain_data_loader


root_path = 'D:\\Data\\ds000003_R2.0.2_raw\\ds000003_R2.0.2\\'
paths = [file for file in glob.glob(root_path + '*\\*\\*.gz') if 'inplane' in file]

# test_file = nib.load(paths[0]).get_fdata()

transforms = a.Compose([
    a.HorizontalFlip(p=0.5),
    a.VerticalFlip(p=0.5),
    a.RandomRotate90(p=0.5),
])

loaded_data = brain_data_loader(image_list=paths,
                                transforms=transforms,
                                dims=3,
                                batch_size=16,
                                n_duplicates=100)

test_data = next(iter(loaded_data))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device = {device}')

vae = models.VAE3d(256)
vae = vae.to(device)
# test_data = test_data.to(device)
# #
# out = vae(test_data)
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

losses = train.train(1000,
                     vae,
                     loaded_data,
                     optimizer,
                     device
                     )

torch.save(vae.state_dict(), 'D:\\ConceptionX\\Theia.ai\\Software\\Models\\vae3d_13p_100d_1000e.h5')
vae.load_state_dict(torch.load('D:\\ConceptionX\\Theia.ai\\Software\\Models\\vae3d_13p_100e.h5'))

from vae.plotting import plot_predictions_slices

plot_predictions_slices(test_data, vae, 4, device)

loaded_data = brain_data_loader(image_list=paths,
                                transforms=transforms,
                                dims=3,
                                batch_size=16,
                                n_duplicates=1)


par_path = root_path + 'participants.tsv'
from vae.comparison import images_to_latent, get_nearest_neighbors

combined = images_to_latent(loaded_data, vae, par_path, device)

nns = get_nearest_neighbors(4, combined, k=5)