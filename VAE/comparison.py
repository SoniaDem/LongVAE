import numpy as np
from scipy.spatial import cKDTree


def images_to_latent(dataloader,
                     model,
                     tab_data_path,
                     device='cpu'):

    # The first line contains the headings.
    tab_data = np.loadtxt(tab_data_path, dtype=str)[1:]

    # Now pass the data through the model.
    model = model.to(device)
    latent_spaces = np.empty((0, 256))

    for batch in dataloader:
        batch = batch.to(device)
        x = model.encoder(batch)
        mu = model.linear_mu(x)
        log_var = model.linear_log_var(x)
        z = model.reparameterise(mu, log_var)
        z = z.cpu().detach().numpy()
        latent_spaces = np.concatenate((latent_spaces, z), axis=0)

    if tab_data.shape[0] == latent_spaces.shape[0]:
        latent_spaces = np.concatenate((tab_data, latent_spaces), axis=1)

    return latent_spaces


def get_nearest_neighbors(subject,
                          data,
                          k=3,
                          ):

    # The first 3 columns are the other data so don't use that in the nearest neighbour search.
    tree = cKDTree(data[:, 3:], leafsize=100)
    # Below is k+1 becuase it return the original queried subject with a distance of zero.
    dist, idx = tree.query(data[subject, 3:], k=k+1)
    dist = np.expand_dims(dist, 1)
    output_data = data[:, :3][idx]
    output_data = np.concatenate((output_data, dist), axis=1)
    return output_data






