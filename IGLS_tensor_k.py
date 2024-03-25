"""
This IGLS estimation method from Sonia's work but in tensor
format and using the correct dimensions to be used in a VAE.
"""

# --------------------------------------------------------------------------------------------------------------

from torch import tensor, repeat_interleave, normal, zeros, ones, eye, inverse, flatten, cat, bmm, mul, add
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
from random import randint

# --------------------------------------------------------------------------------------------------------------

# Define parameters
ni = 50  # number of subjects
nj = 7  # number of time points
k_dims = 100  # the size of the latent space variable z

# Define an empty z_ijk matrix
z_ijk = zeros((k_dims, ni * nj))

# Define variables that are the same for each k
times = torch.arange(1, nj+1).repeat(ni)
subj_ids = repeat_interleave(torch.arange(1, ni+1), nj)

initial_params = []
for k in range(k_dims):
    # Define parameters for this z dimension
    # beta0 = randint(1, 10)
    # beta1 = randint(1, 10)
    # sigma2_a0 = randint(5, 9) / 10
    # sigma2_a1 = randint(5, 15) / 10
    # sigma_a01 = randint(1, 5) / 10
    # sigma_e = 0.5

    beta0 = 2
    beta1 = 5
    sigma2_a0 = 0.9
    sigma2_a1 = 1
    sigma_a01 = 0.3
    sigma_e = 0.5

    # Append these parameters to a list for later comparison following the complete estimation.
    initial_params.append([beta0, beta1, sigma2_a0, sigma2_a1, sigma_a01, sigma_e])

    mean = zeros(2)
    covariance = tensor([[sigma2_a0, sigma_a01],
                         [sigma_a01, sigma2_a1]])
    effs = MultivariateNormal(loc=mean,
                              covariance_matrix=covariance).sample([ni])

    effs_i = repeat_interleave(effs[:, 0], nj)
    effs_s = repeat_interleave(effs[:, 1], nj)

    # This creates z values at each subject and time point for this specific z dimension (k).
    z_ij = beta0 + effs_i + ((beta1 + effs_s) * times) + normal(mean=0, std=sigma_e, size=(ni * nj,))
    z_ijk[k, :] = z_ij.T

z_ijk = z_ijk.T

# --------------------------------------------------------------------------------------------------------------

# Intial calculation of betahat

nnij = z_ijk.shape[0]
z1 = eye(nnij)
z2 = zeros((nnij, nnij))
z3 = zeros((nnij, nnij))
z4 = zeros((nnij, nnij))

for i in range(nnij):
    for j in range(nnij):
        subj_i = subj_ids[i]
        subj_j = subj_ids[j]

        visit_i = times[i]
        visit_j = times[j]

        if subj_i == subj_j:
            z2[i, j] = 1
            z3[i, j] = visit_i + visit_j
            z4[i, j] = visit_i * visit_j


# z_ijk.shape = ([100, k_dims])
xx = ones((k_dims, nnij, 2))  # size (k_dims, 100, 2)
xx[:, :, 1] = times.repeat(k_dims, 1)  # size (k_dims, 100, 2)
b1 = inverse(bmm(xx.transpose(2, 1), xx))  # following bmm, the size is (k_dims, 2, 2). This will do the inverse of
                                            # each (2, 2) matrix.
b2 = bmm(xx.transpose(2, 1), z_ijk.expand(1, -1, -1).transpose(2, 0))
betahat = bmm(b1, b2)   # size (k_dims, 2, 1) so you have [[[b0, b0]], [[b1, b1]]]

vz1 = flatten(z1.transpose(1, 0)).expand(1, -1).T  # size (10000, 1)
vz2 = flatten(z2.transpose(1, 0)).expand(1, -1).T
vz3 = flatten(z3.transpose(1, 0)).expand(1, -1).T
vz4 = flatten(z4.transpose(1, 0)).expand(1, -1).T
zz = cat((vz1, vz2, vz3, vz4), axis=1)  # size (10000, 4)

z1 = z1.repeat(k_dims, 1, 1)      # size (k_dims, 100, 100)
z2 = z2.repeat(k_dims, 1, 1)
z3 = z3.repeat(k_dims, 1, 1)
z4 = z4.repeat(k_dims, 1, 1)

# --------------------------------------------------------------------------------------------------------------


def expand_vec(mat, vec):
    """
        As an example, there is a matrix of ones with size (3, 2, 2). We want to multiply
        the vector [1, 2, 3] so we get:
        [[[1  1],
          [1  1]],

         [[2  2],
          [2  2]],

         [[3  3],
          [3  3]]]
    """
    extra_dims = (1,) * (mat.dim() - 1)
    return vec.view(-1, *extra_dims)


# Estimation loop

sig_memory = []

iter = 10
k = 1

while k <= iter:
    print('Iteration', k)
    zhat = betahat[:, 0] + (betahat[:, 1] * times)  # size (k_dims, 100)
    ztilde = zhat.T - z_ijk  # size (100, k_dims)
    ztilde = ztilde.expand(1, -1, -1).transpose(2, 0)  # size (k_dims, 100, 1)
    ztz = bmm(ztilde, ztilde.transpose(2, 1))  # size (k_dims, 100, 100)
    ztz = flatten(ztz, start_dim=1, end_dim=2).T  # size (k_dims, 10000)

    sig_est = inverse(zz.T @ zz) @ (zz.T @ ztz)  # size (4, k_dims)

    s_e = expand_vec(z1, sig_est[0])
    s_a0 = expand_vec(z2, sig_est[1])
    s_a01 = expand_vec(z3, sig_est[2])
    s_a1 = expand_vec(z4, sig_est[3])

    sigma_update = (s_e * z1) + (s_a0 * z2) + (s_a01 * z3) + (s_a1 * z4) # size (k_dims, 100, 100)

    b1 = inverse(bmm(bmm(xx.transpose(2, 1), inverse(sigma_update)), xx))  # size (k_dims, 2, 2)
    # b2 size (k_dims, 2, 1)
    b2 = bmm(bmm(xx.transpose(2, 1), inverse(sigma_update)), z_ijk.expand(1, -1, -1).transpose(2, 0))
    betahat = bmm(b1, b2)

    sig_memory.append(sig_est.T.tolist())

    k += 1


for m in sig_memory[-1]:
    print('')
    print(m)

print('')
sig_memory = tensor(sig_memory[-1])
for i in range(4):
    print(sig_memory[:, i].mean().item(), sig_memory[:, i].std().item())

# square_diffs = []
# estimates = []
# for k in range(k_dims):
#     print('')
#     init_s_a0, init_s_a01, init_s_a1, init_s_e = initial_params[k][2:]
#     est_s_e, est_s_a0, est_s_a01, est_s_a1 = sig_memory[-1][k]
#     print(f'{k} Inital:    {init_s_a0}\t\t\t{init_s_a01}\t\t\t{init_s_a1}\t\t\t{init_s_e}')
#     print(f'{k} Estimates: {est_s_a0:.6f}\t{est_s_a01:.6f}\t{est_s_a1:.6f}\t{est_s_e:.6f}')
#
#     diff_e = (init_s_e - est_s_e) ** 2
#     diff_s_a0 = (init_s_a0 - est_s_a0) ** 2
#     diff_s_a01 = (init_s_a01 - est_s_a01) ** 2
#     diff_s_a1 = (init_s_a1 - est_s_a1) ** 2
#     square_diffs.extend([diff_e, diff_s_a0, diff_s_a01, diff_s_a1])
#
# rmse = (sum(square_diffs) / len(square_diffs)) ** 0.5

k = 0
cov_mat = sigma_update[k]
a = MultivariateNormal(loc=zeros(350), covariance_matrix=cov_mat).sample([1])

# sigma update (100, 350, 350)
mean = zeros((100, 350))
aa = MultivariateNormal(loc=mean,
                          covariance_matrix=sigma_update).sample([1])

