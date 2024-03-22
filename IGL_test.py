import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------------------------------------

ni = 50
nj = 7

beta0 = 2
beta1 = 5
sigma2_a0 = 0.9
sigma2_a1 = 1
sigma_a01 = 0.3
sigma_e = 0.5

mean = [0,0]
covariance = [[sigma2_a0, sigma_a01],[sigma_a01, sigma2_a1]]
effs = np.random.multivariate_normal(mean, covariance, ni)

ids = []
effs_i = []
effs_s = []
for i in range(ni):
    ids.extend([i+1] * nj)
    effs_i.extend([effs[i][0]] * nj)
    effs_s.extend([effs[i][1]] * nj)

data = pd.DataFrame({'id': ids,
                     'time': list(range(1, nj+1)) * ni,
                     'effs_i': effs_i,
                     'effs_s': effs_s})

z_ij = []
for i in range(ni * nj):
    z = beta0 + effs_i[i] + ((beta1 + effs_s[i]) * data['time'].iloc[i]) + np.random.normal(0, sigma_e)
    z_ij.append(z)

data['z_ij'] = z_ij

# -----------------------------------------------------------------------------------------------------------

nnij = data.shape[0]

z1 = np.zeros((nnij, nnij))
z2 = np.zeros((nnij, nnij))
z3 = np.zeros((nnij, nnij))
z4 = np.zeros((nnij, nnij))

for i in range(nnij):
    for j in range(nnij):
        subj_i = data['id'].iloc[i]
        subj_j = data['id'].iloc[j]

        visit_i = data['time'].iloc[i]
        visit_j = data['time'].iloc[j]

        if subj_i == subj_j and visit_i == visit_j:
            z1[i, j] = 1
        if subj_i == subj_j:
            z2[i, j] = 1
            z3[i, j] = visit_i + visit_j
            z4[i, j] = visit_i * visit_j


# z2 = z2.T + z2
# np.diag(z2) = np.diag(z2) / 2

xx = np.ones((nnij, 2))
xx[:, 1] = data['time']
b1 = np.linalg.inv(np.matmul(xx.T, xx))
b2 = np.matmul(xx.T, np.asarray(data['z_ij']))
betahat = np.matmul(b1, b2)


vz1 = np.expand_dims(z1.flatten('F'), axis=-1)
vz2 = np.expand_dims(z2.flatten('F'), axis=-1)
vz3 = np.expand_dims(z3.flatten('F'), axis=-1)
vz4 = np.expand_dims(z4.flatten('F'), axis=-1)

zz = np.concatenate((vz1, vz2, vz3, vz4), axis=1)

sig_memory = []
iter = 10
k = 1

while k <= iter:
    print('Iteration', k)
    zhat = betahat[0] + betahat[1] * np.asarray(data['time'])
    ztilde = np.expand_dims(zhat - np.asarray(data['z_ij']), axis=-1)
    ztz = np.matmul(ztilde, ztilde.T)
    ztz = ztz.flatten('F')

    sig_est1 = np.linalg.inv(np.matmul(zz.T, zz))
    sig_est2 = np.matmul(zz.T, ztz)
    sig_est = np.matmul(sig_est1, sig_est2)

    s_e = sig_est[0]
    s_a0 = sig_est[1]
    s_a01 = sig_est[2]
    s_a1 = sig_est[3]

    sigma_update = np.zeros((nnij, nnij))
    for i in range(nnij):
        for j in range(nnij):
            subj_i = data['id'].iloc[i]
            subj_j = data['id'].iloc[j]

            visit_i = data['time'].iloc[i]
            visit_j = data['time'].iloc[j]

            if subj_i == subj_j:
                sigma_update[i, j] = s_a0

            if subj_i == subj_j and visit_i == visit_j:
                sigma_update[i, j] = s_e + s_a0 + (s_a01 * z3[i, j]) + (s_a1 * z4[i, j])

    b1 = np.linalg.inv(xx.T @  np.linalg.inv(sigma_update) @ xx)
    b2 = xx.T @  np.linalg.inv(sigma_update) @ np.asarray(data['z_ij'])
    betahat = b1 @ b2

    sig_memory.append(sig_est.tolist())
    k += 1


for i in range(len(sig_memory)):
    print(sig_memory[i])