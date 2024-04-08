from VAE.plotting import plot_losses
from VAE.train import loss_txt_to_array


project_name = 'IGLS_noa01'
path = f'D:\\ADNI_VAE\\Projects\\{project_name}\\{project_name}_loss.txt'
# path = f'D:\\ADNI_VAE\\Projects\\IGLS_v1\\IGLS_zijk_no_slope_loss.txt'

loss_lines = [l.strip('\n') for l in open(path, 'r')]

losses = loss_txt_to_array(path)

plot_losses(losses[:, 3000:])
plot_losses(losses)

# --------------------------------------- unrelated stuff --------------------------------------------
#
# from torch import tensor
# import pandas as pd
#
# t = tensor([[-3.0537e-02,  4.4004e-03,  2.7408e-02,  3.1124e-03,  2.7135e-03,
#           4.9436e-03,  8.8195e-03,  6.4958e-03,  2.4258e-02,  1.3057e-02,
#           4.7954e-03,  5.5620e-03,  1.8054e-03,  2.1687e-02,  3.4970e-03,
#           1.6058e-03,  6.9812e-03,  3.9799e-03,  5.1496e-03,  2.6361e-03,
#           4.4956e-03,  2.8701e-02,  9.8469e-03,  1.0240e-02,  6.8072e-03,
#           2.6982e-03,  1.5621e-03,  9.0027e-03,  3.6680e-03,  4.2742e-03,
#           1.9314e-03,  9.1636e-03,  9.9442e-03,  1.1520e-02,  2.4367e-03,
#           3.0948e-02,  4.3803e-02,  6.0642e-02,  2.8542e-03,  2.0869e+00,
#           5.3611e+01,  4.0679e-03,  4.2372e-03,  1.4882e-02,  3.9244e-03,
#           1.3322e-03,  2.4096e-03,  5.5348e-03,  1.5032e-02,  4.2442e-03,
#           1.6880e-03,  7.1391e-03,  6.6701e+02,  7.3036e-03,  3.6399e-03,
#           2.8672e-03,  4.8786e-03,  6.6910e-02,  7.9668e-03,  6.9361e-03,
#           5.3949e-03,  5.4985e-03,  6.3472e-03,  6.7340e-03],
#         [ 1.8248e+01,  1.1660e-03, -1.1846e-02,  3.7137e-03, -9.2979e-05,
#           1.0762e-02, -5.6588e-03,  2.9405e-04, -2.6150e-03, -1.9273e-03,
#           6.2820e-03,  2.0120e-03,  3.5495e-03, -7.8040e-03, -2.6213e-04,
#           3.7024e-03,  5.2486e-03,  1.9528e-03,  2.9067e-03,  1.4235e-03,
#          -1.5289e-03, -5.6432e-03,  1.2166e-03,  1.2514e-03,  2.8440e-03,
#           1.3215e-03,  1.7587e-03,  5.3218e-04,  5.1956e-03,  1.0973e-03,
#          -3.8675e-05, -4.9271e-03, -6.2725e-03,  5.1092e-04,  4.4330e-03,
#          -1.0168e-02, -6.5963e-03, -2.0929e-02,  6.7171e-03, -8.4263e-01,
#           2.7643e+02,  2.2861e-03,  6.9138e-03, -4.5642e-03,  2.3308e-03,
#           5.8936e-03, -5.4334e-04, -1.5533e-03, -2.3752e-03, -2.4017e-04,
#           2.8457e-03, -1.9128e-03, -2.6308e+02, -2.2623e-03,  2.9932e-03,
#           6.8469e-05,  1.7503e-04, -2.6617e-02, -1.2882e-03, -3.7569e-03,
#           4.5771e-04,  1.5579e-03, -1.4298e-03,  3.2870e-03],
#         [-7.0070e+00,  2.4597e-04,  2.0949e-03, -5.9518e-04,  2.2325e-04,
#           6.4502e-04,  2.8206e-03,  1.8117e-03,  1.7544e-03,  8.2372e-04,
#           1.1513e-04,  3.6211e-04,  8.6840e-04,  1.4302e-03,  2.9630e-04,
#          -2.4608e-04, -6.1010e-04, -7.2724e-05,  1.0744e-04,  1.2426e-04,
#           7.3760e-04,  2.5357e-03,  1.5881e-03, -3.3750e-04,  2.9505e-04,
#          -4.0144e-05,  6.6287e-05, -5.5278e-05,  1.1864e-03,  1.9780e-04,
#           2.3318e-04,  9.2872e-04,  1.5474e-03, -3.2272e-04, -5.1666e-04,
#           2.2352e-03,  1.7723e-03,  3.1505e-03,  2.4672e-04,  3.2950e-01,
#          -6.8705e+00,  1.5815e-04, -1.2645e-04,  6.3546e-04, -6.3853e-05,
#          -4.8933e-04,  3.2827e-04,  5.2796e-04,  8.3359e-04,  4.9092e-04,
#          -1.1934e-04,  2.4078e-04,  3.6347e+01,  5.1990e-04, -4.3217e-04,
#           7.8610e-05,  3.2286e-04,  5.4338e-03,  4.1611e-04,  8.0846e-04,
#          -2.6625e-04,  1.4565e-04,  3.5749e-04,  4.1201e-04],
#         [ 2.6937e+00, -4.9407e-05, -3.4927e-04,  1.2534e-04, -9.6433e-06,
#           3.5821e-05, -9.8256e-05, -2.8815e-04, -4.1739e-04, -1.6829e-04,
#           9.6173e-05, -9.1966e-05, -8.9409e-05, -3.6110e-04, -8.0257e-05,
#           2.9584e-05,  2.3461e-04,  2.8135e-05, -1.3193e-04, -6.1651e-05,
#           1.7104e-04, -5.5315e-04, -1.5847e-04,  1.1973e-04, -6.1700e-05,
#           1.5186e-05, -3.9648e-05, -2.4243e-05,  3.3685e-04,  6.2965e-06,
#          -4.9675e-05, -2.0568e-04, -2.2790e-04,  7.5283e-05,  7.3783e-05,
#          -4.6620e-04, -5.9758e-04, -6.8088e-04, -3.5576e-05, -5.6794e-02,
#          -8.1923e-01, -6.6512e-05, -1.0775e-05, -9.6036e-05,  5.4179e-05,
#           4.8541e-05, -5.2609e-05, -1.4619e-04, -2.2342e-04, -1.0738e-04,
#           2.0649e-05,  2.2697e-05, -6.8157e+00, -1.3825e-05,  1.7992e-04,
#           1.0893e-04, -4.8842e-05, -1.1694e-03, -1.0676e-04, -1.3916e-04,
#           7.6671e-05, -6.2783e-05, -1.2095e-04, -1.2069e-04]])
#
# times = tensor([3, 1, 5, 7, 3, 3, 1, 7, 5, 7, 5, 2, 1, 5, 7, 3, 2, 1, 4, 7, 2, 5, 3, 3,
#         1, 5, 4, 2, 1, 3, 2, 7, 1, 3, 4, 2, 4, 3, 2, 1, 1, 2, 5, 7, 2, 4, 5, 3,
#         7, 1, 3, 4, 5, 1, 2, 5, 3, 3, 5, 1, 2, 5, 1, 3, 7, 4, 1, 3, 5, 2, 5, 2,
#         3, 1, 5, 7, 3, 2, 1, 4, 2, 3, 3, 2, 7, 5, 1, 2, 7, 4, 1, 3, 3, 4, 1, 2,
#         5, 5, 3, 2])
#
# subj_id = tensor([103, 309, 309, 309, 309,  82,  82,  82,  82, 215, 215, 215, 215,  36,
#          36,  36,  36,  36, 106, 106, 106, 106, 106, 300, 300, 300, 300, 300,
#         181, 181, 181, 181, 154, 154, 154, 154, 220, 220, 220, 220,  35,  35,
#          35,  35, 221, 221, 221, 221, 221, 162, 162, 162, 162,  33,  33,  33,
#          33, 359, 359, 359, 359,  65,  65,  65,  65,  65, 130, 130, 130, 130,
#          48,  48,  48,  48, 192, 192, 192, 192, 106, 106, 106, 106, 296, 296,
#         296, 296, 296, 176, 176, 176, 176, 176, 189, 189, 189, 189, 189, 273,
#         273, 273])


# import torch
# file = 'D:\\Projects\\SoniaVAE\\Projects\\IGLS_test_zijk\\Latent Params\\z_ijk.pt'
# t = torch.load(file).detach().cpu().numpy()
# t_df = pd.DataFrame(t)
# t_df.to_csv('D:\\Projects\\SoniaVAE\\Projects\\IGLS_test_zijk\\Broken Z\\z_ijk_not_broken_3421.csv', index=False)
