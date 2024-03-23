from torch.nn import Conv2d, MaxPool2d, Linear, ConvTranspose2d, \
    BatchNorm2d, Flatten, Unflatten, Module, Sigmoid, Conv3d, \
    BatchNorm3d, ConvTranspose3d
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import sigmoid, exp, randn_like, tensor, repeat_interleave, \
    normal, zeros, ones, eye, inverse, flatten, cat, bmm, mul, add
import torch
from VAE.utils import expand_vec


class AE(Module):
    def __init__(self):
        super(AE, self).__init__()

        """ Encoder """
        self.conv1 = Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = Conv2d(16, 4, kernel_size=(3, 3), padding=1)
        self.pool = MaxPool2d(2, 2)
        self.lin1 = Linear(in_features=4*32*32, out_features=2048)

        """ Decoder """
        self.lin2 = Linear(in_features=2048, out_features=4*32*32)
        self.t_conv1 = ConvTranspose2d(4, 16, kernel_size=(2, 2), stride=(2, 2))
        self.t_conv2 = ConvTranspose2d(16, 1, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        """ Forward pass the data. """

        """ Encode """
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = Flatten()(x)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = Unflatten(1, (4, 32, 32))(x)

        """ Decode """
        x = F.leaky_relu(self.t_conv1(x))
        x = sigmoid(self.t_conv2(x))

        return x


class UnflattenManual(Module):
    def forward(self, x):
        return x.view(x.size(0), 32, 14, 14)


class Encoder(Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = Conv2d(1, 8, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.batch1 = BatchNorm2d(8)
        self.conv2 = Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.batch2 = BatchNorm2d(16)
        self.conv3 = Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.batch3 = BatchNorm2d(32)
        self.conv4 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.batch4 = BatchNorm2d(32)
        self.flatten = Flatten()

    def forward(self, x):
        x = self.batch1(self.conv1(x))
        x = F.leaky_relu(x)
        x = self.batch2(self.conv2(x))
        x = F.leaky_relu(x)
        x = self.batch3(self.conv3(x))
        x = F.leaky_relu(x)
        x = self.batch4(self.conv4(x))
        # print(x.shape)
        x = F.leaky_relu(x)
        x = self.flatten(x)
        return x


class Decoder(Module):
    def __init__(self, z_dims):
        super(Decoder, self).__init__()
        self.linear = Linear(z_dims, 32*4*7*7)
        self.unflatten = UnflattenManual()
        self.conv1 = ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.batch1 = BatchNorm2d(16)
        self.conv2 = ConvTranspose2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        self.batch2 = BatchNorm2d(16)
        self.conv3 = ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        self.batch3 = BatchNorm2d(8)
        self.conv4 = ConvTranspose2d(8, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        # print(x.shape)
        x = self.unflatten(x)
        # print(x.shape)
        x = self.batch1(self.conv1(x))
        x = F.leaky_relu(x)
        x = self.batch2(self.conv2(x))
        x = F.leaky_relu(x)
        x = self.batch3(self.conv3(x))
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x


class VAE(Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.linear_mu = Linear(6272, z_dim)
        self.linear_log_var = Linear(6272, z_dim)
        self.decoder = Decoder(z_dims=z_dim)

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        mu = self.linear_mu(x)
        # print(mu.shape)
        log_var = self.linear_log_var(x)
        # print(log_var.shape)
        z = self.reparameterise(mu, log_var)
        # print(z.shape)
        x = self.decoder(z)
        return x, mu, log_var

    def reparameterise(self, mu, log_var):
        std = exp(0.5 * log_var)
        e = randn_like(std)
        return mu + (std * e)


class Encoder3d(Module):
    def __init__(self):
        super(Encoder3d, self).__init__()

        self.conv1 = Conv3d(1, 8, kernel_size=3, stride=2, padding=1)
        self.batch1 = BatchNorm3d(8)
        self.conv2 = Conv3d(8, 16, kernel_size=3, stride=2, padding=1)
        self.batch2 = BatchNorm3d(16)
        self.conv3 = Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.batch3 = BatchNorm3d(32)
        self.conv4 = Conv3d(32, 32, kernel_size=3, stride=1, padding=0)
        self.batch4 = BatchNorm3d(32)
        self.flatten = Flatten()

    def forward(self, x):
        x = self.batch1(self.conv1(x))
        x = F.leaky_relu(x)
        x = self.batch2(self.conv2(x))
        x = F.leaky_relu(x)
        x = self.batch3(self.conv3(x))
        x = F.leaky_relu(x)
        x = self.batch4(self.conv4(x))
        # print(x.shape)
        x = F.leaky_relu(x)
        x = self.flatten(x)
        return x


class Decoder3d(Module):
    def __init__(self, z_dims):
        super(Decoder3d, self).__init__()
        self.linear = Linear(z_dims, 32*4*7*7*3)
        self.unflatten = UnflattenManual3d()
        self.conv1 = ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=0)
        self.batch1 = BatchNorm3d(16)
        self.conv2 = ConvTranspose3d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.batch2 = BatchNorm3d(16)
        self.conv3 = ConvTranspose3d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.batch3 = BatchNorm3d(8)
        self.conv4 = ConvTranspose3d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        # print(x.shape)
        x = self.unflatten(x)
        # print(x.shape)
        x = self.batch1(self.conv1(x))
        x = F.leaky_relu(x)
        x = self.batch2(self.conv2(x))
        x = F.leaky_relu(x)
        x = self.batch3(self.conv3(x))
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x


class UnflattenManual3d(Module):
    def forward(self, x):
        return x.view(x.size(0), 32, 14, 14, 3)


class VAE3d(Module):
    def __init__(self, z_dim):
        super(VAE3d, self).__init__()
        self.encoder = Encoder3d()
        self.linear_mu = Linear(18816, z_dim)
        self.linear_log_var = Linear(18816, z_dim)
        self.decoder = Decoder3d(z_dims=z_dim)

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        mu = self.linear_mu(x)
        # print(mu.shape)
        log_var = self.linear_log_var(x)
        # print(log_var.shape)
        z = self.reparameterise(mu, log_var)
        # print(z.shape)
        x = self.decoder(z)
        return x, mu, log_var

    def reparameterise(self, mu, log_var):
        std = exp(0.5 * log_var)
        e = randn_like(std)
        return mu + (std * e)


class VAE_IGLS(Module):
    def __init__(self, z_dim):
        super(VAE_IGLS, self).__init__()
        self.encoder = Encoder3d()
        self.linear_mu = Linear(18816, z_dim)
        self.linear_log_var = Linear(18816, z_dim)
        self.decoder = Decoder3d(z_dims=z_dim)
        self.igls_iterations = 10
        self.k_dims = z_dim


    def forward(self, x, subject_ids, times):

        print('x.shape', x.shape)
        print('subject_ids.shape', subject_ids.shape)
        print('times.shape', times.shape)

        self.batch_size = x.shape[0]
        z_ijk = self.encoder(x)
        print('z_ijk.shape', z_ijk.shape)

        sigma_est, betahat = self.igls(z_ijk, subject_ids, times)
        print('sigma_est.shape', sigma_est.shape)

        z_hat = self.igls_reparameterise(sigma_est)
        print('z_hat.shape', z_hat.shape)

        x = self.decoder(z_hat)
        print('x.shape', x.shape)

        return x, sigma_est, betahat

        # # print(x.shape)
        # mu = self.linear_mu(x)
        # # print(mu.shape)
        # log_var = self.linear_log_var(x)
        # # print(log_var.shape)
        # z = self.reparameterise(mu, log_var)
        # # print(z.shape)
        # x = self.decoder(z)
        # return x, mu, log_var
    #
    # def reparameterise(self, mu, log_var):
    #     std = exp(0.5 * log_var)
    #     e = randn_like(std)
    #     return mu + (std * e)

    def igls_reparameterise(self, cov_mat):
        return MultivariateNormal(loc=zeros(self.k_dims, self.batch_size),
                                  covariance_matrix=cov_mat).sample([1])  # size (1, k_dims, batch_size)


    def igls(self, z_ijk, subject_ids, times):

        z1 = eye(self.batch_size)
        z2 = zeros((self.batch_size, self.batch_size))
        z3 = zeros((self.batch_size, self.batch_size))
        z4 = zeros((self.batch_size, self.batch_size))

        for i in range(self.batch_size):
            for j in range(self.batch_size):

                subj_i = subject_ids[i]
                subj_j = subject_ids[j]

                visit_i = times[i]
                visit_j = times[j]

                if subj_i == subj_j:
                    z2[i, j] = 1
                    z3[i, j] = visit_i + visit_j
                    z4[i, j] = visit_i * visit_j

        xx = ones((self.k_dims, self.batch_size, 2))  # size (k_dims, batch_size, 2)
        xx[:, :, 1] = times.repeat(self.k_dims, 1)  # size (k_dims, batch_size, 2)
        b1 = inverse(bmm(xx.transpose(2, 1), xx))  # following bmm, the size is (k_dims, 2, 2). This will do the
        # inverse of  each (2, 2) matrix.
        b2 = bmm(xx.transpose(2, 1), z_ijk.expand(1, -1, -1).transpose(2, 0))
        betahat = bmm(b1, b2)  # size (k_dims, 2, 1)

        vz1 = flatten(z1.transpose(1, 0)).expand(1, -1).T  # size (batch_size^2, 1)
        vz2 = flatten(z2.transpose(1, 0)).expand(1, -1).T
        vz3 = flatten(z3.transpose(1, 0)).expand(1, -1).T
        vz4 = flatten(z4.transpose(1, 0)).expand(1, -1).T
        zz = cat((vz1, vz2, vz3, vz4), axis=1)  # size (batch_size^2, 4)

        z1 = z1.repeat(self.k_dims, 1, 1)  # size (k_dims, batch_size, batch_size)
        z2 = z2.repeat(self.k_dims, 1, 1)
        z3 = z3.repeat(self.k_dims, 1, 1)
        z4 = z4.repeat(self.k_dims, 1, 1)

        sig_est = zeros(4, self.k_dims)
        for _ in range(self.igls_iterations):
            zhat = betahat[:, 0] + (betahat[:, 1] * times)  # size (k_dims, batch_size)
            ztilde = zhat.T - z_ijk  # size (batch_size, k_dims)
            ztilde = ztilde.expand(1, -1, -1).transpose(2, 0)  # (k_dims, batch_size, 1)
            ztz = bmm(ztilde, ztilde.transpose(2, 1))  # size (k_dims, batch_size, batch_size)
            ztz = flatten(ztz, start_dim=1, end_dim=2).T  # size (k_dims, batch_size^2)

            # size (4, k_dims)
            sig_est = inverse(zz.T @ zz) @ (zz.T @ ztz)

            # size (k_dims, 1, 1)
            s_e = expand_vec(z1, sig_est[0])
            s_a0 = expand_vec(z2, sig_est[1])
            s_a01 = expand_vec(z3, sig_est[2])
            s_a1 = expand_vec(z4, sig_est[3])

            # size (k_dims, batch_size, batch_size)
            sigma_update = (s_e * z1) + (s_a0 * z2) + (s_a01 * z3) + (s_a1 * z4)

            # size (k_dims, 2, 2)
            b1 = inverse(bmm(bmm(xx.transpose(2, 1), inverse(sigma_update)), xx))
            # b2 size (k_dims, 2, 1)
            b2 = bmm(bmm(xx.transpose(2, 1), inverse(sigma_update)), z_ijk.expand(1, -1, -1).transpose(2, 0))
            # size (k_dims, 2, 1)
            betahat = bmm(b1, b2)

        return sig_est, betahat
