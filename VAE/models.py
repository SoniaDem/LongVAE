from torch.nn import Conv2d, MaxPool2d, Linear, ConvTranspose2d, \
    BatchNorm2d, Flatten, Unflatten, Module, Sigmoid, Conv3d, \
    BatchNorm3d, ConvTranspose3d
import torch.nn.functional as F
from torch import sigmoid, exp, randn_like


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
