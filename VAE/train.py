import numpy as np
import torch
import torch.nn as nn
from torch import bmm, inverse, log, det, exp
import torch.nn.functional as F
from VAE.utils import batch_diag


def train_ae(n_epochs,
             model,
             dataloader,
             optimizer,
             loss_function,
             device,
             ):
    """
    This function is used as a simple training loop given the data,
    a set of criteria and the number of epochs.
    :param n_epochs: The number of epochs to train for.
    :type n_epochs: positive int
    :param model: The torch model to be trained.
    :type model: torch.model
    :param dataloader: The dataloader containing the data to train on.
    :type dataloader: torch.data.DataLoader
    :param optimizer: The optimizer used for backpropagation.
    :type optimizer: torch.optim
    :param loss_function: The loss function used to determine the models
        changes in back propagation.
    :type loss_function: torch.nn.$Loss$
    :param device: The device that the model is on.
    :type device: torch.device
    :return:
    """

    losses = []
    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            batch = batch.to(device).type(torch.cuda.FloatTensor)
            y_pred = model(batch)
            loss = loss_function(y_pred, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*batch.size(0)

        train_loss = train_loss / len(dataloader)
        losses.append(train_loss)
        print('Epoch: {}\tTraining Loss: {:.6f}'.format(epoch, train_loss))
    return losses


def train(n_epochs,
          model,
          dataloader,
          optimizer,
          device,
          ):
    """
    This function is used as a simple training loop given the data,
    a set of criteria and the number of epochs.
    :param n_epochs: The number of epochs to train for.
    :type n_epochs: positive int
    :param model: The torch model to be trained.
    :type model: torch.model
    :param dataloader: The dataloader containing the data to train on.
    :type dataloader: torch.data.DataLoader
    :param optimizer: The optimizer used for backpropagation.
    :type optimizer: torch.optim
    :param device: The device that the model is on.
    :type device: torch.device
    :return:
    """

    model = model.to(device)
    model.train()
    losses = []
    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            batch = batch.to(device).type(torch.cuda.FloatTensor)
            y_pred, mean, log_var = model(batch)
            loss = loss_fn(batch, y_pred, mean, log_var)
            # print(loss.item())
            train_loss += loss.item()
            loss.backward(inputs=list(model.parameters()), retain_graph=True)
            optimizer.step()

        train_loss = train_loss / len(dataloader)
        losses.append(train_loss)
        print('Epoch: {}\tTraining Loss: {:.6f}'.format(epoch, train_loss))
    return losses


def loss_fn(target,
            output,
            mean,
            log_var,
            beta=5):

    # reproduction_loss = F.binary_cross_entropy(target, output, reduction='sum')
    reproduction_loss = F.mse_loss(target, output, reduction='mean')
    bce = torch.sum(0.5 * reproduction_loss)
    kld = -0.5 * torch.sum(1 + log_var - (mean ** 2) - torch.exp(log_var))
    kld /= torch.numel(mean.data)
    loss = (beta * torch.sum(kld)) + torch.sum(bce)
    return loss


def d_loss(target,
           output):
    loss = nn.BCELoss(reduction='mean')
    return loss(target, output)


def lvae_loss(target,
              output,
              prior_z,
              post_z,
              mu,
              cov_mat,
              igls_vars=None,
              beta=1,
              gamma=1,
              bse=True,
              kl=True,
              align=True):
    """
    This function calculates the loss for the longitudinal VAE.
    It consists of 3 components:
        1) reproduction of input and output image,
        2) KL divergence
        3) alignment loss between z_ijk and z_hat (denoted as z_prior and z_post).

    Along with the total loss value, a list is returned for the losses of each of the individual losses.
    If these losses are not included then they will be returned as 0's. The losses are returned as
    [total loss, reconstruction loss, kl loss, align loss].

    :param target: The ground truth input image.
    :param output: The image output from the model.
    :param prior_z: z_ijk which is output from the encoder and the linear layer.
    :param post_z: z_hat which is following the IGLS estimation method and sampling of
        the multivariate normal distribution.
    :param cov_mat: The covariance of z_ijk.
    :param mu: The mean of z_ijk.
    :param igls_vars: a matrix of ([sig_a0, sig_a1, sig_e], k_dims)
    :param beta: A parameter for weighing the importance of the KL divergence loss on the total loss.
    :param gamma: A parameter for weighing the importance of the alignment loss on the total loss.
    :param bse: If True then implement the reproduction loss.
    :param kl: If True then implement the KL diverge loss.
    :param align: If true then implement the alignment loss.
    :return:
    """

    total_loss = 0
    losses = [0] * 4
    if bse:
        reproduction_loss = F.mse_loss(target, output, reduction='mean')
        bce_loss = torch.sum(torch.sum(0.5 * reproduction_loss))
        total_loss += bce_loss
        losses[1] = bce_loss.item()

    if kl:
        a0_kl = -0.5 * torch.sum(1 + torch.log(igls_vars[0]) - igls_vars[0])
        a0_kl /= torch.numel(igls_vars[0])
        a1_kl = -0.5 * torch.sum(1 + torch.log(igls_vars[1]) - igls_vars[1])
        a1_kl /= torch.numel(igls_vars[1])
        e_kl = -0.5 * torch.sum(1 + torch.log(igls_vars[2]) - igls_vars[2])
        e_kl /= torch.numel(igls_vars[2])
        kl_loss = (a0_kl + a1_kl + e_kl) / 3

        total_loss += (beta * kl_loss)
        losses[2] = (beta * kl_loss.item())

    if align:
        align_loss = F.mse_loss(prior_z, post_z, reduction='mean')
        total_loss += (gamma * align_loss)
        losses[3] = (gamma * align_loss.item())

    losses[0] = total_loss.item()

    return total_loss, losses


def lvaegan_loss(target,
                 output,
                 d_output,
                 d_labels,
                 prior_z,
                 post_z,
                 mu,
                 cov_mat,
                 igls_vars=None,
                 beta=1,
                 gamma=1,
                 bse=True,
                 disc_loss=True,
                 align=True):
    """
    This function calculates the loss for the longitudinal VAE.
    It consists of 3 components:
        1) reproduction of input and output image,
        2) disciminator loss
        3) alignment loss between z_ijk and z_hat (denoted as z_prior and z_post).

    Along with the total loss value, a list is returned for the losses of each of the individual losses.
    If these losses are not included then they will be returned as 0's. The losses are returned as
    [total loss, reconstruction loss, kl loss, align loss].

    :param target: The ground truth input image.
    :param output: The image output from the model.
    :param d_output: The image and reconstruction after being passed through the discriminator.
    :param d_labels: The labels containing  0s and 1s for the discriminator.
    :param prior_z: z_ijk which is output from the encoder and the linear layer.
    :param post_z: z_hat which is following the IGLS estimation method and sampling of
        the multivariate normal distribution.
    :param cov_mat: The covariance of z_ijk.
    :param mu: The mean of z_ijk.
    :param igls_vars: a matrix of ([sig_a0, sig_a1, sig_e], k_dims)
    :param beta: A parameter for weighing the importance of the KL divergence loss on the total loss.
    :param gamma: A parameter for weighing the importance of the alignment loss on the total loss.
    :param bse: If True then implement the reproduction loss.
    :param disc_loss: If True then implement the KL diverge loss.
    :param align: If true then implement the alignment loss.
    :return:
    """

    total_loss = 0
    losses = [0] * 5
    if bse:
        reproduction_loss = F.mse_loss(target, output, reduction='mean')
        bce_loss = torch.sum(torch.sum(0.5 * reproduction_loss))
        total_loss += bce_loss
        losses[1] = bce_loss.item()

    if align:
        align_loss = F.mse_loss(prior_z, post_z, reduction='mean')
        total_loss += (gamma * align_loss)
        losses[2] = (gamma * align_loss.item())

    if disc_loss:
        loss_d = d_loss(d_output, d_labels)
        total_loss += (beta * loss_d)
        losses[3] = (beta * loss_d.item())

    losses[0] = total_loss.item()

    return total_loss, losses


def lvaegan2_loss(target,
                  output,
                  lin_z_hat,
                  mm_z_hat,
                  lin_mu,
                  lin_logvar,
                  mm_mu,
                  mm_var,
                  d_output=None,
                  d_labels=None,
                  beta=1,
                  gamma=1,
                  nu=1,
                  recon=True,
                  kl=True,
                  align=True,
                  disc_loss=True):
    """
    This function calculates the loss for the longitudinal VAE.
    It consists of 4 components:
        1) reproduction of input and output image,
        2) KL divergence,
        3) alignment loss between z_ijk and z_hat (denoted as z_prior and z_post),
        4) discriminator loss.

    Along with the total loss value, a list is returned for the losses of each of the individual losses.
    If these losses are not included then they will be returned as 0's. The losses are returned as
    [total loss, reconstruction loss, kl loss, align loss].

    :param target: The ground truth input image.
    :param output: The image output from the model.
    :param lin_z_hat: lin_z_hat is the output from sampling from a normal distribution from mu and sigma from
        linear layers.
    :param mm_z_hat: This is the output from the IGLS model.
    :param lin_mu: The mean from linear layer
    :param lin_logvar: The log variance from the linear layer
    :param mm_mu: The mean from mixed model (beta0 + beta1 * t).
    :param mm_var: The variance form the mixed model.
    :param d_output: The image and reconstruction after being passed through the discriminator.
    :param d_labels: The labels containing  0s and 1s for the discriminator.
    :param beta: A parameter for weighing the importance of the KL divergence loss on the total loss.
    :param gamma: A parameter for weighing the importance of the alignment loss on the total loss.
    :param nu: A parameter for weighing the importance of the discriminator loss on the total loss.
    :param recon: If True then implement the reproduction loss.
    :param kl: If True then implement the KL diverge loss.
    :param align: If true then implement the alignment loss.
    :param disc_loss: If true then implement the discriminator loss.
    :return:
    """

    total_loss = 0
    losses = [0] * 5
    if recon:
        reproduction_loss = F.mse_loss(target, output, reduction='mean')
        bce_loss = torch.sum(torch.sum(0.5 * reproduction_loss))
        total_loss += bce_loss
        losses[1] = bce_loss.item()

    if kl:

        # Convert the variations to diagonal matrices. Need to do this for each subject within the batch
        lin_cov_mat = batch_diag(exp(lin_logvar).T).double()  # (k, b, b)
        lin_cov_mat = lin_cov_mat.to(lin_logvar.device)
        mm_cov_mat = batch_diag(mm_var.T).double()  # (k, b, b)
        mm_cov_mat = mm_cov_mat.to(mm_var.device)
        mm_mu = mm_mu.T.unsqueeze(-1)  # (k, b, 1)
        lin_mu = lin_mu.T.unsqueeze(-1)  # (k, b, 1)

        mm_cov_mat_inv = inverse(mm_cov_mat)  # (k, b, b)

        kl0 = (bmm(mm_cov_mat_inv, lin_cov_mat)).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # (z)
        # print('kl0', kl0.shape)
        mu_diff = mm_mu - lin_mu  # (k, b, 1)
        # print('mu_diff', mu_diff.shape)
        mu_diff_t = mu_diff.transpose(2, 1)  # (k, 1, b)
        # print('mu_diff.t', mu_diff_t.shape)
        kl1 = bmm(mu_diff_t, mm_cov_mat_inv.float())  # (k, 1, b)
        # print('kl1', kl1.shape)
        kl1_1 = bmm(kl1, mu_diff).squeeze(-1).squeeze(-1)  # (k)
        # print('kl1_1', kl1_1.shape)
        kl2 = log(det(mm_cov_mat) / det(lin_cov_mat))  # (k)
        kl2 = kl2.float()
        # print('kl2', kl2.shape)
        kl_tot = 0.5 * (kl0 - lin_logvar.shape[0] + kl1_1 + kl2)  # (64)
        # print('kl_tot', kl_tot.shape)
        kl_tot = beta * kl_tot.mean()
        total_loss += kl_tot
        losses[4] += kl_tot.item()
        # print('kl_tot', kl_tot)

        # raise Exception('Not implemented')

    if align:
        align_loss = F.mse_loss(lin_z_hat, mm_z_hat, reduction='mean')
        total_loss += (gamma * align_loss)
        losses[2] = (gamma * align_loss.item())
        # raise Exception('Not implemented')

    if disc_loss:
        loss_d = d_loss(d_output, d_labels)
        total_loss += (nu * loss_d)
        losses[3] = (nu * loss_d.item())

    losses[0] = total_loss.item()

    return total_loss, losses


def loss_txt_to_array(path):
    """
    The loss txt file has one line which is structured as:
        total_loss\trecon_loss\tkl_loss\talign_loss\n
    :param path:
    :return:
    """
    loss_lines = [l.strip('\n') for l in open(path, 'r')]
    loss_lines = [list(map(float, l.split(' '))) for l in loss_lines]
    return np.asarray(loss_lines).T

