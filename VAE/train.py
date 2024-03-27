import torch
import torch.nn.functional as F


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


def lvae_loss(target,
              output,
              prior_z,
              post_z,
              mu1,
              cov_mat1,
              mu2,
              cov_mat2,
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
    :param cov_mat1: The covariance of z_ijk.
    :param mu1: The mean of z_ijk.
    :param cov_mat2: The covariance matrices from each of the IGLS models of z_ijk.
    :param mu2: The mean values for each IGLS models of z_ijk.
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

        kl_loss = 0
        kl_loss = -0.5 * torch.sum(1 + cov_mat - (mean ** 2) - torch.exp(cov_mat))
        kl_loss /= torch.numel(mean.data)
        total_loss += (beta * kl_loss)
        losses[2] = (beta * kl_loss.item())
        raise Exception('Not implemented')

    if align:
        align_loss = F.mse_loss(prior_z, post_z, reduction='mean')
        total_loss += (gamma * align_loss)
        losses[3] = (gamma * align_loss.item())
        raise Exception('Not implemented')

    losses[0] = total_loss.item()

    return total_loss, losses



# def loss_fn(target,
#             output,
#             mean,
#             log_var,
#             beta=5):
#
#     # reproduction_loss = F.binary_cross_entropy(target, output, reduction='sum')
#     reproduction_loss = F.mse_loss(target, output, reduction='sum')
#     bce = torch.sum(0.5 * reproduction_loss)
#     kld = -0.5 * torch.sum(1 + log_var - (mean ** 2) - torch.exp(log_var))
#     kld /= torch.numel(mean.data)
#     loss = (beta * torch.sum(kld)) + torch.sum(bce)
#     return loss
#
#
# def train3d(n_epochs,
#             model,
#             dataloader,
#             optimizer,
#             device,
#             ):
#     """
#     This function is used as a simple training loop given the data,
#     a set of criteria and the number of epochs.
#     :param n_epochs: The number of epochs to train for.
#     :type n_epochs: positive int
#     :param model: The torch model to be trained.
#     :type model: torch.model
#     :param dataloader: The dataloader containing the data to train on.
#     :type dataloader: torch.data.DataLoader
#     :param optimizer: The optimizer used for backpropagation.
#     :type optimizer: torch.optim
#     :param device: The device that the model is on.
#     :type device: torch.device
#     :return:
#     """
#
#     model.train()
#     losses = []
#     for epoch in range(1, n_epochs + 1):
#         train_loss = 0
#         for batch in dataloader:
#             optimizer.zero_grad()
#             batch = batch.to(device).type(torch.cuda.FloatTensor)
#             y_pred, mean, log_var = model(batch)
#             loss = loss_fn(batch, y_pred, mean, log_var)
#             # print(loss.item())
#             train_loss += loss.item()
#             loss.backward(inputs=list(model.parameters()), retain_graph=True)
#             optimizer.step()
#
#         train_loss = train_loss / len(dataloader)
#         losses.append(train_loss)
#         print('Epoch: {}\tTraining Loss: {:.6f}'.format(epoch, train_loss))
#     return losses

