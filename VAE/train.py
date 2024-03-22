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
    reproduction_loss = F.mse_loss(target, output, reduction='sum')
    bce = torch.sum(0.5 * reproduction_loss)
    kld = -0.5 * torch.sum(1 + log_var - (mean ** 2) - torch.exp(log_var))
    kld /= torch.numel(mean.data)
    loss = (beta * torch.sum(kld)) + torch.sum(bce)
    return loss

#
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