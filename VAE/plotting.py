import torch
import matplotlib.pyplot as plt
import numpy as np
from VAE.train import loss_txt_to_array
from VAE.utils import moving_average


def plotting_predictions(model,
                         dataloader,
                         device='cpu',
                         cmap='GnBu',
                         samples=1,
                         size=8,
                         save=None):
    """
    This is a function for plotting the input image, passing it through the model,
    plotting the output and then plotting the difference between them. The number of
    images can be input.
    :param model: The deep learning model.
    :type model: class
    :param dataloader: The dataloader containing the input images.
    :type dataloader: torch.data.DataLoader
    :param device: Cuda or cpu. (default :obj:`cpu`).
    :type device: torch.device
    :param cmap: The colour of the output images.
    :type cmap: str
    :param samples: The number of images to pass through the model. (default :obj:`1`).
    :type samples: positive int
    :param size: The output is square and this determines the size of that square.
    :type size: int
    :param save: To save the figure, enter the path including the figure name.
    :type save: str
    :return:
    """

    model = model.to(device)
    model.eval()
    counter = 0
    plt.figure(figsize=(size, int(samples * size)))
    for batch in dataloader:
        batch = batch.to(device)
        batch_pred, mean, log_var = model(batch)
        batch = batch.cpu().detach().numpy()
        batch_pred = batch_pred.cpu().detach().numpy()
        diff = np.abs(np.subtract(batch, batch_pred))

        for i in range(batch.shape[0]):
            if counter < samples:
                print(counter)

                plt.subplot(samples, 3, 1 + (counter*3))
                plt.imshow(batch[i, 0, :, :], cmap=cmap)
                plt.axis('equal')
                plt.axis('off')
                plt.title('Input')

                plt.subplot(samples, 3, 2 + (counter*3))
                plt.imshow(batch_pred[i, 0, :, :], cmap=cmap)
                plt.axis('equal')
                plt.axis('off')
                plt.title('Recreation')

                plt.subplot(samples, 3, 3 + (counter*3))
                plt.imshow(diff[i, 0, :, :], cmap=cmap)
                plt.axis('equal')
                plt.axis('off')
                plt.title('Difference')

                counter += 1
            else:
                break

        if counter == samples:
            break
    if save is not None:
        save = save + '.png' if save[-3:] != 'png' else save
        plt.savefig(save, dpi=300)
    plt.show()


def plot_loss(losses):
    """
    Simple function for plotting the loss.
    :param losses: A list of the losses over each epoch.
    :type losses: list
    :return:
    """
    plt.figure(figsize=(8, 8))
    plt.plot([i for i in range(len(losses))], losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Training')
    plt.show()


def plot_predictions_slices(x,
                            model,
                            index=0,
                            device='cpu'
                            ):
    """
    This method is to take the index of an image in the batch and passing it through the model.
    Then plot 3 slices side-by-side with the predictions from the model.
    :param x: This is one batch from the dataloader.
    :type x: torch.tensor
    :param model: The model to be evaluated.
    :type model: class
    :param index: The index of the image to display within the batch. (default :obj:`0`)
    :type index: int
    :param device: The device to compute the algorithm on. (default :obj:`cpu`)
    :type device: torch.device
    :return:
    """

    # First get the output from the model.
    x = x.to(device)
    model = model.to(device)
    y, _, _ = model(x)

    x = x.cpu()
    y = y.cpu().detach().numpy()

    # Now both the input and output should be of shape (batch, 1, 128, 128, 40)
    plt.figure(figsize=(15, 20))

    # Plot first slice side-by-side
    plt.subplot(3, 2, 1)
    plt.imshow(x[index, 0, :, :, 5], cmap='Greys_r')
    plt.axis('off')
    plt.axis('equal')
    plt.title('Input')

    plt.subplot(3, 2, 2)
    plt.imshow(y[index, 0, :, :, 5], cmap='Greys_r')
    plt.axis('off')
    plt.axis('equal')
    plt.title('Prediction')

    plt.subplot(3, 2, 3)
    plt.imshow(x[index, 0, :, :, 15], cmap='Greys_r')
    plt.axis('off')
    plt.axis('equal')
    plt.title('Input')

    plt.subplot(3, 2, 4)
    plt.imshow(y[index, 0, :, :, 15], cmap='Greys_r')
    plt.axis('off')
    plt.axis('equal')
    plt.title('Prediction')

    plt.subplot(3, 2, 5)
    plt.imshow(x[index, 0, :, :, 25], cmap='Greys_r')
    plt.axis('off')
    plt.axis('equal')
    plt.title('Input')

    plt.subplot(3, 2, 6)
    plt.imshow(y[index, 0, :, :, 25], cmap='Greys_r')
    plt.axis('off')
    plt.axis('equal')
    plt.title('Prediction')

    plt.show()


def plot_slice_prediction(x, y, axis=0, title=None):
    """
    This method takes the axis as input for the 3 image. Then in that axis,
    takes the middle slice and plots this slice from the input and output.

    :param x:
    :param y:
    :param axis:
    :param title:
    :return:
    """

    # Get middle slice.
    slice_id = int(x.shape[axis] / 2)

    plt.figure(figsize=(6, 3.5))

    plt.subplot(1, 2, 1)
    if axis == 0:
        plt.imshow(x[slice_id, :, :], cmap='Greys_r')
    if axis == 1:
        plt.imshow(x[:, slice_id, :], cmap='Greys_r')
    if axis == 2:
        plt.imshow(x[:, :, slice_id], cmap='Greys_r')
    plt.axis('off')
    plt.axis('equal')
    plt.title('Ground Truth')

    plt.subplot(1, 2, 2)
    if axis == 0:
        plt.imshow(y[slice_id, :, :], cmap='Greys_r')
    if axis == 1:
        plt.imshow(y[:, slice_id, :], cmap='Greys_r')
    if axis == 2:
        plt.imshow(y[:, :, slice_id], cmap='Greys_r')
    plt.axis('off')
    plt.axis('equal')
    plt.title('Prediction')

    if title is not None:
        plt.suptitle(f'{title} - Axis {axis}, Slice {slice_id}', fontsize=15, weight='bold')

    plt.tight_layout()
    plt.show()


def plot_losses(losses,
                start,
                window=None,
                save=None):
    """
    Either receive the loss path or an array of losses.
    The array should be formatted as (epochs, 4)
    4 = total loss, recon loss, kl loss and align loss.
    :param losses:
    :param start:
    :param window: Is specified then will plot the moving average with a window of this size.
    :param save:
    :return:
    """

    losses = loss_txt_to_array(losses) if type(losses) == str else losses
    losses = losses[:, start:]

    if window is not None:
        losses = moving_average(losses, window)
        epochs = list(range(start + window, losses.shape[1]))
    else:
        epochs = list(range(start, losses.shape[1]))

    n = losses.shape[0]
    c = 3 if n == 5 else 2

    plt.figure(figsize=(6.5, 6.5))

    plt.subplot(c, 2, 1)
    plt.plot(epochs, losses[0, :], color='tab:red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Total Loss', weight='bold')

    plt.subplot(c, 2, 2)
    plt.plot(epochs, losses[1, :], color='tab:blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Reconstruction Loss', weight='bold')

    plt.subplot(c, 2, 3)
    plt.plot(epochs, losses[2, :], color='tab:orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Alignment Loss', weight='bold')

    plt.subplot(c, 2, 4)
    plt.plot(epochs, losses[3, :], color='tab:green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('KL Divergence Loss', weight='bold')

    if n == 5:
        plt.subplot(c, 2, 5)
        plt.plot(epochs, losses[4, :], color='gold')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Discriminator Loss', weight='bold')

    plt.tight_layout()
    plt.show()

