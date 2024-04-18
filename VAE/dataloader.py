import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import Sampler
from torchvision.transforms import ToTensor
import numpy as np
import nibabel as nib
from os.path import basename


class BrainDataset2D(Dataset):
    def __init__(self, image_list, transformations):

        self.image_list = image_list
        self.transformations = transformations

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        """
        Here we are going to be iterating through each image,
        transforming it and converting to a tensor.
        """
        # This produces a numpy array of shape (n_images, h, w)
        image = np.load(self.image_list[idx])
        # Transform the image given the transformations.
        transformed = self.transformations(image=image)
        # Retrieve the transformed image.
        trans_im = transformed['image']
        # Subtract the mean.
        # trans_im -= 240.8326
        # The normalization from albumentations only works on rgb images.
        norm = (trans_im - np.min(trans_im)) / (np.max(trans_im) - np.min(trans_im))
        # Convert to tensor.
        tensor_image = ToTensor()(norm)
        tensor_image = tensor_image.type(torch.FloatTensor)
        return tensor_image


def brain_data_loader(image_list,
                      transforms,
                      dims=2,
                      n_duplicates=1,
                      batch_size=4,
                      num_workers=0,
                      shuffle=False,
                      ):
    """
    This method will take the list of paths to the images and will apply
    the albumentations.transforms. This will produce one dataset.
    The method also takes an argument to say how many datasets to produce.
    Given non-binary transformations with p > 0, the datasets will be different.
    :param image_list: List containing the path to the images.
    :type image_list: list
    :param transforms: A set of transforms to apply to the images.
    :type transforms: albumentations.compose()
    :param dims: The number of dimensions that the data possesses. (default :obj:`2`)
    :type dims: int
    :param n_duplicates: The number of augmented datasets to produce. (default :obj:`1`)
    :type n_duplicates: positive int
    :param batch_size: Number of instances in a batch. (default :obj:`1`)
    :type batch_size: positive int
    :param num_workers: The number of subprocesses to use for data loading. A default of 0
        results in the data loader, loading data sequentially. (default :obj:`0`)
    :type num_workers: int
    :param shuffle: Whether or not to shuffle the instances in a batch. (default :obj:`True`)
    :type shuffle: bool
    :return:
    """
    # Create a list to store the datasets.
    datasets = []
    # Iterate through the number of duplicates and append to list of datasets.
    if dims == 2:
        for _ in range(n_duplicates):
            datasets.append(BrainDataset2D(image_list=image_list,
                                           transformations=transforms))
    if dims == 3:
        for _ in range(n_duplicates):
            datasets.append(BrainDataset3D(image_list=image_list,
                                           transformations=transforms))

    # Combine the datasets.
    combined_dataset = ConcatDataset(datasets)
    # Convert into a dataloader.
    dataloader = DataLoader(dataset=combined_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=shuffle)
    return dataloader


class BrainDataset3D(Dataset):
    def __init__(self, image_list, transformations):

        self.image_list = image_list
        self.transformations = transformations

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        """
        Here we are going to be iterating through each image,
        transforming it and converting to a tensor.
        """
        # This produces a numpy array of shape (n_images, h, w)
        image = nib.load(self.image_list[idx]).get_fdata()
        transformed = self.transformations(image=image)
        # Retrieve the transformed image.
        trans_im = transformed['image']
        # Subtract the mean.
        norm = (trans_im - np.min(trans_im)) / (np.max(trans_im) - np.min(trans_im))
        # I think i will also add blank region at the bottom of the image.
        filler = np.zeros((128, 128, 7))
        norm = np.concatenate((norm, filler), axis=2)
        # Convert to tensor.
        tensor_image = ToTensor()(norm)
        tensor_image = tensor_image.unsqueeze(dim=0)
        tensor_image = tensor_image.transpose_(3, 1)
        tensor_image = tensor_image.type(torch.FloatTensor)
        return tensor_image


class LongDataset(Dataset):
    def __init__(self, image_list, subject_key, transformations, min_data=None):

        self.image_list = image_list
        self.subject_key = subject_key
        self.transformations = transformations

        self.subj_dict = {}
        for idx in range(self.__len__()):
            image_name = basename(self.image_list[idx])
            image_name = image_name.replace('_', '-').split('-')
            subject_id_adni = image_name[1]
            subject_id_num = self.subject_key[self.subject_key["ADNI_ID"] == subject_id_adni]['NUM_ID'].item()
            if subject_id_num not in self.subj_dict:
                self.subj_dict[subject_id_num] = [idx]
            else:
                self.subj_dict[subject_id_num].append(idx)

        if min_data is not None:
            rm_subj = [k for k in self.subj_dict.keys() if len(self.subj_dict[k]) < min_data]
            for k in rm_subj:
                self.subj_dict.pop(k)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        """
        Here we are going to be iterating through each image,
        transforming it and converting to a tensor.
        """
        # This produces a numpy array of shape (n_images, h, w)
        image = nib.load(self.image_list[idx]).get_fdata()
        image_name = basename(self.image_list[idx])
        image_name = image_name.replace('_', '-').split('-')

        time_list = [0, 6, 12, 18, 24, 30, 36]
        time = int(image_name[3][1:])
        times = time_list.index(time) + 1

        subject_id_adni = image_name[1]
        subject_id_num = self.subject_key[self.subject_key["ADNI_ID"] == subject_id_adni]['NUM_ID'].item()

        transformed = self.transformations(image=image)
        # Retrieve the transformed image.
        trans_im = transformed['image']
        # Subtract the mean.
        norm = (trans_im - np.min(trans_im)) / (np.max(trans_im) - np.min(trans_im))
        # I think i will also add blank region at the bottom of the image.
        filler = np.zeros((48, 56, 48))
        filler[2:45, 2:55, 1:47] = norm
        # Convert to tensor.
        tensor_image = ToTensor()(filler)
        tensor_image = tensor_image.unsqueeze(dim=0)
        tensor_image = tensor_image.transpose_(3, 1)
        tensor_image = tensor_image.type(torch.FloatTensor)
        return tensor_image, subject_id_num, times


class SubjectBatchSampler(Sampler):
    """
    Ensure that there are a specified minimum number of data points for each unique subject in each batch.

    Args:
        subject_dict: A dictionary containing {subject_id: [times]}
        batch_size: size of mini-batch
        min_data: Minimum number of data points for each unique subject.
        max_data: Maximum number of data points for each unique subject.
    """

    def __init__(self, subject_dict, batch_size, min_data=3, max_data=6):
        # build data sampling here
        self.subj_dict = subject_dict
        self.batch_size = batch_size
        self.min_data = min_data
        self.max_data = max_data

    def __iter__(self):
        # implement logic for sampling here

        # Create a random sample of subject ids
        rand_subj = torch.randint(low=0, high=len(self.subj_dict.keys()), size=(len(self.subj_dict.keys()),))

        batch = []
        for i in rand_subj:
            subj_id = list(self.subj_dict.keys())[i]
            sample_times = torch.randint(low=self.min_data, high=self.max_data, size=(1,)).item()
            # print('sample times', sample_times)
            # print(len(self.subj_dict[subj_id]))
            sample_size = len(self.subj_dict[subj_id]) if len(self.subj_dict[subj_id]) < sample_times else sample_times

            rand_times = torch.randperm(len(self.subj_dict[subj_id]))[:sample_size]

            # rand_times = torch.randint(low=0, high=len(self.subj_dict[subj_id]), size=(sample_size,))
            # rand_times = torch.randperm(sample_size)
            for t in rand_times:
                batch.append(self.subj_dict[subj_id][t.item()])
                # print(self.subj_dict[subj_id][t.item()])
                if len(batch) == self.batch_size:
                    # print(batch)
                    yield batch
                    batch = []

        # batch = []
        # for i, (img, subj_id, time) in enumerate(self.data):
        #     batch.append(i)
        #     if len(batch) == self.batch_size
        #         yield batch
        #         batch = []
    def __len__(self):
        return len(self.subj_dict.keys())


class SubjectPerBatchSampler(Sampler):
    """
    For this sampler each batch contains only the data from a single subject.
    Subjects are not included if they have a specified minimum number of time points.

    Args:
        subject_dict: A dictionary containing {subject_id: [times]}
        batch_size: size of mini-batch
        min_data: Minimum number of data points for each unique subject.
    """

    def __init__(self, subject_dict, min_data=3):
        # build data sampling here
        self.subj_dict = subject_dict

    def __iter__(self):

        for subj in self.subj_dict.keys():
            batch = self.subj_dict[subj]
            yield batch

    def __len__(self):
        return len(self.subj_dict.keys())