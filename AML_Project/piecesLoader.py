"""
Custom datasets and custom dataloader functions for the puzzle pieces to train
the buddies net

@author: Dominique Cheray
"""
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import gzip


class PiecesDataTrain(Dataset):
    """Dataset for the training pieces for the buddies net"""

    def __init__(self):
        """Init pieces and labels"""
        self.X = []
        self.Y = []
        # load path to pieces and the labels
        pieces_file = open("pieces/train_pieces.txt")
        self.X = pieces_file.read().splitlines()
        pieces_file.close()
        labels_file = open("pieces/train_labels.txt")
        self.Y = labels_file.read().splitlines()
        labels_file.close()
       

    def __getitem__(self, index):
        """Override to give pytorch access to the pieces and labels"""
        # load the piece
        f = gzip.GzipFile(self.X[index], "r")
        piece = np.load(f)
        # flatten it
        piece = piece.flatten()
        # transform it to a Tensor
        piece = torch.from_numpy(piece).float()
        # convert label to Tensor
        label = torch.from_numpy(np.asarray(np.float32(self.Y[index]))).long()
        return piece, label


    def __len__(self):
        """Override to give pytorch the size of the dataset"""
        return len(self.X)


class PiecesDataTest(Dataset):
    """Dataset for the training pieces for the buddies net"""

    def __init__(self):
        """Init pieces and labels"""
        self.X = []
        self.Y = []
        # load path to pieces and the labels
        pieces_file = open("pieces/test_pieces.txt")
        self.X = pieces_file.read().splitlines()
        pieces_file.close()
        labels_file = open("pieces/test_labels.txt")
        self.Y = labels_file.read().splitlines()
        labels_file.close()


    def __getitem__(self, index):
        """Override to give pytorch access to the pieces and labels"""
        # load the piece
        f = gzip.GzipFile(self.X[index], "r")
        piece = np.load(f)
        # flatten it
        piece = piece.flatten()
        # transform it to a Tensor
        piece = torch.from_numpy(piece).float()
        # convert label to Tensor
        label = torch.from_numpy(np.asarray(np.float32(self.Y[index]))).long()
        return piece, label


    def __len__(self):
        """Override to give pytorch the size of the dataset"""
        return len(self.X)


def get_train_and_valid_loader(batch_size, use_cuda, random_seed=0,
                               valid_size=0.2, shuffle=True):
    """Usefull function that creates and returns train and validation dataloader
    for the dataset

    Args:
        batch_size:    desired batch size
        random_seed:   fix it for reproducability of the split into train and
                       validation set, default is 0 meaning he will not be fixed
        valid_size:    percentage of the data used for the validation set. Should
                       be a float between 1 and 0
        shuffle:       whether to shuffle the train/validation indices
        use_cuda:      whether cuda is available or not

    Returns:
        train_loader:  dataloader for the training set
        valid_loader:  dataloader for the validation set"""

    # load the dataset
    train_set = PiecesDataTrain()
    valid_set = PiecesDataTrain()
    # get size of dataset and use it to create indices for train and validation set
    num_train = len(train_set)
    indices = list(range(num_train))

    #determine where to split for validation set
    split_idx = (np.floor(valid_size * num_train)).astype(int)
    # shuffle indices if desired
    if shuffle:
        # if desired fix random seed
        if random_seed !=0:
            np.random.seed(random_seed)
        np.random.shuffle(indices)
    # now split the indices for train and validation sets
    train_indices = indices[split_idx:]
    valid_indices = indices[:split_idx]

    # create radom subsampler for validation and train set that use
    # the determined indices
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    # now create the two dataloaders and give them their respective
    # subsamplers
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=train_sampler, num_workers=1,
                              pin_memory=use_cuda)
    valid_loader = DataLoader(valid_set, batch_size=batch_size,
                              sampler=valid_sampler, num_workers=1,
                              pin_memory=use_cuda)

    return train_loader, valid_loader


def get_test_loader(batch_size, use_cuda, shuffle=True):
    """Usefull function that creates and returns a dataloader for the testset

    Args:
        batch_size:    desired batch size
        use_cuda:      whether cuda is available or not
        shuffle:       whether to shuffle the data or not

    Returns:
        test_loader:   dataloader for the test set"""

    # load the dataset
    test_set = PiecesDataTest()
    #create the dataloader
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle,
                             num_workers=1, pin_memory=use_cuda)
    return test_loader
