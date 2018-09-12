"""Functions needed to calculate the dissimilarity between two pieces

@author: Dominique Cheray and Manuel Krämer
"""


# Required imports
import numpy as np
import heapq
import imgCrop
from accessory import Orientations, oppositeOrientation
from buddiesNet import BuddiesNet
import torch
import torch.nn.functional as F
import math


class Memoize:
    """Class to memoize functions so as to avoid repeated calculations
       of the same calculations"""
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        ids = []
        for arg in args:
            ids.append((id(arg)))
        ids = tuple(ids)
        if not ids in self.memo:
            self.memo[ids] = self.f(*args)
        return self.memo[ids]

    def clearMemo(self):
        self.memo = {}


@Memoize
def getBuddiesNetPrediction(slice1, slice2, slice3, slice4):
    """"Returns the prediction of the buddies net if two pieces are neighbors
        or not given their borders"""
    # create model
    model = BuddiesNet()
    # load pretrained model
    ckpt = torch.load('ckpt.t7')
    model.load_state_dict(ckpt['model'])
    model.eval()
    # stack borders of the pieces together
    data = np.stack((slice2,slice1,slice3,slice4), axis=1)
    # flatten array 
    data = data.flatten()
    # transform it to a Tensor
    data = torch.from_numpy(data).float()
    # get output of model
    prediction = model(data)
    # softmax to get class prediction
    softmax = F.softmax(prediction, dim=0)
    # index of most likely class
    _, idx = softmax.max(0)
    # convert to numpy
#    idx = idx.cpu().numpy()[0]
    idx = idx.cpu().numpy()
    # and finally return the prediction
    return idx


def slices(pi, pj,  orientation):
    """Returns the slices of two given image that have to be used to caldulate
       the dissimilarity and are needed for the neural net to determine if two
       pieces are DNN-Buddies"""
    K = pi.shape[0] - 1
    if orientation == Orientations.right:
        return pi[:, K, :], pi[:, (K-1), :], pj[:, 0, :], pj[:,1,:]
    if orientation == Orientations.left:
        return pi[:, 0, :], pi[:, 1, :], pj[:, K, :], pj[:,(K-1),:]
    if orientation == Orientations.up:
        return pi[0, :, :], pi[1, :, :], pj[K, :, :], pj[(K-1),:,:]
    if orientation == Orientations.down:
        return pi[K, :, :], pi[(K-1), :, :], pj[0, :, :], pj[1,:,:]


@Memoize
def dissimilarity(pi, pj, orientation):
    """given two pieces and the orientation (seen from the first piece)
       the dissmiliarity of the two pieces is returned"""
    pi, pj = np.array(pi.data), np.array(pj.data)
    slice1, slice2, slice3, slice4 = slices(pi, pj, orientation)
    dissim = np.sqrt(np.sum((slice1 - slice3)**2))
    return dissim


@Memoize
def compatibility(pi, pj, orientation):
    """returns the compatibility between two pieces given the orientation
       as seen from the first piece"""
    dissimilarityPiPj = dissimilarity(pi, pj, orientation)
    return -dissimilarityPiPj


@Memoize
def areDNNBuddies(pi, pj, orientation):
    """Returns if two pieces are DNN-Buddies in the given orientation"""
    #piece itself can not be its own DNN-Buddy
    if np.array_equal(pi.data, pj.data):
        return False
    # get the borders of the two pieces
    s1, s2, s3, s4 = slices(pi.data, pj.data, orientation)
    # get the prediction of the BuddiesNet
    pred = getBuddiesNetPrediction(s1, s2, s3, s4)
    # 1 means they are DNN-Buddies
    if pred == 1:
        return True
    # 0 means they are not
    else:
        return False


@Memoize
def getMostCompatiblePiece(pi, orientation, allPieces):
    """Returns the most compatible Piece of the given Piece in the given
       orientation"""
    # array to store the compatibilities of all Pieces to the given piece
    compat = np.ones((len(allPieces)))
    for idx, k in enumerate(allPieces):
        # piece itself can not be its most compatible piece
        if pi is k:
            compat[idx] = -math.inf
        else:
            compat[idx] = compatibility(pi.data, k.data, orientation)
    # get index of highest compatibility
    maxCompIdx = np.argmax(compat)
    # return most compatible piece
    return allPieces[maxCompIdx]


@Memoize
def getDNNBuddy(pi, orientation, allPieces):
    """Returns the DNN-Buddy of the given Piece in the given direction or None
       if there is no DNN-Buddy in the given orientation"""
    # get most compatible piece
    mostCompatiblePiece = getMostCompatiblePiece(pi, orientation, allPieces)
    # check if piece and its most compatible piece are DNN-Buddies
    areBuddies = areDNNBuddies(pi, mostCompatiblePiece, orientation)
    # if they are DNN-Buddies return the most compatible piece
    if areBuddies:
        return mostCompatiblePiece
    # otherwise return None
    else:
        return None

@Memoize
def hasDNNBuddy(pi, orientation, allPieces):
    """Returns if the given Piece has a DNN-Buddy in the given orientation"""
    # get the most compatible Piece
    mostCompPiece = getMostCompatiblePiece(pi, orientation, allPieces)
    # if Piece and its most compatible piece are DNN-Buddies Piece has a DNN-Buddy
    return areDNNBuddies(pi, mostCompPiece, orientation)
