from enum import Enum
import numpy as np
import heapq
import imgCrop


# enumeration for the orientations used to determine dissimilarity
# between pieces
class Orientations(Enum):
    left = 1
    right = 2
    up = 3
    down = 4


# returns opposite orientation for given orientation
def oppositeOrientation(orientation):
    if orientation == Orientations.right:
        return Orientations.left
    if orientation == Orientations.left:
        return Orientations.right
    if orientation == Orientations.up:
        return Orientations.down
    if orientation == Orientations.down:
        return Orientations.up


# returns the slices of the pieces that have to be used to calculate
# the dissimilarity
def slices(pi, pj,  orientation):
    K = pi.shape[0] - 1
    if orientation == Orientations.right:
        return pi[:, K, :], pi[:, (K-1), :], pj[:, 0, :]
    if orientation == Orientations.left:
        return pi[:, 0, :], pi[:, 1, :], pj[:, K, :]
    if orientation == Orientations.up:
        return pi[0, :, :], pi[1, :, :], pj[K, :, :]
    if orientation == Orientations.down:
        return pi[K, :, :], pi[(K-1), :, :], pj[0, :, :]


# given two pieces and the orientation (seen from the first piece)
# the dissmiliarity of the two pieces is returned
def dissmiliarity(pi, pj, orientation):
    slice1, slice2, slice3 = slices(pi, pj, orientation)
    return np.sum(np.abs((2 * slice1 - slice2) - slice3))


# returns the second best similarity for a given piece in the
# given orientation
def secondBestDissmilarity(pi, orientation, allPieces):
    allDissmiliarities = []
    # calculate dissmiliarity between all pieces
    for k in allPieces:
        # do not calculate dissmiliarity of piece to itself
        if not np.array_equal(k, pi):
            allDissmiliarities.append(dissmiliarity(pi, k, orientation))
    # return second smalles dissmiliarity
    return  heapq.nsmallest(2, allDissmiliarities)[-1]


# returns the compatibility between two pieces given the orientation and the
# second best dissmiliarity for the first piece
def compatibility(pi, pj, orientation, secondDissimilarity):
    if secondDissimilarity == 0:
        return 0
    dissimilarityPiPj = dissmiliarity(pi, pj, orientation)
    return 1 - (dissimilarityPiPj / secondDissimilarity)


# returns if two pieces are best buddies in the given orientation
def areBestBuddies(pi, pj, orientation, allPieces):
    # piece itself cannot be its own best buddy
    if np.array_equal(pi, pj):
        return False
    opposOrient = oppositeOrientation(orientation)
    secondBestDissPi = secondBestDissmilarity(pi, orientation, allPieces)
    secondBestDissPj = secondBestDissmilarity(pj, opposOrient, allPieces)
    compPiPj = compatibility(pi, pj, orientation, secondBestDissPi)
    compPjPi = compatibility(pj, pi, opposOrient, secondBestDissPj)
    for k in allPieces:
        if (not np.array_equal(k, pi)) and (not np.array_equal(k, pj)):
            if compatibility(pi, k, orientation, secondBestDissPi) > compPiPj:
                return False
            if compatibility(pj, k, opposOrient, secondBestDissPj) > compPjPi:
                return False
    return True


# returns best buddy for a given piece in the given direction or None
# if there is no best buddy in given orientation
def bestBuddy(pi, orientation, allPieces):
    for k in allPieces:
        if areBestBuddies(pi, k, orientation, allPieces):
            return k
    return None
