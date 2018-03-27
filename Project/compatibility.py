from enum import IntEnum
import numpy as np
import heapq
import imgCrop

EPSILON = 0.000001

# enumeration for the orientations used to determine dissimilarity
# between pieces
class Orientations(IntEnum):
    left = 1
    right = 2
    up = 3
    down = 4


class Memoize:
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
@Memoize
def dissmiliarity(pi, pj, orientation):
    slice1, slice2, slice3 = slices(pi, pj, orientation)
    dissim = np.sum(np.abs((2 * slice1 - slice2) - slice3))
    if dissim == 0.0:
        return EPSILON
    return dissim


# returns the second best similarity for a given piece in the
# given orientation
@Memoize
def secondBestDissmilarity(pi, orientation, allPieces):
#    if len(allPieces) < 2:
#        return 1
    allDissmiliarities = []
    # calculate dissmiliarity between all pieces
    for k in allPieces:
        # do not calculate dissmiliarity of piece to itself
        if not k is pi:
            allDissmiliarities.append(dissmiliarity(pi, k, orientation))
    # return second smalles dissmiliarity
    secondBest = heapq.nsmallest(2, allDissmiliarities)[-1]
    if secondBest == 0.0:
        return EPSILON
    return  secondBest


# returns the compatibility between two pieces given the orientation and the
# second best dissmiliarity for the first piece
@Memoize
def compatibility(pi, pj, orientation, secondDissimilarity):
    if secondDissimilarity == 0:
        secondDissimilarity = 0.000001
    dissimilarityPiPj = dissmiliarity(pi, pj, orientation)
    return 1 - (dissimilarityPiPj / secondDissimilarity)


# returns if two pieces are best buddies in the given orientation
@Memoize
def areBestBuddies(pi, pj, orientation, opposOrient,  allPieces, secondBestDissPi, secondBestDissPj, compPiPj, compPjPi):
    # piece itself cannot be its own best buddy
    if pi is pj:
        return False
    for k in allPieces:
        if (not k is pi) and (not k is pj):
            if compatibility(pi, k, orientation, secondBestDissPi) >= compPiPj:
                return False
            if compatibility(pj, k, opposOrient, secondBestDissPj) >= compPjPi:
                return False
    return True


# returns best buddy for a given piece in the given direction or None
# if there is no best buddy in given orientation
@Memoize
def bestBuddy(pi, orientation, allPieces):
    opposOrient = oppositeOrientation(orientation)
    secondBestDissPi = secondBestDissmilarity(pi, orientation, allPieces)
    for k in allPieces:
        if pi is k:
            continue
        else:
            secondBestDissPj = secondBestDissmilarity(k, opposOrient, allPieces)
            compPiPj = compatibility(pi, k, orientation, secondBestDissPi)
            compPjPi = compatibility(k, pi, opposOrient, secondBestDissPj)
        if areBestBuddies(pi, k, orientation, opposOrient, allPieces,
                          secondBestDissPi, secondBestDissPj, compPiPj, compPjPi):
            return k
    return None
