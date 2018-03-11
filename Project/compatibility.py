from enum import Enum
import numpy as np
import imgCrop


# enumeration for the orientations used to determine dissimilarity
# between pieces
class Orientations(Enum):
    left = 1
    right = 2
    up = 3
    down = 4


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

pieces = cutIntoPieces("imData/1.png", 50, 50)

dis = dissmiliarity(pieces[0], pieces[1], Orientations.down)
print(dis)
