"""
Additional material needed for solving the puzzle

@author: Dominique Cheray and Manuel Krämer
"""


# required imports
from enum import IntEnum


class Orientations(IntEnum):
    """Enumeration for the orientations used to determine dissimilarity beteween pieces"""
    left = 1
    right = 2
    up = 3
    down = 4


def oppositeOrientation(orientation):
    """Function to get opposite Orientation of given Orientation"""
    if orientation == Orientations.right:
        return Orientations.left
    if orientation == Orientations.left:
        return Orientations.right
    if orientation == Orientations.up:
        return Orientations.down
    if orientation == Orientations.down:
        return Orientations.up


class Piece:
    """Class for the puzzle pieces. Each Piece knows its true neighbors it
       has in the original image and its neighbors in the reconstructed image"""

    def __init__(self, data, NeighborRight, NeighborLeft, NeighborUp, NeighborDown):
        #Initialize a piece with the image data and true neighboring pieces
        self.data = data
        self.trueNeighborRight = NeighborRight
        self.trueNeighborLeft = NeighborLeft
        self.trueNeighborUp = NeighborUp
        self.trueNeighborDown = NeighborDown
        # create variables that later hold the neighbors of this piece in the
        # reconstructed image
        self.NeighborRight = []
        self.NeighborLeft= []
        self.NeighborUp = []
        self.NeighborDown = []
