#Assumptions:
#1) function compatibility(p1,p2,r) calculates and returns the compatibility score of p1 and p2 in direction r
#2) function bestBuddy(p,r) calculates and returns the best buddy (if existent) in direction r
#3) numpy array 'pieces' contains all pieces in a specific order


import numpy as np
from compatibility import *
from imgCrop import *
from accessory import *
from preprocessing import *


@Memoize
def mutualCompatibility(p1, p2, r, pieces):
    # Calculate the mutual compatibility of pieces p1 and p2 in direction r

    r1 = r
    r2 = oppositeOrientation(r)
    c1 = assymCompatibility(p1, p2, r1,secondBestDissmilarity(p1, r1, pieces))
    c2 = assymCompatibility(p2, p1, r2, secondBestDissmilarity(p2, r2, pieces))
    return ((c1+c2)/2)

@Memoize
def hasFourBB(x, pieces):
    for orientation in Orientations:
        if not hasDNNBuddy(x, orientation, pieces):
            return False
    return True


def findFirstPiece(pieces):
    # Find a piece that has best buddies in all four spatial dimensions
    # and maximizes the mutual compatibility

    distinctivePieces = list(filter(lambda x: hasFourBB(x, pieces), pieces))
    # print(distinctivePieces)

    piecesInDistinctiveRegion = []
    piecesInDistinctiveRegionBB = []

    for x in distinctivePieces:
        left = getDNNBuddy(x, Orientations.left, pieces)
        right = getDNNBuddy(x, Orientations.right, pieces)
        up = getDNNBuddy(x, Orientations.up, pieces)
        down = getDNNBuddy(x, Orientations.down, pieces)

        #if (any(left is x for x in distinctivePieces)) and (any(right is x for x in distinctivePieces)) and (any(up is x for x in distinctivePieces)) and (any(down is x for x in distinctivePieces)):

        if (hasFourBB(left, pieces) and hasFourBB(right, pieces) and hasFourBB(up, pieces) and hasFourBB(down, pieces)):
            piecesInDistinctiveRegion.append(x)
            piecesInDistinctiveRegionBB.append([left,right,up,down])


    mutualComp = [mutualCompatibility(x.data,piecesInDistinctiveRegionBB[i][0].data, Orientations.left, pieces)+
                  mutualCompatibility(x.data,piecesInDistinctiveRegionBB[i][1].data, Orientations.right, pieces)+
                  mutualCompatibility(x.data,piecesInDistinctiveRegionBB[i][2].data, Orientations.up, pieces)+
                  mutualCompatibility(x.data,piecesInDistinctiveRegionBB[i][3].data, Orientations.down, pieces)
                  for i, x in enumerate(piecesInDistinctiveRegion)]

    return piecesInDistinctiveRegion[np.argmax(mutualComp)]
