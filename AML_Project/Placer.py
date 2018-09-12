import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from queue import PriorityQueue
from accessory import *
from compatibility import *
from imgCrop import *
from FindStartingPiece import *
import math, random


def clearAllMemoizedFunctions():
    """Clears the memo of all memoized functions. This is necesseray when
    pool is empty and compatibilities and Buddies ned to be recalculated."""
    getBuddiesNetPrediction.clearMemo()
    dissimilarity.clearMemo()
    compatibility.clearMemo()
    areDNNBuddies.clearMemo()
    getMostCompatiblePiece.clearMemo()
    getDNNBuddy.clearMemo()
    hasDNNBuddy.clearMemo()
    

def getAllBuddies(piece, allPieces):
    """Returns a dictionary containing all Buddies for a given piece found
    in the allPieces list"""
    buddies = dict()
    for orientation in Orientations:
        tmp = getDNNBuddy(piece, orientation, allPieces)
        if tmp is not None:
            buddies[orientation] = tmp
    return buddies


def getPlacingPosition(orientation, x, y):
    """Given the position of a piece and the orientation in which direction
    the neighboring piece should be placed the position for the neighboring
    piece is returned"""
    if orientation == Orientations.right:
        row = x
        col = y+1
    elif orientation == Orientations.left:
        row = x
        col = y-1
    elif orientation == Orientations.up:
        row = x-1
        col = y
    elif orientation == Orientations.down:
        row = x+1
        col = y
    return row, col


def isInPool(piece, pool):
    """Returns if a given piece is in the pool or not"""
    for el in pool:
        if el is piece:
            return True
    return False


def getPieceWithHighestCompatibility(allPieces, unplacedPieces):
    """Return which of the pieces has the highes compatibility among all its
    Buddies in the unplaced pieces"""
    # if there is only one piece left return index 0
    if len(allPieces) < 2:
        return 0
    comp = []
    # for every piece get its Buddies and calculate compatibility between them
    for piece in allPieces:
        buddies = getAllBuddies(piece, unplacedPieces)
        comp.append(sum(compatibility(piece.data, buddies[key].data, key) for key in buddies))
    # return index of piece that has the highest compatibility to all its buddies
    return comp.index(max(comp))


def whereToPlaceNeighbor(piece, placerList, takenIndices, maxCol, maxRow):
    """Returns for given piece the index of its neighbor and the index of the direction in which it should be places beside
    its neighbor"""
    # empty matrix for the for direction and all placed pieces
    dissim = np.zeros((4, len(placerList)))
    # determine mininmal and maximal x and y coordinates of all placed pieces
    minX = min(takenIndices)[0]
    maxX = max(takenIndices)[0]
    minY = min(takenIndices, key = lambda t: t[1])[1]
    maxY = max(takenIndices, key = lambda t: t[1])[1]
    # now for every piece and every orientation if the neighboring position to this piece in this direction is already taken or it would
    # exceed the limits of the puzzle, than the dissimilarity is infinitely high otherwise calculate the dissimilarity between the piece
    # to place and the potential neighbor and save it in the matrix
    for i, el in enumerate(placerList):
        if (el[0], (el[1] - 1)) in takenIndices or (el[1] - 1) < (maxY - maxCol):
            dissim[0][i] = math.inf
        else:
            dissim[0][i] = dissmiliarity(el[2].data, piece.data, Orientations.left)
        if (el[0], el[1] + 1) in takenIndices or (el[1] + 1) > (minY + maxCol):
            dissim[1][i] = math.inf
        else: 
            dissim[1][i] = dissmiliarity(el[2].data, piece.data, Orientations.right)
        if ((el[0] - 1), el[1]) in takenIndices or (el[0] - 1) < (maxX - maxRow):
            dissim[2][i] = math.inf
        else: 
            dissim[2][i] = dissmiliarity(el[2].data, piece.data, Orientations.up)
        if ((el[0] + 1), el[1]) in takenIndices or (el[0] +1) > (minX + maxRow):
            dissim[3][i] = math.inf
        else:
            dissim[3][i] = dissmiliarity(el[2].data, piece.data, Orientations.down)
    # return index of smallest dissimiliarity
    return np.argwhere(dissim == np.min(dissim))[0]


def getNextPiece(unplacedPieces, placerList, takenIndices, maxCol, maxRow):
    """Returns indices of unplaced piece, placed piece and orientation in which this unplaced piece has the highest
    compatibility to this placed piece"""
    # empty matrix to store the compatibilities
    comp = np.zeros((4, len(placerList), len(unplacedPieces)))
    minX = min(takenIndices)[0]
    maxX = max(takenIndices)[0]
    minY = min(takenIndices, key = lambda t: t[1])[1]
    maxY = max(takenIndices, key = lambda t: t[1])[1]
    # now for every placed calculate its compatibility to every unplaced piece in all orientations. If a placed piece has already
    # a neighbor in the given orientation or placing a piece beside it in the given orientation would exceed the limits of the puzzle
    # set the compatibility to an infinitely small number.
    for i, el in enumerate(placerList):
        for j, piece in enumerate(unplacedPieces):
            if (el[0], (el[1] - 1)) in takenIndices or (el[1] - 1) < (maxY - maxCol):
                comp[0][i][j] = -math.inf
            else:
                comp[0][i][j] = compatibility(el[2].data, piece.data, Orientations.left)
            if (el[0], el[1] + 1) in takenIndices or (el[1] + 1) > (minY + maxCol):
                comp[1][i][j] = -math.inf
            else:
                comp[1][i][j] = compatibility(el[2].data, piece.data, Orientations.right)
            if ((el[0] - 1), el[1]) in takenIndices or (el[0] - 1) < (maxX - maxRow):
                comp[2][i][j] = -math.inf
            else:
                comp[2][i][j] = compatibility(el[2].data, piece.data, Orientations.up)
            if ((el[0] + 1), el[1]) in takenIndices or (el[0] +1) > (minX + maxRow):
                comp[3][i][j] = -math.inf
            else:
                comp[3][i][j] = compatibility(el[2].data, piece.data, Orientations.down)
    # return index of unplaced piece, its neighboring placed piece and the direction in which it should be placed
    # if there are several pieces with the same compatibility they are all returned
    return np.argwhere(comp == np.max(comp))



def placer(pieces, maxCol, maxRow):
    # The placer algorithm processes all pieces from 'pieces' to the (hopefully) correct position in the image
    # Image dimension is (horizontalPieces, verticalPieces, Horizontal pixels in one piece, Vertical pixels in one piece, Color)

    # in the beginning all pieces are unplaced
    unplacedPieces = pieces
    # initialize empty pool
    pool = PriorityQueue()
    # initialize empty lists to store the placed pieces and their placing postion, which pieces are already processed and which positions
    # are already taken
    placerList = []
    processedPieces = []
    takenIndices = []

    # get first piece and put it on the pool
    first = findFirstPiece(unplacedPieces)
    pool.put((0,0,1,Orientations.down,first))
    processedPieces.append(first)

    # as long as there are unplaced pieces
    while unplacedPieces:
        # as long as the pool is not empty
        while not pool.empty():
            # in the beginning there are no taken indices since no piece was placed so far
            if takenIndices:
                # if there are taken indices get the minimum and maximum x and y coordinates
                minX = min(takenIndices)[0]
                maxX = max(takenIndices)[0]
                minY = min(takenIndices, key = lambda t: t[1])[1]
                maxY = max(takenIndices, key = lambda t: t[1])[1]
            # get next item from the pool item[0] = compatibility, item[1] = row its buddy is placed,
            # item[2] = row its buddy is place, item[3] = orientation in which it should be placed besides its buddy
            item = pool.get()
            # Get row and col where piece should be placed
            row, col = getPlacingPosition(item[3], item[1], item[2])
            if takenIndices:
                # if there are already placed pieces check if placing this piece would lead to place it at a place where another piece is already placed or
                # to place it outside the limits of the puzzle and if so remove it from the processed list, do not place it and go to get the next piece from the pool
                if ((row, col) in takenIndices) or (row < (maxX - maxRow)) or (row > (minX + maxRow)) or (col < (maxY - maxCol)) or (col > (minY + maxCol)):
                    processedPieces = [el for el in processedPieces if not el is item[4]]
                    continue
            # if placing postion os fine, store the piece and its position in the placer list
            placerList.append((row, col, item[4]))
            # remove it rom the unplaced pieces
            unplacedPieces = [el for el in unplacedPieces if not el is item[4]]
            # add its position to the taken indices
            takenIndices.append((row,col))
            # and get all its buddies
            bestBuddies = getAllBuddies(item[4], pieces)

            # now for all its buddies
            for key in bestBuddies:
                # if the Buddy was already put on the pool continue
                if isInPool(bestBuddies[key], processedPieces):
                    continue
                # calculate the compatibility of the piece and its buddy
                # *(-1) because priority queue returns smallest item
                comp = compatibility(item[4].data, bestBuddies[key].data, key) * -1
                # append the buddy to the processed pieces
                processedPieces.append(bestBuddies[key])
                # put the buddy on the pool together with the compatibility, the position of the piece and and
                # the orientation the buddy should be placed next to the piece
                pool.put((mutComp, row, col, key, bestBuddies[key]))

        # once pool is empty but there are more than 1 unplaced pieces
        if len(unplacedPieces) > 1:
            # clear memory of all memoized functions
            clearAllMemoizedFunctions()
            # get the indices of the pieces that have the highest compatibility to the placed pieces
            pos = getNextPiece(unplacedPieces, placerList, takenIndices, maxCol, maxRow)
            # filter all the pieces with highest compatibility
            nextPieces = [unplacedPieces[pos[i][2]] for i in range(len(pos))]
            # check which of these pieces has the highest compatibility to all its buddys amont the unplaced pieces
            best = getPieceWithHighestCompatibility(nextPieces, unplacedPieces)
            # put this piece on the pool
            pool.put((0, placerList[pos[best][1]][0], placerList[pos[best][1]][1], list(Orientations)[pos[best][0]], unplacedPieces[pos[best][2]]))
            # add it to the processed pieces
            processedPieces.append(unplacedPieces[pos[best][2]])
            # pieces are now all unplaced pieces
            pieces = unplacedPieces
        # once pool is empty and there is only one piece left
        elif len(unplacedPieces) == 1:
            # get index of neighbor and orientation where to place this piece among all placed pieces
            pos = whereToPlaceNeighbor(unplacedPieces[0], placerList, takenIndices, maxCol, maxRow)
            # caculate placing position for this piece
            row, col = getPlacingPosition(list(Orientations)[pos[0]], placerList[pos[1]][0], placerList[pos[1]][1])
            # add piece to the placer list
            placerList.append((row, col, unplacedPieces[0]))
            # remove it from the unplaced pieces
            unplacedPieces = [el for el in unplacedPieces if not el is unplacedPieces[0]]
            # add placing position to the taken positions
            takenIndices.append((row,col))
 
    return placerList


def getImage(sortedList):
    # Input: List containing tuples (row, col, Piece)
    # calculates the reconstructed image
    row = []
    col = []
    for p in sortedList:
        row.append(p[0])
        col.append(p[1])
    
    dim = sortedList[0][2].shape
    coldiff = max(col) - min(col)
    rowdiff = max(row) - min(row)
    image = np.ones((dim[0]*(rowdiff+1),dim[1]*(coldiff+1),3))
    #image = np.ones((dim[1]*(ydiff+1), dim[0]*(xdiff+1),3))

    i = 0
    for p in sortedList:
        xpos = p[0]-min(row)
        ypos = p[1]-min(col)
        image[xpos*dim[0]:(xpos+1)*dim[0], ypos*dim[1]:(ypos+1)*dim[1],:] = p[2]
        #image[ypos*dim[1]:(ypos+1)*dim[1], xpos*dim[0]:(xpos+1)*dim[0],:] = p[2]
        #plt.imsave("results/{}_tmp.png".format(i), color.lab2rgb(image))
        i = i+1
    
    return color.lab2rgb(image)


def getShuffledImage(pieces, imWidth, imHeight):
    # Input: List containing pieces, image width, image height
    # calculates the shuffled image
    dim = pieces[1].data.shape
    cols = imWidth//dim[0]
    rows = imHeight//dim[1]
    
    image = np.ones(((rows)*dim[1], (cols)*dim[0], 3))
    
    for i,p in enumerate(pieces):
        image[(i//cols)*dim[1]:((i//cols)+1)*dim[1], (i%cols)*dim[0]:((i%cols)+1)*dim[0], :] = p.data
    
    return color.yuv2rgb(image)
