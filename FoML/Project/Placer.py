import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from queue import PriorityQueue
from compatibility import *
from imgCrop import *
from FindStartingPiece import *
import math, random


def clearAllMemoizedFunctions():
    dissmiliarity.clearMemo()
    secondBestDissmilarity.clearMemo()
    areBestBuddies.clearMemo()
    bestBuddy.clearMemo()
    mutualCompatibility.clearMemo()
    hasFourBB.clearMemo()


def getAllBuddies(piece, allPieces):
    # returns a dictionary of all best buddies for given piece
    buddies = dict()
    for orientation in Orientations:
        tmp = bestBuddy(piece, orientation, allPieces)
        if tmp is not None:
            buddies[orientation] = tmp
    return buddies


def getPlacingPosition(orientation, x, y):
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
    for el in pool:
        if el is piece:
            return True
    return False


def getPieceWithHighestCompatibility(allPieces, unplacedPieces):
    if len(allPieces) < 2:
        return 0
    mutComp = []
    for piece in allPieces:
        buddies = getAllBuddies(piece, unplacedPieces)
        mutComp.append(sum(mutualCompatibility(piece, buddies[key], key, unplacedPieces) for key in buddies))

    return mutComp.index(max(mutComp))


def whereToPlaceNeighbor(piece, placerList, takenIndices, maxCol, maxRow):
    dissim = np.zeros((4, len(placerList)))
    minX = min(takenIndices)[0]
    maxX = max(takenIndices)[0]
    minY = min(takenIndices, key = lambda t: t[1])[1]
    maxY = max(takenIndices, key = lambda t: t[1])[1]
    for i, el in enumerate(placerList):
        if (el[0], (el[1] - 1)) in takenIndices or (el[1] - 1) < (maxY - maxCol):
            dissim[0][i] = math.inf
        else:
            dissim[0][i] = dissmiliarity(el[2], piece, Orientations.left) + dissmiliarity(piece, el[2], Orientations.right)
        if (el[0], el[1] + 1) in takenIndices or (el[1] + 1) > (minY + maxCol):
            dissim[1][i] = math.inf
        else: 
            dissim[1][i] = dissmiliarity(el[2], piece, Orientations.right) + dissmiliarity(piece, el[2], Orientations.left)
        if ((el[0] - 1), el[1]) in takenIndices or (el[0] - 1) < (maxX - maxRow):
            dissim[2][i] = math.inf
        else: 
            dissim[2][i] = dissmiliarity(el[2], piece, Orientations.up) + dissmiliarity(piece, el[2], Orientations.down)
        if ((el[0] + 1), el[1]) in takenIndices or (el[0] +1) > (minX + maxRow):
            dissim[3][i] = math.inf
        else:
            dissim[3][i] = dissmiliarity(el[2], piece, Orientations.down) + dissmiliarity(piece, el[2], Orientations.up)
    return np.argwhere(dissim == np.min(dissim))[0]


def getNextPiece(unplacedPieces, placerList, takenIndices, maxCol, maxRow):
    mutComp = np.zeros((4, len(placerList), len(unplacedPieces)))
    minX = min(takenIndices)[0]
    maxX = max(takenIndices)[0]
    minY = min(takenIndices, key = lambda t: t[1])[1]
    maxY = max(takenIndices, key = lambda t: t[1])[1]
    for i, el in enumerate(placerList):
        for j, piece in enumerate(unplacedPieces):
            if (el[0], (el[1] - 1)) in takenIndices or (el[1] - 1) < (maxY - maxCol):
                mutComp[0][i][j] = -math.inf
            else:
                mutComp[0][i][j] = mutualCompatibility(el[2], piece, Orientations.left, unplacedPieces)
            if (el[0], el[1] + 1) in takenIndices or (el[1] + 1) > (minY + maxCol):
                mutComp[1][i][j] = -math.inf
            else:
                mutComp[1][i][j] = mutualCompatibility(el[2], piece, Orientations.right, unplacedPieces)
            if ((el[0] - 1), el[1]) in takenIndices or (el[0] - 1) < (maxX - maxRow):
                mutComp[2][i][j] = -math.inf
            else:
                mutComp[2][i][j] = mutualCompatibility(el[2], piece, Orientations.up, unplacedPieces)
            if ((el[0] + 1), el[1]) in takenIndices or (el[0] +1) > (minX + maxRow):
                mutComp[3][i][j] = -math.inf
            else:
                mutComp[3][i][j] = mutualCompatibility(el[2], piece, Orientations.down, unplacedPieces)
    return np.argwhere(mutComp == np.max(mutComp))



def placer(pieces, maxCol, maxRow):
    # The placer algorithm processes all pieces from 'pieces' to the (hopefully) correct position in the image
    # Image dimension is (horizontalPieces, verticalPieces, Horizontal pixels in one piece, Vertical pixels in one piece, Color)

    unplacedPieces = pieces
    pool = PriorityQueue()
    placerList = []
    processedPieces = []
    takenIndices = []

    # get first piece
    first = findFirstPiece(unplacedPieces)
    pool.put((0,0,1,Orientations.down,first))
    processedPieces.append(first)

    while unplacedPieces:
        while not pool.empty():
            if takenIndices:
                minX = min(takenIndices)[0]
                maxX = max(takenIndices)[0]
                minY = min(takenIndices, key = lambda t: t[1])[1]
                maxY = max(takenIndices, key = lambda t: t[1])[1]
            item = pool.get()
            # Remove current item
            row, col = getPlacingPosition(item[3], item[1], item[2])
            if takenIndices:
                if ((row, col) in takenIndices) or (row < (maxX - maxRow)) or (row > (minX + maxRow)) or (col < (maxY - maxCol)) or (col > (minY + maxCol)):
                    processedPieces = [el for el in processedPieces if not el is item[4]]
                    continue
            placerList.append((row, col, item[4]))
            unplacedPieces = [el for el in unplacedPieces if not el is item[4]]
            takenIndices.append((row,col))
            bestBuddies = getAllBuddies(item[4], pieces)
       
            for key in bestBuddies:
                if isInPool(bestBuddies[key], processedPieces):
                    continue
                # *(-1) because priority queue returns smallest item
                mutComp = mutualCompatibility(item[4], bestBuddies[key], key, pieces) * -1
                processedPieces.append(bestBuddies[key])
                pool.put((mutComp, row, col, key, bestBuddies[key]))

        if len(unplacedPieces) > 1:
            clearAllMemoizedFunctions()
            pos = getNextPiece(unplacedPieces, placerList, takenIndices, maxCol, maxRow)
            nextPieces = [unplacedPieces[pos[i][2]] for i in range(len(pos))]
            best = getPieceWithHighestCompatibility(nextPieces, unplacedPieces)
            pool.put((0, placerList[pos[best][1]][0], placerList[pos[best][1]][1], list(Orientations)[pos[best][0]], unplacedPieces[pos[best][2]]))
            processedPieces.append(unplacedPieces[pos[best][2]])
            pieces = unplacedPieces
        elif len(unplacedPieces) == 1:
            pos = whereToPlaceNeighbor(unplacedPieces[0], placerList, takenIndices, maxCol, maxRow)
            row, col = getPlacingPosition(list(Orientations)[pos[0]], placerList[pos[1]][0], placerList[pos[1]][1])
            placerList.append((row, col, unplacedPieces[0]))
            unplacedPieces = [el for el in unplacedPieces if not el is unplacedPieces[0]]
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
    dim = pieces[1].shape
    cols = imWidth//dim[0]
    rows = imHeight//dim[1]
    
    image = np.ones(((rows)*dim[1], (cols)*dim[0], 3))
    
    for i,p in enumerate(pieces):
        image[(i//cols)*dim[1]:((i//cols)+1)*dim[1], (i%cols)*dim[0]:((i%cols)+1)*dim[0], :] = p
    
    return color.lab2rgb(image)
