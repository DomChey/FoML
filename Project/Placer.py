import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from queue import PriorityQueue
from compatibility import *
from imgCrop import *
from FindStartingPiece import *
import copy


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


def getPieceWithHighestCompatibility(allPieces):
    mutComp = []
    for piece in allPieces:
        buddies = getAllBuddies(piece, allPieces)
        mutComp.append(sum(mutualCompatibility(piece, buddies[key], key, allPieces) for key in buddies))

    return allPieces[mutComp.index(max(mutComp))]


def whereToPlaceNeighbor(piece, placerList):
    dissim = np.zeros((4, len(placerList)))
    for i, el in enumerate(placerList):
        dissim[0][i] = dissmiliarity(el[2], piece, Orientations.left)
        dissim[1][i] = dissmiliarity(el[2], piece, Orientations.right)
        dissim[2][i] = dissmiliarity(el[2], piece, Orientations.up)
        dissim[3][i] = dissmiliarity(el[2], piece, Orientations.down)
    return np.argwhere(dissim == np.min(dissim))[0]


def placer(pieces):
    # The placer algorithm processes all pieces from 'pieces' to the (hopefully) correct position in the image
    # Image dimension is (horizontalPieces, verticalPieces, Horizontal pixels in one piece, Vertical pixels in one piece, Color)

    unplacedPieces = pieces
    pool = PriorityQueue()
    placerList = []
    processedPieces = []
    takenIndices = []
    checkDuplicate = True

    # get first piece
    first = findFirstPiece(unplacedPieces)
    pool.put((0,0,1,Orientations.down,first))
    processedPieces.append(first)

    while unplacedPieces:
        while not pool.empty():
            item = pool.get()
            # Remove current item
            row, col = getPlacingPosition(item[3], item[1], item[2])
            if ((row, col) in takenIndices) and checkDuplicate:
                print("takenIndices")
                print(len(unplacedPieces))
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
            print("PoolWasEmpty")
            newfirst = getPieceWithHighestCompatibility(unplacedPieces)
            pos = whereToPlaceNeighbor(newfirst, placerList)
            pool.put((0, placerList[pos[1]][0], placerList[pos[1]][1], list(Orientations)[pos[0]], newfirst))
            processedPieces.append(newfirst)
            pieces = unplacedPieces
            checkDuplicate = False
#            item = pool.get()
#            row, col = getPlacingPosition(item[4], item[1], item[2])
#            placerList.append((row, col, item[3]))
#            unplacedPieces = [el for el in unplacedPieces if not el is item[3]]
#            takenIndices.append((row,col))
            
        elif len(unplacedPieces) == 1:
            print("LastPiece")
            pos = whereToPlaceNeighbor(unplacedPieces[0], placerList)
            pool.put((0, placerList[pos[1]][0], placerList[pos[1]][1], list(Orientations)[pos[0]], unplacedPieces[0]))
            item = pool.get()
            row, col = getPlacingPosition(item[3], item[1], item[2])
            placerList.append((row, col, item[4]))
            unplacedPieces = [el for el in unplacedPieces if not el is item[4]]
            takenIndices.append((row,col))

    print(len(pieces), len(processedPieces), len(unplacedPieces), len(placerList)) 
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
       # plt.imsave("results/{}_tmp.png".format(i), color.lab2rgb(image))
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
