import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from queue import PriorityQueue
from compatibility import *
from imgCrop import *
from FindStartingPiece import *


def shiftImage(image, r):
    # Shift the image in direction r for one piece
    dim = image.shape
    if r == Orientations.up:
        insert = np.ones((dim[0],1,dim[2],dim[3],dim[4]))
        image = np.concatenate(image,insert, axis=1)
    elif r == Orientations.down:
        insert = np.ones((dim[0],1,dim[2],dim[3],dim[4]))
        image = np.concatenate(insert,image, axis=1)
    elif r == Orientations.left:
        insert = np.ones((1,dim[1],dim[2],dim[3],dim[4]))
        image = np.concatenate(image,insert, axis=0)
    elif r == Orientations.right:
        insert = np.ones((1,dim[1],dim[2],dim[3],dim[4]))
        image = np.concatenate(insert,image, axis=0)
    return image



def addPiece(image, piece, pos, r):
    # Adds a piece to the image. pos is the position where the new piece has to be attached, r is the orientation. 
    # If necessary the image get shifted (adding a new piece on the edge of the image)
    dim = image.shape
    if pos[0]==dim[0]:
        image = shiftImage(image, Orientations.left)
    elif pos[0]==0:
        image = shiftImage(image, Orientations.right)
    elif pos[1]==dim[1]:
        image = shiftImage(image, Orientations.up)
    elif pos[1]==0:
        image = shiftImage(image, Orientations.down)


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
        if np.array_equal(el, piece):
            return True
    return False



def placer(pieces):
    # The placer algorithm processes all pieces from 'pieces' to the (hopefully) correct position in the image
    # Image dimension is (horizontalPieces, verticalPieces, Horizontal pixels in one piece, Vertical pixels in one piece, Color)

    unplacedPieces = pieces
    pool = PriorityQueue()
    placerList = []
    processedPieces = []

    # get first piece
    first = findFirstPiece(unplacedPieces)
    unplacedPieces = [el for el in unplacedPieces if not np.array_equal(el, first)]
    pool.put((0,1,1,first))
    processedPieces.append(first)

    while not pool.empty():
        item = pool.get()
        # Remove current item
#        unplacedPieces = [el for el in unplacedPieces if not np.array_equal(el, item[3])]
        placerList.append((item[1], item[2], item[3]))
#        #Exit the loop if there are no more pieces to place
#        if len(unplacedPieces)==1:
#            last = pool.get()
#            placerList.append((last[1], last[2], last[3]))
#            continue
        bestBuddies = getAllBuddies(item[3], pieces)

        for key in bestBuddies:
            if isInPool(bestBuddies[key], processedPieces):
                continue
            row, col = getPlacingPosition(key, item[1], item[2])
            # *(-1) because priority queue returns smallest item
            mutComp = mutualCompatibility(item[3], bestBuddies[key], key, pieces) * -1
            unplacedPieces = [el for el in unplacedPieces if not np.array_equal(el, bestBuddies[key])]
            processedPieces.append(bestBuddies[key])
            pool.put((mutComp, row, col, bestBuddies[key]))

    return placerList


def showImage(sortedList):
    # Input: List containing tuples (row, col, Piece)
    # Shows the reconstructed image
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
    
    for p in sortedList:
        xpos = p[0]-min(row)
        ypos = p[1]-min(col)
        image[xpos*dim[0]:(xpos+1)*dim[0], ypos*dim[1]:(ypos+1)*dim[1],:] = p[2]
        #image[ypos*dim[1]:(ypos+1)*dim[1], xpos*dim[0]:(xpos+1)*dim[0],:] = p[2]
    
    plt.imshow(color.lab2rgb(image))
    plt.show()

pieces = cutIntoPieces("imData/16.png", 100, 100)
pieces = np.array(pieces)
np.random.shuffle(pieces)
pieces = list(pieces)
sort = placer(pieces)

#plt.imshow(color.lab2rgb(sort[-1][2]))
#plt.show()
showImage(sort)
