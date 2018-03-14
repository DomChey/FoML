import numpy as np
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
        newX = x+1
        newY = y
    elif orientation == Orientations.left:
        newX = x-1
        newY = y
    elif orientation == Orientations.up:
        newX = x
        newY = y-1
    elif orientation == Orientations.down:
        newX = x
        newY = y+1
    return newX, newY



def placer(pieces):
    # The placer algorithm processes all pieces from 'pieces' to the (hopefully) correct position in the image
    # Image dimension is (horizontalPieces, verticalPieces, Horizontal pixels in one piece, Vertical pixels in one piece, Color)

    unplacedPieces = pieces
    pool = PriorityQueue()
    placerList = []

    # get first piece
    first = findFirstPiece(unplacedPieces)
    unplacedPieces.remove(first)
    placerList.add((1,1, first))
    bestBuddies = getAllBuddies(first)
    for key in bestBuddies:
        newX, newY = getPlacingPosition(key, 1, 1)
        # +-1 because priority queue returns smallest item
        mutComp = mutualCompatibility(first, bestBuddies[key], key, unplacedPieces) * -1
        pool.put(mutComp, (newX, newY, bestBuddies[key]))


    while pool.not_empty:
        item = pool.get()
        print(item)





pieces = cutIntoPieces("imData/1.png", 50, 50)

placer(pieces)
