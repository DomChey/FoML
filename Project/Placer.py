import numpy as np
from compatibility import *
from imgCrop import *
from FindStartingPiece import *

def shiftImage(image, r):
    #Shift the image in direction r for one piece
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

    



def placer(pieces, imWidth, imHeight):
    # The placer algorithm processes all pieces from 'pieces' to the (hopefully) correct position in the image
    # Image dimension is (horizontalPieces, verticalPieces, Horizontal pixels in one piece, Vertical pixels in one piece, Color)

    pieces = np.array(pieces)
    dim = pieces.shape
    
    image = np.ones((imWidth//dim[1], imHeight//dim[2], dim[1], dim[2], dim[3]))
    notPlaced = set(pieces)
    pool = set()

    # get first piece and place it in the centre
    first = findFirstPiece(pieces)
    notPlaced = notPlaced - {first}
    addPiece(image, first, [dim[0]//2, dim[1]//2], Orientations.down)

    while len(notPlaced) != 0:
        temp = notPlaced.pop


    