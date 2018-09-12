"""
Methods to cut a given image into pieces

@author: Dominique Cheray and Manuel Krämer
"""

from skimage import io, color
from accessory import Piece
import numpy as np


def crop(im, height, width):
    """Crops a given image into pieces of given height and width

    Args:
        im:        Image to crop
        height:    Height of pieces to crop out of image
        width:     Width of the pieces to crop out of image

    Returns:
        Generator for the pieces that are cropped out of the image"""
    imgwidth, imgheight = im.shape[1], im.shape[0]
    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            yield im[(i*height):((i+1)*height), (j*width):((j+1)*width), :]


def cutIntoPieces(infile, height, width, normalize=True):
    """Cuts a given image into pieces of given height and width.
       Pieces are in YUV color space and the channels are normalized.

    Args:
        infile:    Path to the image to crop
        height:    Height of pieces to crop out of image
        width:     Width of the pieces to crop out of image
        normalize: Determines if the image should be normalized

    Returns:
        pieces:    A List containing the pieces of the image
    """

    pieces = []
    image = io.imread(infile)
    image = color.rgb2yuv(image)
    if normalize:
        image[:,:,0] = (image[:,:,0] - np.mean(image[:,:,0]))/np.std(image[:,:,0])
        image[:,:,1] = (image[:,:,1] - np.mean(image[:,:,1]))/np.std(image[:,:,1])
        image[:,:,2] = (image[:,:,2] - np.mean(image[:,:,2]))/np.std(image[:,:,2])

    for k, piece in enumerate(crop(image, height, width)):
        pieces.append(piece)
    return pieces


def createPieces(imfile, width, height, maxRow, maxCol, normalize=True):
    """Creates a list of Pieces from an image file which is cut into
       crops of width x height

    Args:
        imfile:    Path to the image to create the Pieces out of
        width:     Width of the image crops
        height:    Height of the image crops
        maxRow:    Max number of rows of crops the image will be cut into
        maxCol:    Max number of colums of crops the image will be cut into
        normalize: Determines if the image should be normalized before processing

    Returns:
        pieceList:    List containing the Pieces created out of the arrays"""

    pieces = cutIntoPieces(imfile, height, width, normalize)
    pieceList = []
    # maxCol+1 because maxCol and maxRow are set to their maximal numbers
        # when counting is started from zero. 
    totalNumCols = maxCol + 1
    # Create Piece objects from the original data
    for i,p in enumerate(pieces):
        row = int(np.floor(i/totalNumCols))
        col = i % totalNumCols
        
        if row == 0 and col == 0:
            neighborLeft = []
            neighborUp = []
            neighborDown = pieces[totalNumCols]
            neighborRight = pieces[i+1]
        elif row == 0 and col == maxCol:
            neighborRight = []
            neighborUp = []
            neighborDown = pieces[(row+1)*totalNumCols+col]
            neighborLeft = pieces[i - 1]
        elif row == maxRow and col == 0:
            neighborRight = pieces[i+1]
            neighborUp = pieces[(row-1)*totalNumCols+col]
            neighborDown = []
            neighborLeft = []
        elif row == maxRow and col == maxCol:
            neighborRight = []
            neighborUp = pieces[(row-1)*totalNumCols+col]
            neighborDown = []
            neighborLeft = pieces[i-1]
        elif col == 0:
            neighborLeft = []
            neighborUp = pieces[(row-1)*totalNumCols]
            neighborDown = pieces[(row+1)*totalNumCols]
            neighborRight = pieces[i+1]
        elif col == maxCol:
            neighborLeft = pieces[i-1]
            neighborUp = pieces[(row-1)*totalNumCols]
            neighborDown = pieces[(row+1)*totalNumCols]
            neighborRight = []
        elif row == 0:
            neighborRight = pieces[i+1]
            neighborUp = []
            neighborDown = pieces[(row+1)*totalNumCols+col]
            neighborLeft = pieces[i-1]
        elif row == maxRow:
            neighborRight = pieces[i+1]
            neighborUp = pieces[(row-1)*totalNumCols+col]
            neighborDown = []
            neighborLeft = pieces[i-1]
        else:
            neighborRight = pieces[i+1]
            neighborUp = pieces[(row-1)*totalNumCols+col]
            neighborDown = pieces[(row+1)*totalNumCols+col]
            neighborLeft = pieces[i-1]
        
        pi = Piece(p, neighborRight, neighborLeft, neighborUp, neighborDown)
        pieceList.append(pi)

    return pieceList

