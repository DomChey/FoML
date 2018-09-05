"""
Methods to cut a given image into pieces

@author: Dominique Cheray and Manuel Krämer
"""

from skimage import io, color
from accessory import Piece


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


def cutIntoPieces(infile, height, width):
    """Cuts a given image into pieces of given height and width.
       Pieces are in YUV color space and the channels are normalized.

    Args:
        infile:    Path to the image to crop
        height:    Height of pieces to crop out of image
        width:     Width of the pieces to crop out of image

    Returns:
        pieces:    A List containing the pieces of the image
    """

    pieces = []
    image = io.imread(infile)
    image = color.rgb2yuv(image)
    image[:,:,0] = (image[:,:,0] - np.mean(image[:,:,0]))/np.std(image[:,:,0])
    image[:,:,1] = (image[:,:,1] - np.mean(image[:,:,1]))/np.std(image[:,:,1])
    image[:,:,2] = (image[:,:,2] - np.mean(image[:,:,2]))/np.std(image[:,:,2])

    for k, piece in enumerate(crop(image, height, width)):
        pieces.append(piece)
    return pieces


def createPieces(arrayPieces):
    """Creates a list of Pieces out of a given list containint the
       arrays of the image pieces. Each Piece knows its true Neighbors
       in the original image

    Args:
        arrayPieces:    List containing the arrays of the pieces
                         cut out of the image

    Returns:
        pieceList:    List containing the Pieces created out of the arrays"""

    pieceList = []
    
    # Create Piece objects from the original data
    # Convention: x axis starts from the upper left corner to the right
    # y axis starts from the upper left corner downwards
    for i,p in enumerate(pieces):
        xpos = i%12
        ypos = int(np.floor(i/12))
        
        if xpos == 0 and ypos == 0:
            neighborLeft = []
            neighborUp = []
            neighborDown = pieces[(ypos+1)*12]
            neighborRight = pieces[xpos+1]
        elif xpos == 11 and ypos == 0:
            neighborRight = []
            neighborUp = []
            neighborDown = pieces[(ypos+1)*12+xpos]
            neighborLeft = pieces[xpos-1]
        elif xpos == 0 and ypos == 16:
            neighborRight = pieces[ypos*12]
            neighborUp = pieces[(ypos-1)*12]
            neighborDown = []
            neighborLeft = []
        elif xpos == 11 and ypos == 16:
            neighborRight = []
            neighborUp = pieces[(ypos-1)*12 + xpos]
            neighborDown = []
            neighborLeft = pieces[(ypos)*12 + xpos-1]
        elif xpos == 0:
            neighborLeft = []
            neighborUp = pieces[(ypos-1)*12]
            neighborDown = pieces[(ypos+1)*12]
            neighborRight = pieces[ypos*12 + xpos+1]
        elif xpos == 11:
            neighborLeft = pieces[ypos*12 + xpos-1]
            neighborUp = pieces[(ypos-1)*12]
            neighborDown = pieces[(ypos+1)*12]
            neighborRight = []
        elif ypos == 0:
            neighborRight = pieces[xpos+1]
            neighborUp = []
            neighborDown = pieces[(ypos+1)*12+xpos]
            neighborLeft = pieces[xpos-1]
        elif ypos == 16:
            neighborRight = pieces[ypos*12 + xpos+1]
            neighborUp = pieces[(ypos-1)*12+xpos]
            neighborDown = []
            neighborLeft = pieces[ypos*12 + xpos-1]
        else:
            neighborRight = pieces[ypos*12 + xpos+1]
            neighborUp = pieces[(ypos-1)*12 + xpos]
            neighborDown = pieces[(ypos+1)*12 + xpos]
            neighborLeft = pieces[ypos*12 + xpos-1]
        
        pi = Piece(p, neighborRight, neighborLeft, neighborUp, neighborDown)
        pieceList.append(pi)

    return pieceList

